import os.path as osp
import os
from glob import glob
import time
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, drop_last, validation_split, num_workers, collate_fn):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_samples = len(dataset)
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'drop_last': self.drop_last,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        idx_full = np.arange(self.n_samples)
        np.random.seed(0)
        np.random.shuffle(idx_full)
        if isinstance(split, int):
            assert 0 < split < self.n_samples
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)
        valid_idx = idx_full[:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))
        self.shuffle = False
        self.n_samples = len(train_idx)
        return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)

import torch
from abc import abstractmethod
import shutil
import numpy as np
import json
from train_utils import extract_id_from_vertices_path, preprocess_kps_pad
from PIL import Image
import torchvision
from utils.utils_correspondence import co_pca, select_mutual_anchors
import tqdm

def log_sinkhorn_uot(Z, log_mu, log_nu, iters: int, alpha: float, beta: float, stabilize: bool = True):
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = alpha * (log_mu - torch.logsumexp(Z + v.unsqueeze(-2), dim=-1))
        v = beta * (log_nu - torch.logsumexp(Z + u.unsqueeze(-1), dim=-2))
        if stabilize:
            u = u - u.amax(dim=-1, keepdim=True)
            v = v - v.amax(dim=-1, keepdim=True)
    return Z + u.unsqueeze(-1) + v.unsqueeze(-2)

class MyBaseDataLoader(BaseDataLoader):
    """
    MyBaseDataLoader, collect all the commonly used member variables and member functions
    """
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 n_eig=100,
                 input_type = "xyz", # image_based -> xyz + RGB
                 descriptor=None, 
                 descriptor_dim=None, 
                 descriptor_dir=None,
                 shuffle=True, 
                 drop_last=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 base_input_dir="input/", 
                 training=True,
                 image_based=True,
                 confidence=0.9,
                 npoint=4196,
                 gpu_id=0,
                 cropped=False,
                 spectral_inputs=True,
                 pseudo_labeling_name='naive_OT',
                 resolution=840,
                 num_patches=60):

        self.dataset = dataset        
        self.n_eig = n_eig
        self.input_type = input_type        
        self.descriptor = descriptor
        self.descriptor_dim = descriptor_dim
        self.descriptor_dir = descriptor_dir
        self.image_based = image_based
        if self.image_based:
            self.split = self.dataset.mode
        else:
            self.split = self.dataset.split
        self.training = training
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        # self.image_based = image_based
        self.confidence = confidence
        self.npoint = npoint
        self.cropped = cropped
        self.spectral_inputs = spectral_inputs
        self.pseudo_labeling_name = pseudo_labeling_name
        self.resolution = resolution
        self.num_patches = num_patches

        if self.split != 'trn':
            batch_size = 1
        super().__init__(dataset, 
                         batch_size, 
                         shuffle, 
                         drop_last, 
                         validation_split, 
                         num_workers, 
                         collate_fn=self._custom_collate_fn)

        self.input_dir = base_input_dir
        if self.spectral_inputs:
            if not osp.exists(self.input_dir):
                print(f"input_dir {self.input_dir} not exists, start preprocess")
                self._preprocess()
            self.diffusion_dir = osp.join(self.input_dir, "diffusion")

            # Load all lists
            if not self.image_based:
                self.vertices_list = sorted(glob(f'{self.input_dir}/vertices/*.pt'))
                self.eVals_list = sorted(glob(f'{self.input_dir}/laplacian/eVals/*.pt'))
                self.eVecs_list  = sorted(glob(f'{self.input_dir}/laplacian/eVecs/*.pt'))
                self.Ls_list = sorted(glob(f'{self.input_dir}/laplacian/Ls/*.pt'))
                self.Ms_list = sorted(glob(f'{self.input_dir}/laplacian/Ms/*.pt'))
                self.desc_list = sorted(
                    glob(f'{self.input_dir}/descriptor/{self.descriptor}/*.pt')) if self.descriptor else None
                # ---diffusionNets---
                self.gradX_list = sorted(glob(f'{self.diffusion_dir}/gradX/*.pt'))
                self.gradY_list = sorted(glob(f'{self.diffusion_dir}/gradY/*.pt'))

        if self.image_based:
            self.pseudo_label_dir = osp.join(self.input_dir, self.pseudo_labeling_name)
            if not osp.exists(self.pseudo_label_dir):
                print(f"pseudo_label_dir {self.pseudo_label_dir} not exists, start preprocess")
                self._generate_pseudo_label()

    @abstractmethod
    def _custom_collate_fn(self, batch):
        '''
        virtual function, should be implemented in the derived class
        '''
        pass

    def _generate_pseudo_label(self):
        '''
        Generate pseudo labels for the dataset.
        '''
        os.makedirs(self.pseudo_label_dir, exist_ok=True)
        print("Generating pseudo labels with Fused GW refinement...")
        time_start = time.time()

        num_anchors = int(getattr(self, "num_anchors", 64))
        fusion_alpha = float(getattr(self, "alpha", 0.6))

        for _, off_path in enumerate(tqdm.tqdm(self.dataset.off_list)):
            print(f'{off_path}...', end='')
            file_names = os.path.basename(off_path).replace('.npz', '')
            category = off_path.split(os.sep)[-2]

            descriptors_collection = []
            masks_collection = []
            for i in range(2):
                desc_id = extract_id_from_vertices_path(file_names, i)
                _features = torch.load(
                    osp.join(self.dataset.data_dir, 'Features', f'{self.resolution}', category, f'{desc_id}.pt'),
                    map_location=self.device
                )
                if self.descriptor == 'sd_dino':
                    features = [_features['sd'], _features['dino']]
                else:
                    features = _features[self.descriptor].squeeze().reshape(self.num_patches, self.num_patches, -1)
                descriptors_collection.append(features)

                mask_dir = osp.join(self.dataset.data_dir, 'SAMMasks', category, f'{desc_id}_mask.png')
                mask_img = Image.open(mask_dir).convert("L")
                mask_tensor = torchvision.transforms.ToTensor()(mask_img)
                masks_collection.append(mask_tensor)

            if self.descriptor == 'sd_dino':
                features1 = descriptors_collection[0][0]
                features2 = descriptors_collection[1][0]
                processed_features1, processed_features2 = co_pca(features1, features2, [256, 256, 256])

                img1_desc_sd = processed_features1.reshape(1, 1, -1, self.num_patches ** 2).permute(0, 1, 3, 2)
                img2_desc_sd = processed_features2.reshape(1, 1, -1, self.num_patches ** 2).permute(0, 1, 3, 2)

                img1_desc_dino = descriptors_collection[0][1]
                img2_desc_dino = descriptors_collection[1][1]

                img1_desc_sd = img1_desc_sd / (img1_desc_sd.norm(dim=-1, keepdim=True) + 1e-8)
                img2_desc_sd = img2_desc_sd / (img2_desc_sd.norm(dim=-1, keepdim=True) + 1e-8)

                img1_desc_dino = img1_desc_dino / (img1_desc_dino.norm(dim=-1, keepdim=True) + 1e-8)
                img2_desc_dino = img2_desc_dino / (img2_desc_dino.norm(dim=-1, keepdim=True) + 1e-8)

                img1_desc = torch.cat((img1_desc_sd, img1_desc_dino), dim=-1).squeeze().reshape(self.num_patches, self.num_patches, -1)
                img2_desc = torch.cat((img2_desc_sd, img2_desc_dino), dim=-1).squeeze().reshape(self.num_patches, self.num_patches, -1)
                descriptor = torch.stack([img1_desc, img2_desc])
            else:
                descriptor = torch.stack(descriptors_collection)

            _, P, _, C = descriptor.shape
            masks = torch.nn.functional.interpolate(
                torch.stack(masks_collection),
                size=(self.num_patches, self.num_patches),
                mode='nearest'
            ).squeeze(1)
            src_mask, trg_mask = masks[0].bool(), masks[1].bool()

            masked_src_desc = descriptor[0].view(-1, C)[src_mask.view(-1)]
            masked_trg_desc = descriptor[1].view(-1, C)[trg_mask.view(-1)]
            if masked_src_desc.shape[0] == 0 or masked_trg_desc.shape[0] == 0:
                continue

            src_indices = torch.nonzero(src_mask.view(-1), as_tuple=False).squeeze(1)
            trg_indices = torch.nonzero(trg_mask.view(-1), as_tuple=False).squeeze(1)

            masked_src_desc = masked_src_desc / (masked_src_desc.norm(dim=-1, keepdim=True) + 1e-8)
            masked_trg_desc = masked_trg_desc / (masked_trg_desc.norm(dim=-1, keepdim=True) + 1e-8)
            sim_masked = masked_src_desc @ masked_trg_desc.t()

            scores = sim_masked.unsqueeze(0)
            b, m, n = scores.shape
            log_mu = scores.new_zeros((b, m))
            log_nu = scores.new_zeros((b, n))
            Z = log_sinkhorn_uot(
                scores, log_mu, log_nu,
                iters=10,
                alpha=0.75,
                beta=0.75
            ).squeeze(0)

            src_id = extract_id_from_vertices_path(file_names, 0)
            trg_id = extract_id_from_vertices_path(file_names, 1)

            pattern = f'{src_id}-{trg_id}'
            search_dirs = [osp.join(self.dataset.data_dir, 'PartialPCs', self.split, category)]
            if self.split == 'trn':
                search_dirs.append(osp.join(self.dataset.data_dir, 'PartialPCs', 'train', category))

            pc_files = []
            for search_dir in search_dirs:
                pc_files = glob(osp.join(search_dir, f"*{pattern}*.npz"))
                if pc_files:
                    break

            if not pc_files:
                os.makedirs(osp.join(self.pseudo_label_dir, category), exist_ok=True)
                torch.save(Z, osp.join(self.pseudo_label_dir, category, f'{src_id}-{trg_id}.pt'))
                continue

            vggt_output = np.load(pc_files[0])
            world_points = torch.tensor(vggt_output['world_points'], dtype=torch.float32, device=self.device)
            if world_points.ndim != 4 or world_points.shape[0] < 2:
                os.makedirs(osp.join(self.pseudo_label_dir, category), exist_ok=True)
                torch.save(Z, osp.join(self.pseudo_label_dir, category, f'{src_id}-{trg_id}.pt'))
                continue

            wp = world_points[:2].permute(0, 3, 1, 2)
            wp_resized = torch.nn.functional.interpolate(
                wp,
                size=(self.num_patches, self.num_patches),
                mode='bicubic',
                align_corners=False
            )
            vertices = wp_resized.permute(0, 2, 3, 1).reshape(2, -1, 3)

            img1_vertices = vertices[0][src_indices]
            img2_vertices = vertices[1][trg_indices]

            dists_s = torch.cdist(img1_vertices.float(), img1_vertices.float())
            dists_t = torch.cdist(img2_vertices.float(), img2_vertices.float())

            num_fgw_epochs = 5
            for _ in range(num_fgw_epochs):
                current_num_anchors = min(num_anchors, Z.numel())
                if current_num_anchors <= 0:
                    break
                anchors_s, anchors_t = select_mutual_anchors(Z, current_num_anchors)
                if anchors_s.numel() == 0 or anchors_t.numel() == 0:
                    break

                R_structure = torch.zeros_like(Z)
                for a_s, a_t in zip(anchors_s, anchors_t):
                    dist_s = dists_s[:, a_s].unsqueeze(1).expand(-1, Z.shape[1])
                    dist_t = dists_t[:, a_t].unsqueeze(0).expand(Z.shape[0], -1)
                    R_structure -= (dist_s - dist_t).abs()

                R_sem_norm = (sim_masked - sim_masked.min()) / (sim_masked.max() - sim_masked.min() + 1e-8)
                R_str_norm = (R_structure - R_structure.min()) / (R_structure.max() - R_structure.min() + 1e-8)
                R_fused = fusion_alpha * R_sem_norm + (1.0 - fusion_alpha) * R_str_norm

                R_fused_b = R_fused.unsqueeze(0)
                b, m, n = R_fused_b.shape
                log_mu = R_fused_b.new_zeros((b, m))
                log_nu = R_fused_b.new_zeros((b, n))

                Z = log_sinkhorn_uot(
                    R_fused_b, log_mu, log_nu,
                    iters=10,
                    alpha=0.75,
                    beta=0.75
                ).squeeze(0)

            os.makedirs(osp.join(self.pseudo_label_dir, category), exist_ok=True)
            torch.save(Z, osp.join(self.pseudo_label_dir, category, f'{src_id}-{trg_id}.pt'))

        print(f"Finished generating pseudo labels in {time.time() - time_start:.2f} seconds.")

    def _process_side(self, pc_data, side_index):
        world_points = pc_data['world_points'][side_index]
        depth_conf = pc_data['depth_conf'][side_index]

        world_points = torch.from_numpy(world_points).float().to(self.device)
        depth_conf = torch.from_numpy(depth_conf).float().to(self.device)

        valid_mask = (depth_conf > self.confidence)
        weights = depth_conf * valid_mask.float()
        world_points = world_points * valid_mask.float().unsqueeze(-1)

        wp_permuted = world_points.permute(2, 0, 1).unsqueeze(0)
        weights_permuted = weights.unsqueeze(0).unsqueeze(0)

        pool = torch.nn.AdaptiveAvgPool2d((self.num_patches, self.num_patches))

        numerator = pool(wp_permuted * weights_permuted)
        denominator = pool(weights_permuted)
        epsilon = 1e-8

        wp_resized = numerator / (denominator + epsilon)
        wp_final = wp_resized.squeeze(0).permute(1, 2, 0)
        final_points = wp_final.reshape(-1, 3)

        if final_points.shape[0] == 0:
            return torch.empty((0, 3), device=self.device), None

        final_points = pc_normalize(pc=final_points, as_numpy=False)
        return final_points, None

    def _save_operators_and_descriptors(self, pcs_tensor, base_name, idx, save_dirs, category=None):
        frames, M, L, eVal, eVec, gradX, gradY = get_operators(pcs_tensor, None, k=self.n_eig, cache_dir=None)
        eVal, eVec = eVal[1:], eVec[:, 1:]

        if category:
            prefix = osp.join(category, f'{base_name}_{idx}')
        else:
            prefix = f'{base_name}_{idx}'

        torch.save(pcs_tensor, osp.join(save_dirs['vertices'], f'vertices_{prefix}.pt'))
        torch.save(eVal.squeeze(), osp.join(save_dirs['eVals'], f'eVal_{prefix}.pt'))
        torch.save(eVec, osp.join(save_dirs['eVecs'], f'eVec_{prefix}.pt'))
        torch.save(L, osp.join(save_dirs['Ls'], f'L_{prefix}.pt'))
        torch.save(M, osp.join(save_dirs['Ms'], f'M_{prefix}.pt'))
        torch.save(gradX, osp.join(save_dirs['gradX'], f'gradX_{prefix}.pt'))
        torch.save(gradY, osp.join(save_dirs['gradY'], f'gradY_{prefix}.pt'))

        if self.descriptor == "hks":
            hks = compute_hks_autoscale(eVal.unsqueeze(0), eVec.unsqueeze(0), self.descriptor_dim).squeeze()
            torch.save(hks, osp.join(save_dirs['descriptor'], f'{self.descriptor}_{prefix}.pt'))
        elif self.descriptor == "deep":
            if os.path.isdir(self.descriptor_dir):
                for file_name in os.listdir(self.descriptor_dir):
                    src_file_path = os.path.join(self.descriptor_dir, file_name)
                    dst_file_path = os.path.join(save_dirs['descriptor'], file_name)
                    shutil.copy(src_file_path, dst_file_path)
            else:
                print(f"The specified path {self.descriptor_dir} is not a directory.")
        elif self.descriptor is not None and self.descriptor != "hks":
            raise ValueError(f"Invalid descriptor type: {self.descriptor}")

    def _preprocess(self):
        '''
        When no input presented, pre-calculate all needed laplacians and descriptors.
        '''
        print("Pre-calculating...")
        time_start = time.time()

        save_vertices_dir = osp.join(self.input_dir, "vertices");
        os.makedirs(save_vertices_dir, exist_ok=True)
        save_evals_dir = osp.join(self.input_dir, "laplacian/eVals");
        os.makedirs(save_evals_dir, exist_ok=True)
        save_evecs_dir = osp.join(self.input_dir, "laplacian/eVecs");
        os.makedirs(save_evecs_dir, exist_ok=True)
        save_Ls_dir = osp.join(self.input_dir, "laplacian/Ls");
        os.makedirs(save_Ls_dir, exist_ok=True)
        save_Ms_dir = osp.join(self.input_dir, "laplacian/Ms");
        os.makedirs(save_Ms_dir, exist_ok=True)
        diffusion_dir = osp.join(self.input_dir, "diffusion");
        os.makedirs(diffusion_dir, exist_ok=True)
        save_gradX_dir = osp.join(diffusion_dir, "gradX");
        os.makedirs(save_gradX_dir, exist_ok=True)
        save_gradY_dir = osp.join(diffusion_dir, "gradY");
        os.makedirs(save_gradY_dir, exist_ok=True)

        save_desc_dir = None
        if self.descriptor:
            save_desc_dir = osp.join(self.input_dir, "descriptor", self.descriptor)
            os.makedirs(save_desc_dir, exist_ok=True)

        save_dirs = {
            "vertices": save_vertices_dir,
            "eVals": save_evals_dir,
            "eVecs": save_evecs_dir,
            "Ls": save_Ls_dir,
            "Ms": save_Ms_dir,
            "gradX": save_gradX_dir,
            "gradY": save_gradY_dir,
            "descriptor": save_desc_dir
        }

        for _, off_path in enumerate(self.dataset.off_list):
            print(f'{off_path}...', end='')

            if self.image_based:
                current_category = off_path.split(os.sep)[-2]
                for d in save_dirs.values():
                    if d:
                        os.makedirs(osp.join(d, current_category), exist_ok=True)

                pc_data = np.load(off_path)
                base_name = osp.basename(off_path).replace('.npz', '')

                src_pc, _ = self._process_side(pc_data, 0)
                trg_pc, _ = self._process_side(pc_data, 1)

                for i, pcs_tensor in enumerate([src_pc, trg_pc]):
                    self._save_operators_and_descriptors(
                        pcs_tensor, base_name, i, save_dirs, category=current_category
                    )
                    
            else:
                idx = extract_number_from_filename(off_path)
                pcs, faces = read_shape(off_path)
                pcs = pc_normalize(pcs, faces)
                pcs_tensor = torch.tensor(pcs, dtype=torch.float32).to(self.device)

                base_name = f"{idx:03d}"
                self._save_operators_and_descriptors(pcs_tensor, base_name, 0, save_dirs)

            print('DONE')
        print("Pre-calculating finished. Time consumed in second: ", time.time() - time_start)

    def pad_data(self, data, max_length, padding_value=0):
        if not self.training:
            return data
        current_length = data.shape[0]

        # eVals, Ms
        if len(data.shape) == 1:
            if current_length < max_length:
                pad_size = max_length - current_length
                data = torch.cat(
                    [data, torch.full((pad_size,), padding_value, dtype=data.dtype, device=data.device)], dim=0)

        elif len(data.shape) == 2:
            if data.shape[1] == 2 or data.shape[1] == 3:
                pad_size = max_length - current_length
                data = torch.cat([data, torch.full((pad_size, data.shape[1]), padding_value, dtype=data.dtype,
                                                   device=data.device)], dim=0)
            else:  # Ls, gradXs, gradYs
                pad_size = max_length - current_length
                data = torch.cat([data, torch.full((pad_size, data.shape[1]), padding_value, dtype=data.dtype,
                                                   device=data.device)], dim=0)
                data = torch.cat(
                    [data, torch.full((max_length, pad_size), padding_value, dtype=data.dtype, device=data.device)],
                    dim=1)

        elif len(data.shape) == 3:
            current_length = data.shape[1]
            pad_size = max_length - current_length
            data = torch.cat([data, torch.full((2, pad_size, 2), padding_value, dtype=data.dtype,
                                               device=data.device)], dim=1)
        return data

class SPair_DataLoader(MyBaseDataLoader):
    """
    DataLoader for image-based SPair-71k dataset.
    All point clouds are sampled to a fixed number of points.
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 n_eig=100,
                 input_type="xyz",
                 descriptor=None,
                 descriptor_dim=None,
                 descriptor_dir=None,
                 shuffle=False,
                 drop_last=False,
                 validation_split=0.0,
                 num_workers=1,
                 base_input_dir="input/",
                 training=True,
                 image_based=True,
                 confidence=0.9,
                 npoint=4196,
                 gpu_id=0,
                 cropped=False,
                 spectral_inputs=True,
                 pseudo_labeling_name='naive_OT',
                 resolution=840,
                 return_target_mask=False,
                 sampling=0,
                 bidirectional=False,
                 bidirectional_mode='eucl_dist_2d',
                 bidirectional_threshold=2.0,
                 bidirectional_symmetric=False,
                 label_update=False,
                 top_k=1):

        # self.resolution = resolution
        num_patches = int(resolution / 14)
        super().__init__(dataset,
                         batch_size,
                         n_eig,
                         input_type,
                         descriptor,
                         descriptor_dim,
                         descriptor_dir,
                         shuffle,
                         drop_last,
                         validation_split,
                         num_workers,
                         base_input_dir,
                         training,
                         image_based,
                         confidence,
                         npoint,
                         gpu_id,
                         cropped,
                         spectral_inputs,
                         pseudo_labeling_name,
                         resolution,
                         num_patches)
        self.return_target_mask = return_target_mask
        self.sampling = sampling
        self.bidirectional = bidirectional
        self.bidirectional_mode = bidirectional_mode
        self.bidirectional_threshold = bidirectional_threshold
        self.bidirectional_symmetric = bidirectional_symmetric
        self.label_update = label_update
        self.top_k = top_k
        self.enable_cache = False
        self.preload_pl_coords = False
        self._pl_coords_cache = {}
        self._masks_cache = {}

        if self.enable_cache and self.preload_pl_coords and (self.split == 'trn') and (not self.spectral_inputs):
            self._preload_pseudo_label_coords()

    def _get_cached_masks(self, category, file_names, src_id, trg_id):
        mkey = f"{category}/{file_names}|P={self.num_patches}"
        if mkey in self._masks_cache:
            return self._masks_cache[mkey]
        self._get_cached_pseudo_label_coords(
            category=category,
            file_names=file_names,
            src_id=src_id,
            trg_id=trg_id,
            bidirectional=self.bidirectional,
            top_k=self.top_k,
            bidirectional_mode=self.bidirectional_mode,
            bidirectional_threshold=self.bidirectional_threshold,
            bidirectional_symmetric=self.bidirectional_symmetric
        )
        return self._masks_cache[mkey]

    def _get_cached_pseudo_label_coords(self, category, file_names, src_id, trg_id, bidirectional=False, top_k=1,
                                        bidirectional_mode='eucl_dist_2d', bidirectional_threshold=2.0,
                                        bidirectional_symmetric=False):
        key = f"{category}/{file_names}|bi={int(bidirectional)}|P={self.num_patches}|up={int(self.label_update)}|k={top_k}|mode={bidirectional_mode}|th={bidirectional_threshold}|sym={int(bidirectional_symmetric)}"
        if key in self._pl_coords_cache:
            return self._pl_coords_cache[key]

        import os.path as osp
        import torch.nn.functional as F
        from glob import glob
        import numpy as np

        P = self.num_patches
        device = self.device if torch.cuda.is_available() else "cpu"

        # -------------------------------------------------------------------------
        # 1) Masks (Always fetched first)
        # -------------------------------------------------------------------------
        src_mask_img = Image.open(osp.join(self.dataset.data_dir, 'SAMMasks', category, f'{src_id}_mask.png')).convert("L")
        trg_mask_img = Image.open(osp.join(self.dataset.data_dir, 'SAMMasks', category, f'{trg_id}_mask.png')).convert("L")

        src_mask = torchvision.transforms.ToTensor()(src_mask_img).unsqueeze(0).to(device)
        trg_mask = torchvision.transforms.ToTensor()(trg_mask_img).unsqueeze(0).to(device)

        with torch.no_grad():
            src_mask = F.interpolate(src_mask, size=(P, P), mode="nearest").squeeze(0).bool()
            trg_mask = F.interpolate(trg_mask, size=(P, P), mode="nearest").squeeze(0).bool()

        # Update Mask Cache (if needed by other methods)
        mkey = f"{category}/{file_names}|P={P}"
        if mkey not in self._masks_cache:
            self._masks_cache[mkey] = (src_mask.detach().cpu(), trg_mask.detach().cpu())

        # -------------------------------------------------------------------------
        # 2) Branching based on self.label_update
        # -------------------------------------------------------------------------
        if self.label_update:
            # === [Branch A] Load 3D Vertices from Point Cloud ===
            pattern = f'{src_id}-{trg_id}'
            search_dir = osp.join(self.dataset.data_dir, 'PartialPCs', self.split, category)
            pc_files = glob(osp.join(search_dir, f"*{pattern}*.npz"))

            if len(pc_files) > 0:
                pc_file_path = pc_files[0]
                vggt_output = np.load(pc_file_path)
                
                # Load World Points: Assumed shape (2, H, W, 3) or similar containing pair
                world_points = torch.tensor(vggt_output['world_points'], dtype=torch.float32, device=device)
                
                # Permute for interpolation: (B, H, W, 3) -> (B, 3, H, W)
                if world_points.ndim == 4:
                    wp = world_points.permute(0, 3, 1, 2)
                else:
                    # Single item case fallback
                    wp = world_points.unsqueeze(0).permute(0, 3, 1, 2)

                with torch.no_grad():
                    wp_resized = F.interpolate(wp, size=(P, P), mode='bicubic', align_corners=False)
                
                # Reshape back to (B, P*P, 3)
                vertices = wp_resized.permute(0, 2, 3, 1).reshape(wp.shape[0], -1, 3)
                src_coords = vertices[0]
                trg_coords = vertices[1]
            else:
                # Handle case where file is missing
                src_coords = torch.empty((0, 3), device=device)
                trg_coords = torch.empty((0, 3), device=device)

            # Return 3D vertices
            # For label_update=True, each point is a unique source (no top-k expansion)
            group_ids = torch.arange(src_coords.shape[0], device=device)
            self._pl_coords_cache[key] = (src_coords.detach().cpu(), trg_coords.detach().cpu(), group_ids.detach().cpu())
            return self._pl_coords_cache[key]

        else:
            # === [Branch B] Load OT Plan (Original Logic) ===
            ot_path = f'{self.pseudo_label_dir}/{category}/{src_id}-{trg_id}.pt'
            ot_plan = torch.load(ot_path, map_location="cpu").to(device)

            # Generate grid coords
            grid_y, grid_x = torch.meshgrid(
                torch.arange(P, device=device),
                torch.arange(P, device=device),
                indexing='ij'
            )
            coords = torch.stack([grid_x, grid_y], dim=-1) # (P,P,2)
            
            # Filter grid coords by mask
            src_coords_all = coords.view(-1, 2)[src_mask.view(-1)]
            trg_coords_all = coords.view(-1, 2)[trg_mask.view(-1)]

            # Top-k matches
            actual_k = min(top_k, ot_plan.shape[1])  # 실제 사용 가능한 k 값으로 제한

            if top_k == 1:
                # Keep original behavior for backward compatibility
                best_match_indices_for_src = torch.argmax(ot_plan, dim=1)  # (N_src,)
            else:
                # Get top-k candidates
                topk_result = torch.topk(ot_plan, k=actual_k, dim=1)
                best_match_indices_for_src = topk_result.indices  # (N_src, actual_k)
                topk_values = topk_result.values  # (N_src, actual_k) - for potential weighting

            if not bidirectional:
                if top_k == 1:
                    src_coords = src_coords_all
                    trg_coords_mapped = trg_coords_all[best_match_indices_for_src]
                    # Each point is a unique source
                    N = src_coords.shape[0]
                    group_ids = torch.arange(N, device=device)
                else:
                    # Expand source coordinates k times: [s1, s1, ..., s1, s2, s2, ..., s2, ...]
                    N = src_coords_all.shape[0]
                    src_coords_expanded = src_coords_all.unsqueeze(1).repeat(1, actual_k, 1)  # (N_src, actual_k, 2)
                    src_coords = src_coords_expanded.reshape(-1, 2)  # (N_src*actual_k, 2)

                    # Map target coordinates for all k candidates
                    trg_coords_mapped = trg_coords_all[best_match_indices_for_src]  # (N_src, actual_k, 2)
                    trg_coords_mapped = trg_coords_mapped.reshape(-1, 2)  # (N_src*actual_k, 2)

                    # Group IDs: [0,0,...,0, 1,1,...,1, ..., N-1,N-1,...,N-1]
                    group_ids = torch.arange(N, device=device).unsqueeze(1).repeat(1, actual_k).reshape(-1)  # (N*actual_k,)
            else:
                # Bidirectional consistency check
                N = ot_plan.shape[0]

                # Load 3D point cloud data only for modes that need it
                if bidirectional_mode in ['eucl_dist_3d', 'quantile']:
                    pc_file_path = osp.join(self.dataset.data_dir, 'PartialPCs', self.split, category, f'{file_names}.npz')
                    vggt_output = np.load(pc_file_path)
                    wp = torch.tensor(vggt_output['world_points'][0], dtype=torch.float32, device=device)

                    with torch.no_grad():
                        wp_bchw = wp.permute(2, 0, 1).unsqueeze(0)
                        wp_resized = F.interpolate(wp_bchw, size=(P, P), mode="bicubic", align_corners=False).squeeze(0)

                    vertices_grid = wp_resized.permute(1, 2, 0).reshape(-1, 3).contiguous()
                    vertices_s = vertices_grid[src_mask.view(-1)]
                else:
                    # For 'strict' and 'eucl_dist_2d' modes, 3D data is not needed
                    vertices_s = None

                if top_k == 1:
                    # Bidirectional filtering for k=1
                    argmax_t = best_match_indices_for_src
                    argmax_s = ot_plan.argmax(dim=0)

                    i_idx = torch.arange(N, device=device)
                    j_idx = argmax_t
                    i_prime = argmax_s[j_idx]

                    # Mode-based branching
                    if bidirectional_mode == 'strict':
                        # Exact match
                        mask = (i_idx == i_prime)

                    elif bidirectional_mode == 'eucl_dist_2d':
                        # 2D grid coordinate distance
                        src_kps = src_coords_all[i_idx]
                        src_kps_cyclic = src_coords_all[i_prime]
                        dists = (src_kps.float() - src_kps_cyclic.float()).norm(dim=-1)
                        mask = dists <= bidirectional_threshold

                    elif bidirectional_mode == 'eucl_dist_3d':
                        # 3D world coordinate distance
                        dists = (vertices_s[i_idx] - vertices_s[i_prime]).norm(dim=-1)
                        mask = dists <= bidirectional_threshold

                    else:  # 'quantile' (original method)
                        # 3D distance with dynamic quantile
                        dists = (vertices_s[i_idx] - vertices_s[i_prime]).norm(dim=-1)

                        q = 0.01
                        if torch.isnan(dists).all():
                            mask = torch.ones_like(i_idx, dtype=torch.bool)
                        else:
                            eps = torch.quantile(dists[~torch.isnan(dists)], q)
                            mask = dists <= eps

                        if mask.sum() == 0 and N > 0:
                            new_q = min(q * 2, 0.5)
                            non_nan_dists = dists[~torch.isnan(dists)]
                            if non_nan_dists.numel() > 0:
                                eps = torch.quantile(non_nan_dists, new_q)
                                mask = dists <= eps
                            else:
                                mask = torch.ones_like(i_idx, dtype=torch.bool)

                    src_coords = src_coords_all[mask]
                    trg_coords_mapped = trg_coords_all[best_match_indices_for_src[mask]]
                    # Each filtered point is a unique source
                    M = mask.sum().item()
                    group_ids = torch.arange(M, device=device)
                else:
                    # Top-k logic with bidirectional filtering
                    # For each source point, we have k target candidates
                    # We need to check consistency for each (src_i, trg_j) pair independently

                    # Expand to check all N*actual_k pairs
                    # best_match_indices_for_src: (N, actual_k)
                    src_indices_flat = torch.arange(N, device=device).unsqueeze(1).repeat(1, actual_k).reshape(-1)  # (N*actual_k,)
                    trg_indices_flat = best_match_indices_for_src.reshape(-1)  # (N*actual_k,)

                    if not bidirectional_symmetric:
                        # Asymmetric: backward uses argmax (top-1)
                        argmax_s = ot_plan.argmax(dim=0)  # (N_trg,)
                        # For each pair (src_i, trg_j), check: src_i -> trg_j -> src_i'
                        i_prime_flat = argmax_s[trg_indices_flat]  # (N*actual_k,)
                    else:
                        # Symmetric: backward also uses top-k
                        # For dim=0, k must be <= N_src
                        actual_k_reverse = min(actual_k, ot_plan.shape[0])
                        topk_s = ot_plan.topk(actual_k_reverse, dim=0).indices  # (actual_k_reverse, N_trg)
                        # For each pair (src_i, trg_j), get top-k sources for trg_j
                        topk_sources_for_trg = topk_s[:, trg_indices_flat]  # (actual_k_reverse, N*actual_k)

                    # Mode-based branching
                    if not bidirectional_symmetric:
                        # Asymmetric: compare with single backward match
                        if bidirectional_mode == 'strict':
                            # Exact match
                            mask_flat = (src_indices_flat == i_prime_flat)

                        elif bidirectional_mode == 'eucl_dist_2d':
                            # 2D grid coordinate distance
                            src_kps = src_coords_all[src_indices_flat]      # (N*actual_k, 2)
                            src_kps_cyclic = src_coords_all[i_prime_flat]   # (N*actual_k, 2)
                            dists_flat = (src_kps.float() - src_kps_cyclic.float()).norm(dim=-1)
                            mask_flat = dists_flat <= bidirectional_threshold

                        elif bidirectional_mode == 'eucl_dist_3d':
                            # 3D world coordinate distance
                            dists_flat = (vertices_s[src_indices_flat] - vertices_s[i_prime_flat]).norm(dim=-1)
                            mask_flat = dists_flat <= bidirectional_threshold

                        else:  # 'quantile' (original method)
                            # 3D distance with dynamic quantile
                            dists_flat = (vertices_s[src_indices_flat] - vertices_s[i_prime_flat]).norm(dim=-1)

                            q = 0.01
                            if torch.isnan(dists_flat).all():
                                mask_flat = torch.ones_like(dists_flat, dtype=torch.bool)
                            else:
                                eps = torch.quantile(dists_flat[~torch.isnan(dists_flat)], q)
                                mask_flat = dists_flat <= eps

                            if mask_flat.sum() == 0 and dists_flat.numel() > 0:
                                new_q = min(q * 2, 0.5)
                                non_nan_dists = dists_flat[~torch.isnan(dists_flat)]
                                if non_nan_dists.numel() > 0:
                                    eps = torch.quantile(non_nan_dists, new_q)
                                    mask_flat = dists_flat <= eps
                                else:
                                    mask_flat = torch.ones_like(dists_flat, dtype=torch.bool)
                    else:
                        # Symmetric: check if src_i is in top-k of trg_j
                        if bidirectional_mode == 'strict':
                            # Exact match: src_i must be in topk_sources_for_trg
                            # topk_sources_for_trg: (actual_k, N*actual_k)
                            # src_indices_flat: (N*actual_k,)
                            mask_flat = (topk_sources_for_trg == src_indices_flat.unsqueeze(0)).any(dim=0)

                        elif bidirectional_mode == 'eucl_dist_2d':
                            # 2D distance: check if any of the top-k sources is close enough
                            src_kps = src_coords_all[src_indices_flat]  # (N*actual_k, 2)
                            # Check against all top-k sources
                            mask_flat = torch.zeros(src_indices_flat.shape[0], dtype=torch.bool, device=device)
                            for k_idx in range(actual_k_reverse):
                                src_kps_cyclic = src_coords_all[topk_sources_for_trg[k_idx]]  # (N*actual_k, 2)
                                dists = (src_kps.float() - src_kps_cyclic.float()).norm(dim=-1)
                                mask_flat |= (dists <= bidirectional_threshold)

                        elif bidirectional_mode == 'eucl_dist_3d':
                            # 3D distance: check if any of the top-k sources is close enough
                            mask_flat = torch.zeros(src_indices_flat.shape[0], dtype=torch.bool, device=device)
                            for k_idx in range(actual_k_reverse):
                                dists = (vertices_s[src_indices_flat] - vertices_s[topk_sources_for_trg[k_idx]]).norm(dim=-1)
                                mask_flat |= (dists <= bidirectional_threshold)

                        else:  # 'quantile'
                            # Find minimum distance to any of the top-k sources
                            min_dists = torch.full((src_indices_flat.shape[0],), float('inf'), device=device)
                            for k_idx in range(actual_k_reverse):
                                dists = (vertices_s[src_indices_flat] - vertices_s[topk_sources_for_trg[k_idx]]).norm(dim=-1)
                                min_dists = torch.min(min_dists, dists)

                            q = 0.01
                            if torch.isnan(min_dists).all():
                                mask_flat = torch.ones_like(min_dists, dtype=torch.bool)
                            else:
                                eps = torch.quantile(min_dists[~torch.isnan(min_dists)], q)
                                mask_flat = min_dists <= eps

                            if mask_flat.sum() == 0 and min_dists.numel() > 0:
                                new_q = min(q * 2, 0.5)
                                non_nan_dists = min_dists[~torch.isnan(min_dists)]
                                if non_nan_dists.numel() > 0:
                                    eps = torch.quantile(non_nan_dists, new_q)
                                    mask_flat = min_dists <= eps
                                else:
                                    mask_flat = torch.ones_like(min_dists, dtype=torch.bool)

                    # Filter pairs that pass consistency check
                    src_coords_expanded = src_coords_all.unsqueeze(1).repeat(1, actual_k, 1).reshape(-1, 2)  # (N*actual_k, 2)
                    trg_coords_mapped_expanded = trg_coords_all[best_match_indices_for_src].reshape(-1, 2)  # (N*actual_k, 2)

                    src_coords = src_coords_expanded[mask_flat]
                    trg_coords_mapped = trg_coords_mapped_expanded[mask_flat]
                    # Group IDs: keep original source index after filtering
                    group_ids = src_indices_flat[mask_flat]

            # Return 2D grid coords with group IDs
            self._pl_coords_cache[key] = (src_coords.detach().cpu(), trg_coords_mapped.detach().cpu(), group_ids.detach().cpu())
            return self._pl_coords_cache[key]

    def _preload_pseudo_label_coords(self):
        import tqdm, os.path as osp
        for off_path in tqdm.tqdm(self.dataset.off_list, desc="Preload pseudo-label coords"):
            file_names = os.path.basename(off_path).replace('.npz', '')
            category = off_path.split(os.sep)[-2]
            src_id = extract_id_from_vertices_path(file_names, 0)
            trg_id = extract_id_from_vertices_path(file_names, 1)
            self._get_cached_pseudo_label_coords(
                category=category,
                file_names=file_names,
                src_id=src_id,
                trg_id=trg_id,
                bidirectional=self.bidirectional,
                top_k=self.top_k,
                bidirectional_mode=self.bidirectional_mode,
                bidirectional_threshold=self.bidirectional_threshold,
                bidirectional_symmetric=self.bidirectional_symmetric
            )

    def _load_kps_and_threshold_for_pair(self, pair_json_path):
        with open(pair_json_path) as f:
            data = json.load(f)

        category = data["category"]
        data_dir = self.dataset.data_dir

        category_anno = list(glob(f'{data_dir}/ImageAnnotation/{category}/*.json'))[0]
        with open(category_anno) as f:
            num_kps = len(json.load(f)['kps'])

        blank_kps = torch.zeros(num_kps, 3)
        src_ann_path = os.path.join(data_dir, 'ImageAnnotation', category, data["src_imname"].replace('.jpg', '.json'))
        with open(src_ann_path, 'r') as f:
            src_ann = json.load(f)
        src_kps_dict = src_ann['kps']
        src_kp_ixs = torch.tensor([int(kp_id) for kp_id, kp in src_kps_dict.items() if kp is not None], dtype=torch.long).view(-1, 1).repeat(1, 3)
        src_kps_vals = [kp for kp in src_kps_dict.values() if kp is not None]
        if len(src_kps_vals) > 0:
            source_raw_kps = torch.cat([torch.tensor(src_kps_vals, dtype=torch.float), torch.ones(src_kp_ixs.size(0), 1)], 1)
            source_kps_filled = blank_kps.scatter(dim=0, index=src_kp_ixs, src=source_raw_kps)
        else:
            source_kps_filled = blank_kps
        source_kps, _, _, _ = preprocess_kps_pad(source_kps_filled, data["src_imsize"][0], data["src_imsize"][1], self.resolution)

        trg_ann_path = os.path.join(data_dir, 'ImageAnnotation', category, data["trg_imname"].replace('.jpg', '.json'))
        with open(trg_ann_path, 'r') as f:
            trg_ann = json.load(f)
        trg_kps_dict = trg_ann['kps']
        trg_kp_ixs = torch.tensor([int(kp_id) for kp_id, kp in trg_kps_dict.items() if kp is not None], dtype=torch.long).view(-1, 1).repeat(1, 3)
        trg_kps_vals = [kp for kp in trg_kps_dict.values() if kp is not None]
        if len(trg_kps_vals) > 0:
            target_raw_kps = torch.cat([torch.tensor(trg_kps_vals, dtype=torch.float), torch.ones(trg_kp_ixs.size(0), 1)], 1)
            target_kps_filled = blank_kps.scatter(dim=0, index=trg_kp_ixs, src=target_raw_kps)
        else:
            target_kps_filled = blank_kps
        target_kps, _, _, trg_scale = preprocess_kps_pad(target_kps_filled, data["trg_imsize"][0], data["trg_imsize"][1], self.resolution)

        target_bbox = np.asarray(data["trg_bndbox"])
        threshold = max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]) * trg_scale
        kps = torch.stack([source_kps, target_kps])
        used_kps_indices, = torch.where(kps[:, :, 2].any(dim=0))
        kps = kps[:, used_kps_indices, :]

        return kps, torch.tensor(threshold, dtype=torch.float32)

    def _process_descriptor(self, descriptors_collection):
        if self.descriptor == 'sd_dino':
            features1 = descriptors_collection[0][0]
            features2 = descriptors_collection[1][0]

            img1_desc_dino = descriptors_collection[0][1]
            img2_desc_dino = descriptors_collection[1][1]

            img1_sd_s3 = features1['s3']
            img1_sd_s4 = torch.nn.functional.interpolate(features1['s4'], size=(self.num_patches, self.num_patches), mode='bilinear')
            img1_sd_s5 = torch.nn.functional.interpolate(features1['s5'], size=(self.num_patches, self.num_patches), mode='bilinear')

            B, _, _, C_dino = img1_desc_dino.shape
            img1_dino_feat = img1_desc_dino.squeeze(1).permute(0, 2, 1).reshape(B, C_dino, self.num_patches, self.num_patches)

            img2_sd_s3 = features2['s3']
            img2_sd_s4 = torch.nn.functional.interpolate(features2['s4'], size=(self.num_patches, self.num_patches), mode='bilinear')
            img2_sd_s5 = torch.nn.functional.interpolate(features2['s5'], size=(self.num_patches, self.num_patches), mode='bilinear')
            img2_dino_feat = img2_desc_dino.squeeze(1).permute(0, 2, 1).reshape(B, C_dino, self.num_patches, self.num_patches)

            img1_aggregation_input = torch.cat([img1_sd_s3, img1_sd_s4, img1_sd_s5, img1_dino_feat], dim=1)
            img2_aggregation_input = torch.cat([img2_sd_s3, img2_sd_s4, img2_sd_s5, img2_dino_feat], dim=1)

            return torch.stack([img1_aggregation_input.squeeze(), img2_aggregation_input.squeeze()])
        else:
            return torch.stack(descriptors_collection)

    def _custom_collate_fn(self, batch):
        descriptors = []
        if self.split == 'trn' and not self.spectral_inputs:
            pseudo_labels = []
            thresholds = []
            group_ids_list = []
        
        if self.spectral_inputs:
            vertices, eVals, eVecs, Ls, Ms, gradXs, gradYs, masks = [], [], [], [], [], [], [], []

        if (self.return_target_mask or self.label_update) and self.split == 'trn' and not self.spectral_inputs:
            masks = []

        for data in batch:
            file_names = os.path.basename(data).replace('.npz', '')
            category = data.split(os.sep)[-2]

            if self.spectral_inputs:
                v_list, eVal_list, eVec_list, L_list, M_list, gX_list, gY_list, masks_collection = [], [], [], [], [], [], [], []
                for i in range(2):
                    v_list.append(torch.load(f'{self.input_dir}/vertices/{category}/vertices_{file_names}_{i}.pt',
                                             map_location=self.device))
                    eVal_list.append(torch.load(f'{self.input_dir}/laplacian/eVals/{category}/eVal_{file_names}_{i}.pt',
                                                map_location=self.device)[:self.n_eig - 1])
                    eVec_list.append(torch.load(f'{self.input_dir}/laplacian/eVecs/{category}/eVec_{file_names}_{i}.pt',
                                                map_location=self.device)[:, :self.n_eig - 1])
                    L = torch.load(f'{self.input_dir}/laplacian/Ls/{category}/L_{file_names}_{i}.pt',
                                   map_location=self.device)
                    L_list.append(L.to_dense() if L.is_sparse else L)
                    M_list.append(torch.load(f'{self.input_dir}/laplacian/Ms/{category}/M_{file_names}_{i}.pt',
                                             map_location=self.device))
                    gX = torch.load(f'{self.diffusion_dir}/gradX/{category}/gradX_{file_names}_{i}.pt',
                                    map_location=self.device)
                    gY = torch.load(f'{self.diffusion_dir}/gradY/{category}/gradY_{file_names}_{i}.pt',
                                    map_location=self.device)
                    gX_list.append(gX.to_dense() if gX.is_sparse else gX)
                    gY_list.append(gY.to_dense() if gY.is_sparse else gY)

                    desc_id = extract_id_from_vertices_path(file_names, i)
                    mask_dir = osp.join(self.dataset.data_dir, 'SAMMasks', category, f'{desc_id}_mask.png')
                    mask_img = Image.open(mask_dir).convert("L")
                    mask_tensor = torchvision.transforms.ToTensor()(mask_img)
                    masks_collection.append(mask_tensor)

                vertices.append(torch.stack(v_list))
                eVals.append(torch.stack(eVal_list))
                eVecs.append(torch.stack(eVec_list))
                Ls.append(torch.stack(L_list))
                Ms.append(torch.stack(M_list))
                gradXs.append(torch.stack(gX_list))
                gradYs.append(torch.stack(gY_list))
                masks.append(torch.nn.functional.interpolate(torch.stack(masks_collection), size=(self.num_patches, self.num_patches), mode='nearest').squeeze(1))

            descs = []
            for i in range(2):
                desc_id = extract_id_from_vertices_path(file_names, i)
                _features = torch.load(osp.join(self.dataset.data_dir, 'Features', f'{self.resolution}', category, f'{desc_id}.pt'),
                                       map_location=self.device)
                if self.descriptor == 'sd_dino':
                    features = [_features['sd'], _features['dino']]
                else:
                    features = _features[self.descriptor].squeeze().reshape(self.num_patches, self.num_patches, -1)
                descs.append(features)

            pair_json_path = os.path.join(self.dataset.annotation_dir, f'{file_names}:{category}.json')
            _, threshold = self._load_kps_and_threshold_for_pair(pair_json_path)
            thresholds.append(threshold.item())

            descriptors.append(self._process_descriptor(descs))
            if self.split == 'trn' and not self.spectral_inputs:
                src_id = extract_id_from_vertices_path(file_names, 0)
                trg_id = extract_id_from_vertices_path(file_names, 1)

                pl_src_cpu, pl_trg_cpu, group_ids_cpu = self._get_cached_pseudo_label_coords(
                    category=category,
                    file_names=file_names,
                    src_id=src_id,
                    trg_id=trg_id,
                    bidirectional=self.bidirectional,
                    top_k=self.top_k,
                    bidirectional_mode=self.bidirectional_mode,
                    bidirectional_threshold=self.bidirectional_threshold,
                    bidirectional_symmetric=self.bidirectional_symmetric
                )

                src_coords = pl_src_cpu.to(self.device)
                trg_coords_mapped = pl_trg_cpu.to(self.device)
                group_ids = group_ids_cpu.to(self.device)
                
                if self.label_update:
                    pseudo_label = torch.stack([src_coords, trg_coords_mapped]) # (2, P*P, 3)
                    group_ids_final = group_ids
                else:
                    if self.sampling > 0:
                        idx = torch.randperm(src_coords.shape[0], device=src_coords.device)[:self.sampling]
                        pseudo_label = torch.stack([src_coords[idx], trg_coords_mapped[idx]])
                        group_ids_final = group_ids[idx]
                    else:
                        # Don't pad here - will pad to batch max size later
                        pseudo_label = torch.stack([src_coords, trg_coords_mapped])
                        group_ids_final = group_ids

                pseudo_labels.append(pseudo_label)
                group_ids_list.append(group_ids_final)

                if self.return_target_mask or self.label_update:
                    # ot_plan = self.pad_data(ot_plan, self.num_patches ** 2, padding_value=0)
                    src_mask_cpu, trg_mask_cpu = self._get_cached_masks(
                        category=category, file_names=file_names, src_id=src_id, trg_id=trg_id
                    )
                    src_mask = src_mask_cpu.to(self.device)  # (1,P,P) bool
                    trg_mask = trg_mask_cpu.to(self.device)

                    if self.label_update or self.bidirectional:
                        masks.append(torch.cat([src_mask, trg_mask]))
                    else:
                        masks.append(trg_mask)

        if self.split == 'trn':
            if self.spectral_inputs:
                return (torch.stack(vertices), torch.stack(eVals), torch.stack(eVecs),
                        torch.stack(Ls), torch.stack(Ms), torch.stack(descriptors),
                        torch.stack(gradXs), torch.stack(gradYs), torch.stack(masks))
            else:
                # Dynamic padding: pad all samples to the max size in the batch
                if self.sampling > 0:
                    # If sampling is used, all samples should already be the same size
                    max_size = self.sampling
                    # Pad samples that are smaller than sampling size
                    padded_pseudo_labels = []
                    padded_group_ids = []
                    for pl, gid in zip(pseudo_labels, group_ids_list):
                        current_size = pl.shape[1]
                        if current_size < max_size:
                            pl_padded = self.pad_data(pl, max_size, padding_value=-1)
                            pad_size = max_size - gid.shape[0]
                            gid_padded = torch.cat([gid, torch.full((pad_size,), -1, dtype=gid.dtype, device=gid.device)])
                        else:
                            pl_padded = pl
                            gid_padded = gid
                        padded_pseudo_labels.append(pl_padded)
                        padded_group_ids.append(gid_padded)
                    pseudo_labels = padded_pseudo_labels
                    group_ids_list = padded_group_ids
                else:
                    # Find max size in the batch
                    max_size = max(pl.shape[1] for pl in pseudo_labels)

                    # Pad all samples to max_size
                    padded_pseudo_labels = []
                    padded_group_ids = []
                    for pl, gid in zip(pseudo_labels, group_ids_list):
                        current_size = pl.shape[1]
                        if current_size < max_size:
                            pl_padded = self.pad_data(pl, max_size, padding_value=-1)
                            pad_size = max_size - gid.shape[0]
                            gid_padded = torch.cat([gid, torch.full((pad_size,), -1, dtype=gid.dtype, device=gid.device)])
                        else:
                            pl_padded = pl
                            gid_padded = gid
                        padded_pseudo_labels.append(pl_padded)
                        padded_group_ids.append(gid_padded)
                    pseudo_labels = padded_pseudo_labels
                    group_ids_list = padded_group_ids

                if self.return_target_mask or self.label_update:
                    return torch.stack(descriptors), torch.stack(pseudo_labels), torch.tensor(thresholds), torch.stack(masks), torch.stack(group_ids_list)
                return torch.stack(descriptors), torch.stack(pseudo_labels), torch.tensor(thresholds), torch.stack(group_ids_list)
        else:
            data = batch[0]
            file_names = os.path.basename(data).replace('.npz', '')
            category = data.split(os.sep)[-2]

            pair_json_path = os.path.join(self.dataset.annotation_dir, f'{file_names}:{category}.json')
            kps, threshold = self._load_kps_and_threshold_for_pair(pair_json_path)

            if self.spectral_inputs:
                 return (torch.stack(vertices), torch.stack(eVals), torch.stack(eVecs),
                        torch.stack(Ls), torch.stack(Ms), torch.stack(descriptors),
                        torch.stack(gradXs), torch.stack(gradYs),
                        kps.to(self.device), threshold.to(self.device))
            else:
                return (torch.stack(descriptors), kps.to(self.device), threshold.to(self.device))

