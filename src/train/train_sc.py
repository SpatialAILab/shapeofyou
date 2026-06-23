import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_EVAL_DIR = _THIS_DIR.parent / 'eval'
for _path in (str(_THIS_DIR), str(_EVAL_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import spair_dataset as module_dataset
import data_loader_spair as module_data_loader
from train_config import ConfigParser
from train_utils import prepare_device, set_seed
import torch
import argparse
import tqdm
import model_utils.projection_network as module_arch
from corr_map_model import Correlation2Displacement
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np
from train_losses import get_sparse_contrastive_loss, get_dense_loss
import wandb

def main(config):
    set_seed(config)
    logger = config.get_logger('train')
    seed = config.config.get('num_anchors')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = config.init_obj('dataset', module_dataset)
    sparse_loss = config["trainer"]["sparse_loss"]
    beta = float(config["trainer"]["beta"]) if config["trainer"]["beta"] is not None else 0.3
    ot_iters = int(config["trainer"].get("ot_iters", 100))
    if config['dataset']['type'] == "SPairPseudoLabel_Dataset":
        train_data_loader = config.init_ftn('data_loader', module_dataset, dataset=train_dataset, seed=seed)
        train_data_loader = train_data_loader()
        descriptor_name = config['dataset']['args']['descriptor']
    else:
        if sparse_loss == 'contrastive':
            return_target_mask = False
        else:
            return_target_mask = True
        train_data_loader = config.init_obj('data_loader', module_data_loader, dataset=train_dataset, training=True, drop_last=True, shuffle=True, return_target_mask=return_target_mask)
        descriptor_name = config['data_loader']['args']['descriptor']
   
    model = config.init_obj('arch', module_arch)
    if sparse_loss == 'ot':
        model.bin_score = nn.Parameter(torch.tensor(1.))
    else:
        model.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    logger.info(model)
    load_model_if_available(config, model, logger)

    device, device_ids = prepare_device(config['n_gpu'], config['gpu_id'])
    model = model.to(device)

    corr_map_net = Correlation2Displacement().to(device)

    all_params = list(filter(lambda p: p.requires_grad, model.parameters())) + \
                list(corr_map_net.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, all_params)
    # Learning rate scheduler (optional)
    if config.config.get('lr_scheduler') is not None:
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    else:
        lr_scheduler = None  # Uncomment the next line if you want to use a scheduler

    lambda_dense = 1.0
    num_epochs = config['trainer']['epochs']
    sparse_loss = config['trainer']['sparse_loss']
    save_dir = os.path.join(str(config._save_dir), 'weights')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        logger.error("'save_dir' not specified in config file. Models will not be saved.")

    use_wandb = bool(config.config.get('wandb', False))
    if use_wandb:
        wandb_name = f"{descriptor_name}_{config['training_name']}"
        wandb.init(project=config['name'], name=wandb_name, config=config.config)

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        epoch_total_loss = 0.0
        epoch_sparse_loss = 0.0
        epoch_dense_loss = 0.0

        pbar = tqdm.tqdm(train_data_loader, desc=f"Epoch {epoch}")
        for data in pbar:
            if train_data_loader.label_update:
                descriptors, points, thresholds, masks, group_ids = [d.to(device) for d in data]
            elif train_data_loader.return_target_mask:
                descriptors, pseudo_labels, thresholds, masks, group_ids = [d.to(device) for d in data]
            else:
                descriptors, pseudo_labels, thresholds, group_ids = [d.to(device) for d in data]
            B, _, _, P, _ = descriptors.shape
            C = model.projection_dim
            if descriptor_name == "dino":
                descriptors = descriptors.permute(0, 1, 4, 2, 3).reshape(B * 2, -1, P, P)
            else:
                descriptors = descriptors.reshape(B * 2, -1, P, P)
            refined_features = model(descriptors)
            src_features, trg_features = refined_features.view(B, 2, C, P, P).split(1, dim=1)
            src_features, trg_features = src_features.squeeze(1), trg_features.squeeze(1)

            total_loss_sparse = 0.0
            total_loss_dense = 0.0
            num_valid_pairs = 0
            for i in range(B):
                if train_data_loader.label_update:
                    raise NotImplementedError('Online label_update mode is not part of the initial public training release.')
                _pseudo_labels = pseudo_labels[i]
                valid_mask = (_pseudo_labels != -1).any(dim=2)
                pseudo_label_src_coords, pseudo_label_trg_coords = _pseudo_labels[0][valid_mask[0]], _pseudo_labels[1][valid_mask[1]]
                if pseudo_label_src_coords is not None and len(pseudo_label_src_coords) > 0:
                    if sparse_loss == 'contrastive' or sparse_loss == 'supcon':
                        pl_src_feats = src_features[i, :, pseudo_label_src_coords[:, 1], pseudo_label_src_coords[:, 0]].T
                        pl_trg_feats = trg_features[i, :, pseudo_label_trg_coords[:, 1], pseudo_label_trg_coords[:, 0]].T

                        # Pass group_ids for supervised contrastive loss
                        # Filter group_ids with valid_mask to match the filtered pseudo labels
                        current_group_ids = group_ids[i][valid_mask[0]] if group_ids is not None else None
                        loss_sparse = get_sparse_contrastive_loss(
                            pl_src_feats, pl_trg_feats,
                            logit_scale=model.logit_scale.exp(),
                            sparse_loss=sparse_loss,
                            group_ids=current_group_ids
                        )
                    elif sparse_loss == 'soft_supcon':
                        # Soft Supervised Contrastive Loss
                        pl_src_feats = src_features[i, :, pseudo_label_src_coords[:, 1], pseudo_label_src_coords[:, 0]].T
                        pl_trg_feats = trg_features[i, :, pseudo_label_trg_coords[:, 1], pseudo_label_trg_coords[:, 0]].T
                        current_group_ids = group_ids[i][valid_mask[0]] if group_ids is not None else None
                        loss_sparse = get_sparse_contrastive_loss(
                            pl_src_feats, pl_trg_feats,
                            logit_scale=model.logit_scale.exp(),
                            sparse_loss=sparse_loss,
                            group_ids=current_group_ids,
                            beta=beta,
                            ot_iters=ot_iters
                        )
                    elif sparse_loss == 'ot':
                        loss_sparse = get_sparse_contrastive_loss(
                            src_features[i],
                            trg_features[i],
                            bin_score=model.bin_score,
                            pseudo_labels=torch.stack([pseudo_label_src_coords, pseudo_label_trg_coords]),
                            trg_mask=masks[i],
                            sparse_loss=sparse_loss,
                        )
                    else:
                        loss_sparse = get_sparse_contrastive_loss(
                            src_features[i],
                            trg_features[i],
                            logit_scale=model.logit_scale.exp(),
                            pseudo_labels=torch.stack([pseudo_label_src_coords, pseudo_label_trg_coords]),
                            trg_mask=masks[i],
                            beta=beta,
                            sparse_loss=sparse_loss
                        )
                    loss_dense = get_dense_loss(src_features[i], trg_features[i],
                                                pseudo_label_src_coords,
                                                pseudo_label_trg_coords,
                                                corr_map_net,
                                                threshold=thresholds[i].item())

                    total_loss_sparse += loss_sparse
                    total_loss_dense += loss_dense
                    num_valid_pairs += 1

            if num_valid_pairs == 0:
                continue

            final_loss_sparse = total_loss_sparse / num_valid_pairs
            final_loss_dense = total_loss_dense / num_valid_pairs
            final_loss = final_loss_sparse + lambda_dense * final_loss_dense

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            if use_wandb:
                wandb.log({
                    "train/iteration_loss": final_loss.item(),
                    "train/iteration_sparse_loss": final_loss_sparse.item(),
                    "train/iteration_dense_loss": final_loss_dense.item()
                })

            # logger.info(f"Batch Loss: {final_loss.item()} (Sparse: {final_loss_sparse.item()}, Dense: {final_loss_dense.item()})")
            epoch_total_loss += final_loss.item()
            epoch_sparse_loss += final_loss_sparse.item()
            epoch_dense_loss += final_loss_dense.item()

            pbar.set_postfix(loss=final_loss.item())

        num_iterations = len(train_data_loader)
        mean_epoch_loss = epoch_total_loss / num_iterations
        mean_sparse_loss = epoch_sparse_loss / num_iterations
        mean_dense_loss = epoch_dense_loss / num_iterations

        logger.info(f"--- Epoch {epoch} Average Training Loss: {mean_epoch_loss:.4f} ---")
        train_log_data = {
            "train/epoch_mean_loss": mean_epoch_loss,
            "train/epoch_mean_sparse_loss": mean_sparse_loss,
            "train/epoch_mean_dense_loss": mean_dense_loss,
        }
        if use_wandb:
            wandb.log({"epoch": epoch, **train_log_data})

        save_path_model = f'{save_dir}/model_epoch_{epoch}.pth'
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(model_state, save_path_model)


def load_model_if_available(config, model, logger):
    """
    Loads the model from a checkpoint if available.
    
    :param config: Configuration object
    :param model: Model to be loaded
    :param logger: Logger for printing information
    """
    if config.model_path is not None:
        logger.info('Loading checkpoint: {} ...'.format(str(config.model_path)))
        checkpoint = torch.load(str(config.model_path))
        if config['n_gpu'] == 1:
            # If loading from multi-GPU checkpoint to single GPU
            if 'module.' in list(checkpoint.keys())[0]:
                checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
            model.load_state_dict(checkpoint)
        else:
            # If loading from single-GPU checkpoint to multi-GPU
            if 'module.' not in list(checkpoint.keys())[0]:
                checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
            temp = torch.nn.DataParallel(model)
            temp.load_state_dict(checkpoint)
            model = temp.module

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config_test.json', type=str)
    args.add_argument('-r', '--resume', default=None, type=str)
    args.add_argument('-d', '--dataset', default=None, choices=["faust, scape"], type=str)
    args.add_argument('-n', '--number', default=100, type=str)
    args.add_argument('-m', '--model_path', default=None, type=str)
    args.add_argument('--run_id', default=None, type=str)
    args.add_argument('--device', default=None, type=str)
    args.add_argument('--num_threads', default=4, type=int,
                    help="Number of CPU threads to use for torch operations")
    config = ConfigParser.from_args(args)
    torch.set_num_threads(config.config.get('num_threads', 4))
    torch.set_num_interop_threads(max(1, config.config.get('num_threads', 4) // 2))
    main(config)