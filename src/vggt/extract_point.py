import os
import argparse
import json
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def save_point_cloud(world_points, rgb_image, confidence, save_path_base):
    points = world_points.reshape(-1, 3)
    colors = np.transpose(rgb_image, (1, 2, 0)).reshape(-1, 3)
    conf_values = confidence.flatten()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    ply_path = save_path_base + '.ply'
    npy_path = save_path_base + '.npy'

    o3d.io.write_point_cloud(ply_path, pcd)
    np.save(npy_path, conf_values)


def save_partial_point_cloud(world_points, rgb_image, confidence, mask, save_path_base):
    H, W = mask.shape
    mask_flat = mask.flatten() > 0.5

    if (H, W) != world_points.shape[:2] or (H, W) != rgb_image.shape[1:]:
        raise ValueError(f"Mask dimensions {(H, W)} must match image dimensions {world_points.shape[:2]}")

    points = world_points.reshape(-1, 3)[mask_flat]
    colors = np.transpose(rgb_image, (1, 2, 0)).reshape(-1, 3)[mask_flat]
    conf_values = confidence.flatten()[mask_flat]

    if len(points) == 0:
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    ply_path = save_path_base + '.ply'
    npy_path = save_path_base + '.npy'

    o3d.io.write_point_cloud(ply_path, pcd)
    np.save(npy_path, conf_values)


def save_prediction_npz(prediction_dict, save_path_base):
    npz_path = save_path_base + ".npz"
    np.savez_compressed(npz_path, **prediction_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SPair-71k point predictions from image pairs using VGGT.")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--category", type=str)
    parser.add_argument("--pair_dir", type=str)
    parser.add_argument("--mode", default="test", type=str, choices=["trn", "val", "test"], help="Mode for SPair-71k dataset.")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of CPU threads for PyTorch.")
    parser.add_argument("--size", type=int, default=840)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    print(f"Loading VGGT model to {device} with {dtype}...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.point_head = None
    model.eval()
    print("Model loaded. Using Depth and Camera heads.")

    image_paths_to_process = []

    names = []
    if not all([args.pair_dir, args.category]):
        raise ValueError("For SPair-71k, --pair_dir and --category are required.")

    output_pc_dir = os.path.join(args.save_dir, 'PartialPCs', args.mode, args.category)

    json_files = [f for f in os.listdir(os.path.join(args.pair_dir, args.mode)) if f.endswith('.json')]
    for json_file in json_files:
        with open(os.path.join(args.pair_dir, args.mode, json_file)) as f:
            pair_data = json.load(f)
        if pair_data['category'] == args.category:
            image_paths_to_process.append((
                os.path.join(args.image_dir, args.category, pair_data['src_imname']),
                os.path.join(args.image_dir, args.category, pair_data['trg_imname']),
            ))
            names.append(json_file.split(':')[0])

    os.makedirs(output_pc_dir, exist_ok=True)

    for idx, item in enumerate(tqdm(image_paths_to_process, desc="Processing SPair-71k")):
        save_base = os.path.join(output_pc_dir, names[idx])

        if os.path.isfile(save_base + ".npz"):
            continue

        current_image_paths = list(item)

        images, _ = load_and_preprocess_images(current_image_paths, mode="pad", bbox_list=None, target_size=args.size)
        images = images.to(device)
        B, _, H, W = images.shape

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images=images, learnable=False)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        for key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        depth_map = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        extrinsics_cam = predictions["extrinsic"]
        intrinsics_cam = predictions["intrinsic"]

        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        predictions["world_points"] = world_points

        save_prediction_npz(predictions, save_base)
        
    print("\nPoint cloud extraction complete.")