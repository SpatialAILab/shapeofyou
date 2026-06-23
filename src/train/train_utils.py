import json
import logging
import os
import random
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def read_json(fname):
    with Path(fname).open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with Path(fname).open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use, gpu_id):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: no GPU is available; training will run on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: requested {n_gpu_use} GPUs, but only {n_gpu} are available.")
        n_gpu_use = n_gpu
    device = torch.device(f'cuda:{gpu_id}' if n_gpu_use > 0 else 'cpu')
    return device, list(range(n_gpu_use))


def set_seed(config):
    seed = int(config['seed']) if config['seed'] is not None else np.random.randint(0, 10**8)
    config._config['seed'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def setup_logging(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'info.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )


def extract_number_from_filename(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None


def extract_id_from_vertices_path(filepath, index):
    parts = filepath.split('-')
    if index == 0:
        return parts[1]
    if index == 1:
        return parts[2]
    raise ValueError('Index must be 0 or 1.')


COUNT_INVIS = False


def preprocess_kps_pad(kps, img_width, img_height, size):
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    if not COUNT_INVIS:
        kps *= kps[:, 2:3].clone()
    return kps, offset_x, offset_y, scale
