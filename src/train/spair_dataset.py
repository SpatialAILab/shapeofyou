import os
from glob import glob
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class SPair_Dataset(Dataset):
    def __init__(self, data_dir, category, split='train'):
        assert split in ['train', 'val', 'test'], f'Invalid split: {split}'
        self.mode = 'trn' if split == 'train' else split
        self.data_dir = data_dir
        self.pc_dir = os.path.join(data_dir, 'PartialPCs')
        self.category = category
        self.annotation_dir = os.path.join(data_dir, 'PairAnnotation', self.mode)
        self.off_list = []

        if self.category == 'all':
            for f in sorted(os.listdir(self.annotation_dir)):
                if f.endswith('.json'):
                    base, cat = f.replace('.json', '').split(':')
                    self.off_list.append(os.path.join(self.pc_dir, self.mode, cat, base + '.npz'))
        else:
            json_files = [f for f in os.listdir(self.annotation_dir) if f.endswith(f':{self.category}.json')]
            for f in sorted(json_files):
                base = f.replace(f':{self.category}.json', '')
                self.off_list.append(os.path.join(self.pc_dir, self.mode, self.category, base + '.npz'))

        if self.mode == 'trn':
            print(f'training with {len(self.off_list)} combinations')
        elif self.mode == 'valid':
            print(f'validation with {len(self.off_list)} combinations')
        elif self.mode == 'test':
            print(f'testing with {len(self.off_list)} combinations')

    def __len__(self):
        return len(self.off_list)

    def __getitem__(self, idx):
        return self.off_list[idx]
