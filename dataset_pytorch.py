import os
import warnings

import numpy as np
from skimage import io

import torch
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class NucleiDataset(Dataset):
    def __init__(self, root_dir, channels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
        self.image_ids = next(os.walk(root_dir))[1]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_name,
                                'images', f'{image_name}.png')
        image = io.imread(os.path.join(img_path))[:, :, :self.channels]

        masks = []
        mask_path = os.path.join(self.root_dir, str(image_name), 'masks')
        combined_mask = mask = np.zeros(image.shape[:2], dtype=np.bool)
        for file in next(os.walk(mask_path))[2]:
            mask_part = io.imread(os.path.join(mask_path, file))
            combined_mask = np.maximum(combined_mask, mask_part)
            masks.append(mask_part)

        sample = {'image': image,
                  'combined_mask': combined_mask, 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample
