import os
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from skimage import io, transform

warnings.filterwarnings("ignore")


class NucleiPredictDataset(Dataset):
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
        image = io.imread(img_path)[:, :, :self.channels]
        sample = {'image': image, 'orig_image': image, 'orig_size': image.shape[:-1], 'file_name': image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class PredictRescale():
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, orig_image, orig_size, image_name = sample['image'], sample['orig_image'], sample['orig_size'], sample['file_name']

        img = transform.resize(image, self.output_size)

        return {'image': img, 'orig_image': orig_image, 'orig_size': orig_size, 'file_name': image_name}


class PredictToTensor():
    def __call__(self, sample):
        image, orig_image, orig_size, image_name = sample['image'], sample['orig_image'], sample['orig_size'], sample['file_name']
        img = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(img).type(torch.FloatTensor), 'orig_image': orig_image, 'orig_size': orig_size, 'file_name': image_name}


class PredictNormalize():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, orig_image, orig_size, image_name = sample['image'], sample['orig_image'], sample['orig_size'], sample['file_name']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'orig_image': orig_image, 'orig_size': orig_size, 'file_name': image_name}
