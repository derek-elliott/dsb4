import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from skimage import io, transform

warnings.filterwarnings("ignore")


class NucleiTrainDataset(Dataset):
    def __init__(self, root_dir, channels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
        self.image_ids = next(os.walk(root_dir))[1]
        bad_images = ['b1eb0123fe2d8c825694b193efb7b923d95effac9558ee4eaf3116374c2c94fe',
                      '19f0653c33982a416feed56e5d1ce6849fd83314fd19dfa1c5b23c6b66e9868a',
                      '9bb6e39d5f4415bc7554842ee5d1280403a602f2ba56122b87f453a62d37c06e',
                      '1f0008060150b5b93084ae2e4dabd160ab80a95ce8071a321b80ec4e33b58aca',
                      '58c593bcb98386e7fd42a1d34e291db93477624b164e83ab2afa3caa90d1d921',
                      '12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40',
                      '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80',
                      '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9']
        self.image_ids = [i for i in self.image_ids if i not in bad_images]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_name = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_name,
                                'images', f'{image_name}.png')
        image = io.imread(img_path)[:, :, :self.channels]

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


class TrainRescale():
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']

        img = transform.resize(image, self.output_size)
        combined_msk = transform.resize(combined_mask, self.output_size)

        msks = []
        for mask in masks:
            msks.append(transform.resize(mask, self.output_size))

        return {'image': img, 'combined_mask': combined_msk, 'masks': msks}


class TrainToTensor():
    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        img = image.transpose((2, 0, 1))
        comb_mask = image.transpose((2, 0, 1))
        msks = []
        for mask in masks:
            msks.append(torch.from_numpy(mask).type(torch.FloatTensor))

        return {'image': torch.from_numpy(img).type(torch.FloatTensor), 'combined_mask': torch.from_numpy(combined_mask).type(torch.FloatTensor), 'masks': msks}


class TrainNormalize():
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        image = (image - self.minimum) / (self.maximum - self.minimum)
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}
