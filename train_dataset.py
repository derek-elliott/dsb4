import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from skimage import io, transform
from skimage.color import rgb2gray
from skimage.exposure import adjust_gamma, equalize_adapthist

warnings.filterwarnings("ignore")


class NucleiTrainDataset(Dataset):
    def __init__(self, root_dir, bad_images, channels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
        self.image_ids = [i for i in next(os.walk(root_dir))[1] if i not in bad_images]

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
        img = (image - self.minimum) / (self.maximum - self.minimum)
        return {'image': img, 'combined_mask': combined_mask, 'masks': masks}


class TrainToGrayScale():
    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        img = rgb2gray(image)
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        return {'image': np_img, 'combined_mask': combined_mask, 'masks': masks}


class TrainCLAHEEqualize():
    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        img = equalize_adapthist(image)
        return {'image': img, 'combined_mask': combined_mask, 'masks': masks}


class TrainAdjustGamma():
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        img = adjust_gamma(image, gamma=self.gamma)
        return {'image': img, 'combined_mask': combined_mask, 'masks': masks}
