import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from skimage import io, transform
from skimage.color import rgb2gray
from skimage.exposure import adjust_gamma, equalize_adapthist

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
        sample = {'image': image,
                  'orig_size': image.shape[:-1], 'file_name': image_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class PredictRescale():
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, orig_size, image_name = sample['image'], sample['orig_size'], sample['file_name']

        img = transform.resize(image, self.output_size)

        return {'image': img, 'orig_size': orig_size, 'file_name': image_name}


class PredictToTensor():
    def __call__(self, sample):
        image, orig_size, image_name = sample['image'], sample['orig_size'], sample['file_name']
        img = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(img).type(torch.FloatTensor), 'orig_size': orig_size, 'file_name': image_name}


class PredictNormalize():
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, sample):
        image, orig_size, image_name = sample['image'], sample['orig_size'], sample['file_name']
        img = (image - self.minimum) / (self.maximum - self.minimum)
        return {'image': img, 'orig_size': orig_size, 'file_name': image_name}


class PredictToGrayScale():
    def __call__(self, sample):
        image, orig_size, image_name = sample['image'], sample['orig_size'], sample['file_name']
        img = rgb2gray(image)
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        return {'image': img, 'orig_size': orig_size, 'file_name': image_name}


class PredictCLAHEEqualize():
    def __call__(self, sample):
        image, orig_size, image_name = sample['image'], sample['orig_size'], sample['file_name']
        img = equalize_adapthist(image)
        return {'image': img, 'orig_size': orig_size, 'file_name': image_name}


class PredictAdjustGamma():
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, sample):
        image, orig_size, image_name = sample['image'], sample['orig_size'], sample['file_name']
        img = adjust_gamma(image, gamma=self.gamma)
        return {'image': img, 'orig_size': orig_size, 'file_name': image_name}
