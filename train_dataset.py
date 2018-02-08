import numbers
import os
import random
import warnings
import math

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from skimage import io, exposure

warnings.filterwarnings("ignore")


class NucleiTrainDataset(Dataset):
    def __init__(self, root_dir, bad_images, channels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
        self.image_ids = [i for i in next(os.walk(root_dir))[
            1] if i not in bad_images]

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
            mask_part = Image.fromarray(mask_part)
            masks.append(mask_part)

        sample = {'image': Image.fromarray(image), 'combined_mask': Image.fromarray(combined_mask), 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TrainRescale():
    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']

        img = F.resize(image, self.output_size)
        combined_msk = F.resize(combined_mask, self.output_size)
        msks = [F.resize(mask, self.output_size) for mask in masks]

        return {'image': img, 'combined_mask': combined_msk, 'masks': msks}


class TrainToTensor():
    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        masks = [F.to_tensor(mask).type(torch.FloatTensor) for mask in masks]

        return {'image': F.to_tensor(image).type(torch.FloatTensor), 'combined_mask': F.to_tensor(combined_mask).type(torch.FloatTensor), 'masks': masks}


class TrainNormalize():
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}

class TrainGrayscale():
    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        num_output_channels = 1 if image.mode == 'L' else 3
        image = F.to_grayscale(image, num_output_channels=num_output_channels)
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}


class RandomResizedCrop():
    def __init__(self, size, scale=(0.08,1.0), ratio=(3./4, 4./3), prob=0.5, interpolation=Image.BILINEAR):
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError("Size must be int or len 2.")
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.prob = prob

    @staticmethod
    def get_params(img, scale, ratio, prob):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < prob:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']

        i, j, h, w = self.get_params(image, self.scale, self.ratio, self.prob)
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        combined_mask = F.resized_crop(combined_mask, i, j, h, w, self.size, self.interpolation)
        masks = [F.resized_crop(mask, i, j, h, w, self.size, self.interpolation) for mask in masks]
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}


class RandomHorizontalFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        if random.random() < self.prob:
            image = F.hflip(image)
            combined_mask = F.hflip(combined_mask)
            masks = map(F.hflip, masks)
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}


class RandomVerticalFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        if random.random() < self.prob:
            image = F.vflip(image)
            combined_mask = F.vflip(combined_mask)
            masks = map(F.vflip, masks)
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}


class ColorJitter():
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        if self.brightness > 0:
            brightness_factor = np.random.uniform(
                max(0, 1 - self.brightness), 1 + self.brightness)
            image =  F.adjust_brightness(image, brightness_factor)
        if self.contrast > 0:
            contrast_factor = np.random.uniform(
                max(0, 1 - self.contrast), 1 + self.contrast)
            image =  F.adjust_contrast(image, contrast_factor)
        if self.saturation > 0:
            saturation_factor = np.random.uniform(
                max(0, 1 - self.saturation), 1 + self.saturation)
            image = F.adjust_saturation(image, saturation_factor)
        if self.hue > 0:
            hue_factor = np.random.uniform(-self.hue, self.hue)
            image = F.adjust_hue(image, hue_factor)

        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}


class RandomRotation():
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError(
                    "If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        angle = self.get_params(self.degrees)

        image = F.rotate(image, angle, self.resample, self.expand, self.center)
        combined_mask = F.rotate(combined_mask, angle,
                                 self.resample, self.expand, self.center)
        masks = [F.rotate(mask, angle, self.resample,
                          self.expand, self.center) for mask in masks]

        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}

class RandomInvert():
    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']

        if random.random() < self.prob:
            image = ImageOps.invert(image)
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}

class TrainCLAHEEqualize():
    def __call__(self, sample):
        image, masks, combined_mask = sample['image'], sample['masks'], sample['combined_mask']
        image = exposure.equalize_adapthist(np.array(image))
        image = Image.fromarray(image.astype('uint8'))
        return {'image': image, 'combined_mask': combined_mask, 'masks': masks}
