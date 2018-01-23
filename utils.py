# Use this guy to run your first model: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
# Use this to try to do some preprocessing: https://github.com/orobix/retina-unet/blob/master/lib/pre_processing.py

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from skimage.io import imread, imshow
from skimage.morphology import label
from skimage.transform import resize


class Images:
    def __init__(self, path, height, width, channels, is_training=True):
        self.path = path
        self.height = height
        self.width = width
        self.channels = channels
        self.is_training = is_training
        self.original_sizes = []
        self.image_ids = next(os.walk(path))[1]
        self.images = np.zeros(
            (len(self.image_ids), height, width, channels), dtype=np.uint8)
        self.masks = np.zeros(
            (len(self.image_ids), height, width, 1), dtype=np.bool)
        self.predictions = []

    def load_images(self, bad_images=[]):
        print("Loading and resizing images...")
        bar = progressbar.ProgressBar(max_value=len(self.image_ids))
        for n, image_id in bar(enumerate(self.image_ids)):
            if image_id in bad_images:
                continue
            img_path = os.path.join(self.path, str(
                image_id), 'images', f'{image_id}.png')
            img = imread(img_path)[:, :, :self.channels]
            self.original_sizes.append([img.shape[0], img.shape[1]])
            img = resize(img,
                         (self.height, self.width),
                         mode='constant',
                         preserve_range=True)
            self.images[n] = img
            if self.is_training:
                mask = np.zeros((self.height, self.width, 1), dtype=np.bool)
                mask_path = os.path.join(self.path, str(image_id), 'masks')
                for file in next(os.walk(mask_path))[2]:
                    mask_part = imread(os.path.join(mask_path, file))
                    mask_part = np.expand_dims(resize(mask_part,
                                                      (self.height, self.width),
                                                      mode='constant',
                                                      preserve_range=True),
                                               axis=-1)
                    mask = np.maximum(mask, mask_part)
                self.masks[n] = mask

    def clahe_equalize(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgs_equalized = np.empty(self.images.shape)
        for i in range(self.images.shape[0]):
            imgs_equalized[i, 0] = clahe.apply(
                np.array(self.images[i, 0], dtype=np.uint8))
        self.images = imgs_equalized

    def normalize_images(self):
        imgs_normalized = np.empty(self.images.shape)
        imgs_std = np.std(self.images)
        imgs_mean = np.mean(self.images)
        imgs_normalized = (self.images - imgs_mean) / imgs_std
        for image in imgs_normalized:
            image = ((image - np.min(image)) /
                     (np.max(image) - np.min(image))) * 255
        self.images = imgs_normalized

    def adjust_gamma(self, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) *
                          255 for i in np.arange(0, 256)]).astype("uint8")
        new_imgs = np.empty(self.images.shape)
        for i in range(self.images.shape[0]):
            new_imgs[i, 0] = cv2.LUT(
                np.array(self.images[i, 0], dtype=np.uint8), table)
        self.images = new_imgs

    def show_image(self, index):
        imshow(self.images[index])
        plt.show()

    def show_mask(self, index):
        imshow(np.squeeze(self.masks[index]))
        plt.show()

    def upsample_masks(self):
        upsampled = []
        for index, mask in enumerate(self.predictions):
            upsampled.append(resize(np.squeeze(mask),
                                    (self.original_sizes[index][0],
                                     self.original_sizes[index][1]),
                                    mode='constant', preserve_range=True))
        self.predictions = upsampled

    def _rle_encode(self, x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def _prob_to_rles(self, index, cutoff=0.5):
        lab_img = label(self.predictions[index] > cutoff)
        for i in range(1, lab_img.max() + 1):
            yield self._rle_encode(lab_img == i)

    def generate_submission(self, file_name):
        new_image_ids = []
        rles = []
        for n, image_id in enumerate(self.image_ids):
            rle = list(self._prob_to_rles(n))
            rles.extend(rle)
            new_image_ids.extend([image_id] * len(rle))
        with open(os.path.join('submissions', f'{file_name}.csv'), 'w') as f:
            f.write('ImageId,EncodedPixels\n')
            for image_id, run_length in zip(new_image_ids, rles):
                output_rle = ' '.join(str(item) for item in run_length)
                f.write(f'{image_id},{output_rle}\n')
