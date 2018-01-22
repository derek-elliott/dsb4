# Use this guy to run your first model: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
# Use this to try to do some preprocessing: https://github.com/orobix/retina-unet/blob/master/lib/pre_processing.py


import csv
import os

import numpy as np

import matplotlib.pyplot as plt
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

    def load_images(self):
        print("Loading and resizing images...")
        bar = progressbar.ProgressBar(max_value=len(self.image_ids))
        for n, image_id in bar(enumerate(self.image_ids)):
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

    def show_image(self, index):
        imshow(self.images[index])
        plt.show()

    def show_mask(self, index):
        imshow(np.squeeze(self.masks[index]))
        plt.show()

    def _rle_encode(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def _prob_to_rles(x, cutoff=0.5):
        lab_img = label(x > cutoff)
        for i in range(1, lab_img.max() + 1):
            yield self._rle_encode(lab_img == 1)

    def generate_submission(self, file_name):
        new_image_ids = []
        rles = []
        for n, image_id in enumerate(self.image_ids):
            rle = list(self._prob_to_rles(self.predictions[n]))
            rles.extend(rle)
            new_test_ids.extend([image_id] * len(rle))
        with open(os.path.join('submissions', f'{file_name}.csv'), 'wb') as f:
            wr = csv.writer(f)
            wr.writerow('ImageId,EncodedPixels')
            for image_id, run_length in zip(new_image_ids, rles):
                output_rle = ' '.join(run_length)
                wr.writerow(f'{image_id},{output_rle}')
