# Use this guy to run your first model: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
# Use this to try to do some preprocessing: https://github.com/orobix/retina-unet/blob/master/lib/pre_processing.py

import os

import numpy as np

import cv2
import matplotlib.pyplot as plt
import progressbar
from scipy import ndimage as ndi
from skimage.exposure import adjust_gamma, equalize_adapthist
from skimage.feature import peak_local_max
from skimage.io import imread, imshow
from skimage.morphology import label, watershed
from skimage.transform import resize


class Images:
    def __init__(self, path, height, width, channels, is_training=True):
        # Ignore divide by 0 warnings
        np.seterr(divide='ignore', invalid='ignore')

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
        imgs_equalized = np.empty(self.images.shape)
        for i, image in enumerate(self.images):
            imgs_equalized[i] = equalize_adapthist(image)
        self.images = imgs_equalized

    def modify_gamma(self, gamma=1.0):
        new_imgs = np.empty(self.images.shape)
        for i, image in enumerate(self.images):
            new_imgs[i] = adjust_gamma(image, gamma=gamma)
        self.images = new_imgs

    def show_image_and_mask(self, index):
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        img = imshow(self.images[index])
        a = fig.add_subplot(1,2,2)
        img = imshow(np.squeeze(self.masks[index]))
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
        lab_img = list(self._watershed_segment(
            self.predictions[index] > cutoff))
        for img in lab_img:
            yield self._rle_encode(img)

    def _watershed_segment(self, image):
        distance = ndi.distance_transform_edt(image)
        local_max = peak_local_max(
            distance, min_distance=7, labels=image, indices=False)
        markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
        labels = watershed(-distance, markers, mask=image)
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = np.zeros(image.shape, dtype='uint8')
            mask[labels == label] = 1
            # self._display_images(image, distance, mask)
            yield mask

    def _display_images(self, image1, image2, image3):
        fig = plt.figure()
        a = fig.add_subplot(1,3,1)
        img = imshow(image1)
        a = fig.add_subplot(1,3,2)
        img = imshow(image2)
        a = fig.add_subplot(1,3,3)
        img = imshow(image3)
        plt.show()

    def generate_submission(self, cutoff, file_name):
        new_image_ids = []
        rles = []
        for n, image_id in enumerate(self.image_ids):
            rle = list(self._prob_to_rles(n, cutoff))
            rles.extend(rle)
            new_image_ids.extend([image_id] * len(rle))
        with open(os.path.join('submissions', f'{file_name}.csv'), 'w') as f:
            f.write('ImageId,EncodedPixels\n')
            for image_id, run_length in zip(new_image_ids, rles):
                output_rle = ' '.join(str(item) for item in run_length)
                f.write(f'{image_id},{output_rle}\n')
