#!/usr/bin/env python
import os
import sys
import time
from optparse import OptionParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from predict_dataset import (NucleiPredictDataset, PredictCLAHEEqualize,
                             PredictGrayscale, PredictNormalize,
                             PredictRescale, PredictToTensor)
from scipy import ndimage as ndi
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.io import imshow
from skimage.morphology import label, watershed
from skimage.transform import resize
from tqdm import tqdm
from unet import UNet


def predict(net, data_cfg, batch_size, channels, visualize=False, cutoff=0.5, use_gpu=False):
    transform = transforms.Compose(
        [PredictGrayscale(),
         PredictRescale((data_cfg['img_height'], data_cfg['img_width'])),
         PredictToTensor(),
         PredictNormalize()])

    dataset = NucleiPredictDataset(
        data_cfg['root_path'], channels=channels, transform=transform)

    predictions = {}

    data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=2)

    bar = tqdm(data_loader)
    for i, image in enumerate(bar):
        bar.set_description(desc=f'Image {i+1}/{len(dataset)}')
        if use_gpu:
            X = Variable(image['image'], volatile=True).cuda()
        else:
            X = Variable(image['image'], volatile=True)
        net.eval()
        y_pred = net(X)
        mask = resize(np.squeeze(y_pred.data.numpy()), tuple(
            int(i) for i in image['orig_size']), mode='constant', preserve_range=True)

        predictions[image['file_name'][0]] = mask

        if visualize:
            tqdm.write(
                f'Visualizing results for {image.get("file_name")[0]}, close to continue...')

            masks = watershed_segment(mask > cutoff)
            combined_mask = np.zeros(tuple(int(i)
                                           for i in image['orig_size']), dtype=np.int32)
            for i, seg_mask in enumerate(masks):
                combined_mask = np.maximum(combined_mask, (seg_mask * (i + 1)))
            img_with_labels = label2rgb(
                combined_mask, image=np.squeeze(np.array(image['orig_image'])))

            imshow(img_with_labels)
            plt.show()
    return predictions


def unnormalize(image, mean, std):
    rev_norm = image
    for t, m, s in zip(rev_norm, mean, std):
        t.mul_(s).add_(m)
    return rev_norm


def rle_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(pred, cutoff=0.5):
    lab_img = list(watershed_segment(pred > cutoff))
    for img in lab_img:
        yield rle_encode(img)


def watershed_segment(image):
    distance = ndi.distance_transform_edt(image)
    local_max = peak_local_max(
        distance, min_distance=4, labels=image, indices=False)
    markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance, markers, mask=image)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(image.shape, dtype='uint8')
        mask[labels == label] = 1
        yield mask


def generate_submission(predictions, file_name, cutoff=0.5,):
    new_image_ids = []
    rles = []
    for image_id, pred in predictions.items():
        rle = list(prob_to_rles(pred, cutoff))
        rles.extend(rle)
        new_image_ids.extend([image_id] * len(rle))
    sub_path = os.path.join('submissions', f'{file_name}.csv')
    with open(sub_path, 'w') as f:
        f.write('ImageId,EncodedPixels\n')
        for image_id, run_length in zip(new_image_ids, rles):
            output_rle = ' '.join(str(item) for item in run_length)
            f.write(f'{image_id},{output_rle}\n')
    tqdm.write(
        f'Submission with length {len(new_image_ids)} saved to {sub_path}.')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='use_gpu',
                      default=False, help='Use CUDA (Defaults to false)')
    parser.add_option('-v', '--viz', action='store_true', dest='viz', default=False,
                      help='Visualize predictions at each step (Defaults to false)')

    (options, args) = parser.parse_args()

    with open('predict_config.yml', 'r') as f:
        cfg = yaml.load(f)

    net = UNet(cfg['model']['channels'])

    if options.use_gpu:
        net.cuda()
    gen_cfg = cfg['misc']
    model_cfg = cfg['model']

    print(f'Loading model at {gen_cfg.get("model_path")}.')
    net.load_state_dict(torch.load(gen_cfg['model_path']))
    print('Model loaded.')

    preds = predict(net, cfg['data'], gen_cfg['batch_size'],
                    model_cfg['channels'], options.viz, gen_cfg['cutoff'], options.use_gpu)

    if gen_cfg['make_submission']:
        generate_submission(
            preds, gen_cfg['submission_name'], gen_cfg['cutoff'])
