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
from skimage.io import imshow

import datasets
import matplotlib.pyplot as plt
from predict_dataset import (NucleiPredictDataset, PredictNormalize,
                             PredictRescale, PredictToTensor)
from tqdm import tqdm
from unet import UNet


def predict(net, data_cfg, batch_size, vizualize, cutoff=0.5, use_gpu):
    transform = transforms.Compose(
        [PredictRescale((data_cfg['img_height'], data_cfg['img_width'])),
         PredictNormalize(0, 255),
         PredictToTensor()])

    dataset = NucleiPredictDataset(
        data_cfg['root_path'], channels=3, transform=transform)

    predictions = {}

    bar = tqdm(dataset)
    for i, image in enumerate(bar):
        bar.set_description(desc=f'Image {i+1}/{len(dataset)}')
        if use_gpu:
            X = Variable(image['image'], volatile=True).cuda()
        else:
            X = Variable(image['image'], volatile=True)
        net.eval()
        y_pred = net(X)
        mask = transform.resize(y_pred.from_numpy().transpose((1, 2, 0)), image['orig_size']
        predictions[image['file_name']] = mask

        if visualize:
            print(f'Visualizing results for {image['file_name']}, close to continue...')
            fig=plt.figure()
            a = fig.add_subplot(1,2,1)
            a.set_title('Input image')
            imshow(image['image'])

            masks = watershed_segment(y_pred > cutoff)
            combined_mask = mask = np.zeros(y_pred.shape, dtype=np.int32)
            for i, mask in enumerate(masks):
                combined_mask = np.maximum(combined_mask, (mask * (i + 1)))

            b = fig.add_subplot(1,2,2)
            b.set_title('Output mask')
            imshow(combined_mask)

            plt.show()
    return predictions

def rle_encode(x):
    dots=np.where(x.T.flatten() == 1)[0]
    run_lengths=[]
    prev=-2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev=b
    return run_lengths

def prob_to_rles(pred, cutoff=0.5):
    lab_img=list(watershed_segment(pred > cutoff))
    for img in lab_img:
        yield rle_encode(img)

def watershed_segment(image):
    distance=ndi.distance_transform_edt(image)
    local_max=peak_local_max(
        distance, min_distance=4, labels=image, indices=False)
    markers=ndi.label(local_max, structure=np.ones((3, 3)))[0]
    labels=watershed(-distance, markers, mask=image)
    for label in np.unique(labels):
        if label == 0:
            continue
        mask=np.zeros(image.shape, dtype='uint8')
        mask[labels == label]=1
        yield mask

def generate_submission(predictions, cutoff=0.5, file_name):
    new_image_ids=[]
    rles=[]
    for image_id, pred in predictions.items():
        rle=list(prob_to_rles(pred, cutoff))
        rles.extend(rle)
        new_image_ids.extend([image_id] * len(rle))
    sub_path = os.path.join('submissions', f'{file_name}.csv')
    with open(sub_path, 'w') as f:
        f.write('ImageId,EncodedPixels\n')
        for image_id, run_length in zip(new_image_ids, rles):
            output_rle=' '.join(str(item) for item in run_length)
            f.write(f'{image_id},{output_rle}\n')
    print(f'Submission with length {len(new_image_ids)} saved to {sub_path}.')

if __name__ == '__main__':
    parser=OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='use_gpu',
                      defaut=False, help='Use CUDA (Defaults to false)')
    parser.add_option('-v', '--viz', action='store_true', dest='viz', default=Fase, help='Visualize predictions at each step (Defaults to false)')

    (options, args)=parser.parse_args()

    with open('predict_config.yml', 'r') as f:
        cfg=yaml.load(f)

    net=UNet(cfg['model']['channels'])

    if options.use_gpu:
        net.cuda()
    gen_cfg=cfg['misc']

    print(f'Loading model at {gen_cfg['model_path']}.')
    net.load_state_dict(torch.load(gen_cfg['model_path']))
    print('Model loaded.')

    preds = predict(net, cfg['data'], gen_cfg['batch_size'], options.viz, cutoff, options.use_gpu)

    if gen_cfg['make_submission']:
        generate_submission(preds, gen_cfg['cutoff'], gen_cfg['submission_name'])
