#!/usr/bin/env python
import os
import sys
import time
from datetime import datetime
from optparse import OptionParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from evaluate_model import eval_net
from loss import dice_loss
from tqdm import tqdm
from train_dataset import (ColorJitter, NucleiTrainDataset,
                           RandomHorizontalFlip, RandomResizedCrop,
                           RandomRotation, RandomVerticalFlip,
                           TrainCLAHEEqualize, TrainGrayscale, TrainNormalize,
                           TrainToTensor)
from unet import UNet


def train(net, data_cfg, model_cfg, epochs=50, batch_size=32, val_split=0.1, shuffle=False, seed=42, early_stop_loss=5, use_gpu=False):
    transform = transforms.Compose(
        [TrainCLAHEEqualize(),
         TrainGrayscale(),
         RandomResizedCrop((data_cfg['img_height'], data_cfg['img_width'])),
         RandomHorizontalFlip(prob=0.5),
         RandomVerticalFlip(prob=0.5),
         ColorJitter(brightness=1.5, contrast=1.5,
                     saturation=1.5, hue=0.25),
         RandomRotation(degrees=90),
         TrainToTensor(),
         TrainNormalize()])
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    checkpoint = f'models/pt-model-dsbowl{now}.pth'

    dataset = NucleiTrainDataset(
        data_cfg['root_path'], bad_images=data_cfg['bad_images'], channels=model_cfg['channels'], transform=transform)

    N_train = len(dataset)

    optimizer = optim.SGD(
        net.parameters(), lr=model_cfg['lr'], momentum=model_cfg['momentum'], weight_decay=model_cfg['weight_decay'])
    criterion = dice_loss

    last_epoch_loss = 0
    unchanged_loss_run = 0

    for epoch in range(epochs):
        train_idx, val_idx = train_valid_split(
            dataset, val_split, shuffle, seed)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=train_sampler, num_workers=data_cfg['workers'])
        val_data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, sampler=val_sampler, num_workers=data_cfg['workers'])

        epoch_loss = 0
        bar = tqdm(train_data_loader)
        bar.set_description(desc=f'Epoch {epoch+1}/{epochs}')
        for i, image in enumerate(bar):
            if use_gpu:
                X = Variable(image['image']).cuda()
                y = Variable(image['combined_mask']).cuda()
            else:
                X = Variable(image['image'])
                y = Variable(image['combined_mask'])
            net.train()
            y_pred = net(X)
            net.eval()

            loss = criterion(y_pred, y)
            epoch_loss += loss.data[0]

            bar.set_postfix(loss=loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # val_score = eval_net(net, val_data_loader, use_gpu)
        val_score = 0
        tqdm.write(
            f'Epoch finished --- Loss: {epoch_loss/(batch_size if batch_size != 1 else N_train)}  Dice Error: {val_score}')
        if last_epoch_loss < epoch_loss:
            unchanged_loss_run += 1
        else:
            tqdm.write(
                f'Loss decreased by {(last_epoch_loss - epoch_loss)}, saving checkpoint.')
            torch.save(net.state_dict(), checkpoint)
            unchanged_loss_run = 0
        if unchanged_loss_run > early_stop_loss:
            tqdm.write('Stopping early...')
            break
        last_epoch_loss = epoch_loss


def train_valid_split(dataset, val_split, shuffle=False, seed=42):
    length = len(dataset)
    indicies = list(range(0, length))

    if shuffle:
        random.seed(seed)
        random.shuffle(indicies)

    split = np.floor(val_split * length)
    return indicies[int(split):], indicies[:int(split)]


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='use_gpu',
                      default=False, help='Use CUDA (Defaults to false)')
    parser.add_option('-m', '--model', action='store_true', dest='load_model',
                      default=False, help='Load model from file (Defaults to false)')

    (options, args) = parser.parse_args()

    with open('train_config.yml', 'r') as f:
        cfg = yaml.load(f)

    gen_cfg = cfg['misc']

    net = UNet(cfg['model']['channels'], cfg['model']['dropout'])

    if options.use_gpu:
        net.cuda()
        cudann.benchmark = True

    if options.load_model:
        print(f'Loading model at {gen_cfg.get("model_path")}.')
        net.load_state_dict(torch.load(gen_cfg['model_path']))
        print('Model loaded.')

    try:
        train(net,
              cfg['data'],
              cfg['model'],
              gen_cfg['epochs'],
              gen_cfg['batch_size'],
              gen_cfg['val_split'],
              gen_cfg['shuffle'],
              gen_cfg['seed'],
              gen_cfg['early_stop_loss'],
              options.use_gpu)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
