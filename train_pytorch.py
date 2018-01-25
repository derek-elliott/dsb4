#!/usr/bin/env python
import os
import sys
from datetime import datetime
from optparse import OptionParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset_pytorch import NucleiDataset
from eval_pytorch import eval_net
from transforms_pytorch import Rescale, ToTensor
from unet_pytorch import UNet


def train(net, root_path, epochs=50, batch_size=32, lr=0.1, val_split=0.1, shuffle=False, seed=42, early_stop_loss=5):
    transform = transforms.Compose([Rescale((128, 128)), ToTensor()])
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    checkpoint = f'models/pt-model-dsbowl{now}.pth'

    dataset = NucleiDataset(root_path, channels=3, transform=transform)
    N_train = len(dataset)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    last_epoch_loss = 0
    unchanged_loss_run = 0

    for epoch in range(epochs):
        train_idx, val_idx = train_valid_split(
            dataset, val_split, shuffle, seed)
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler)
        val_data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=val_sampler)

        print(f'Epoch {epoch+1}/{epochs}')

        epoch_loss = 0

        for i, image in enumerate(train_data_loader):
            X = Variable(image['image'])
            y = Variable(image['combined_mask'])

            y_pred = net(X)
            probs = F.sigmoid(y_pred)
            probs_flat = probs.view(-1)

            y_flat = y.views(-1)

            loss = criterion(probs_flat, y_flat.float())
            epoch_loss += loss.data[0]

            print(f'{i*batch_size/N_train} --- loss: {loss.data[0]}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_score = eval_net(net, val_data_loader)
        print(
            f'Epoch finished --- Loss: {epoch_loss/batch_size}  Mean IOU: {val_score}')
        if last_epoch_loss < epoch_loss:
            unchanged_loss_run += 1
        else:
            torch.save(net.state_dict(), checkpoint)
        if unchanged_loss_run < early_stop_loss:
            print('Stopping early...')
        last_epoch_loss = epoch_loss


def train_valid_split(dataset, val_split, shuffle=False, seed=42):
    length = len(dataset)
    indicies = list(range(1, length))

    if shuffle:
        random.seed(seed)
        random.shuffle(indicies)

    split = np.floor(val_split * length)
    return indicies[int(split):], indicies[:int(split)]


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('-p', '--path', dest='root_path', help='path to data')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=32,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-s', '--split', dest='val_split',
                      default=0.1, type='float', help='Train-val split percentage')
    parser.add_option('-S', '--shuffle', dest='shuffle',
                      default=False, help='Shuffle the images')
    parser.add_option('-r', '--seed', dest='seed',
                      default=42, type='int', help='Random seed for shuffle')
    parser.add_option('-L', '--loss', dest='early_stop_loss',
                      default=5, type='int', help='Epocs in a row with no decrease in loss to stop early')

    (options, args) = parser.parse_args()
    net = UNet(3, 0.2)

    try:
        train(net, options.root_path, options.epochs, options.batch_size, options.lr, options.val_split, options.shuffle,
              options.seed, options.early_stop_loss)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
