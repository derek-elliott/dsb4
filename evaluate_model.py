import numpy as np
import torch
from torch.autograd import Variable
from loss import dice_coef


def eval_net(net, dataset, use_gpu=False):
    total = 0
    for i, image in enumerate(dataset):
        if use_gpu:
            X = Variable(image['image']).cuda()
            y = Variable(image['combined_mask']).cuda()
        else:
            X = Variable(image['image'])
            y = Variable(image['combined_mask'])

        y_pred = net(X)

        score = dice_coef(y_pred, y)
        total += score
    return total / len(dataset)
