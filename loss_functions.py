# Pulled from https://github.com/milesial/Pytorch-UNet/blob/master/myloss.py

import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn.modules.loss import _Loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input, target) + 0.0001
        self.union = torch.sum(input) + torch.sum(target) + 0.0001

        t = 2 * self.inter.float() / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = Variable(torch.FloatTensor(1).cuda().zero_())
    else:
        s = Variable(torch.FloatTensor(1).zero_())

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class DiceLoss(_Loss):
    def forward(self, input, target):
        return 1 - dice_coeff(input, target)

class IoULoss(_Loss):
    def forward(self, pred, gt, cutoff):
        pred = (pred > cutoff)
        mask = (gt != 255)
        gt = (gt == 1)
        union = (gt | pred)[mask].long().sum()
        if not union:
            return 0.
        else:
            intersection = (gt & pred)[mask].long().sum()
            return 1. - intersection / union
