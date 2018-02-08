import torch

def dice_coef(pred,target, smooth=1):
    intersection = torch.sum(pred * target)
    return (2. * intersection + smooth) / (torch.sum(target) + torch.sum(pred) + smooth)

def dice_loss(pred, target):
    return -dice_coef(pred, target)
