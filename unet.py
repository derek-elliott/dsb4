# Look at me and try to figure it out! https://arxiv.org/pdf/1505.04597.pdf
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, n_channels, dropout=0.01):
        super(UNet, self).__init__()
        self.inconv = DoubleConv3(n_channels, 64, dropout)
        self.pool1 = Pool()
        self.conv1 = DoubleConv3(64, 128, dropout)
        self.pool2 = Pool()
        self.conv2 = DoubleConv3(128, 256, dropout)
        self.pool3 = Pool()
        self.conv3 = DoubleConv3(256, 512, dropout)
        self.pool4 = Pool()
        self.conv4 = DoubleConv3(512, 1024, dropout)
        self.up1 = ConvUp2(1024, 512)
        self.conv5 = DoubleConv3(1024, 512, dropout)
        self.up2 = ConvUp2(512, 256)
        self.conv6 = DoubleConv3(512, 256, dropout)
        self.up3 = ConvUp2(256, 128)
        self.conv7 = DoubleConv3(256, 128, dropout)
        self.up4 = ConvUp2(128, 64)
        self.conv8 = DoubleConv3(128, 64, dropout)
        self.conv9 = nn.Conv2d(64, 2, 1)
        self.outconv = nn.Conv2d(2, 1, 1)
        self.outact = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inconv(x)
        p1 = self.pool1(x1)
        c1 = self.conv1(p1)
        p2 = self.pool2(c1)
        c2 = self.conv2(p2)
        p3 = self.pool3(c2)
        c3 = self.conv3(p3)
        p4 = self.pool4(c3)
        c4 = self.conv4(p4)
        u1 = self.up1(c4, c3)
        c5 = self.conv5(u1)
        u2 = self.up2(c5, c2)
        c6 = self.conv6(u2)
        u3 = self.up3(c6, c1)
        c7 = self.conv7(u3)
        u4 = self.up4(c7, x1)
        c8 = self.conv8(u4)
        c9 = self.conv9(c8)
        out = self.outconv(c9)
        out = self.outact(out)
        return out

class DoubleConv3(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(DoubleConv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvUp2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvUp2, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.center_crop(x2, x1.size()[2])
        x = torch.cat([x2, x1], dim=1)
        return x

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(x)
        return x
