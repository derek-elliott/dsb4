# Look at me and try to figure it out! https://arxiv.org/pdf/1505.04597.pdf
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, n_channels, dropout=0.1):
        super(UNet, self).__init__()
        self.inconv = DoubleConv3(n_channels, 64, dropout)
        self.down1 = Down(64, 128, dropout)
        self.down2 = Down(128, 256, dropout)
        self.down3 = Down(256, 512, dropout)
        self.down4 = Down(512, 1024, dropout)
        self.up1 = Up(1024, 512, dropout)
        self.up2 = Up(512, 256, dropout)
        self.up3 = Up(256, 128, dropout)
        self.up4 = Up(128, 64, dropout)
        self.outconv1 = nn.Conv2d(64, 1, 1)
        self.outact = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.outconv1(x9)
        x = self.outact(x10)
        return x

class Down(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv3(in_channel, out_channel, dropout)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(Up, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        # self.conv1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = DoubleConv3(in_channel, out_channel, dropout)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.center_crop(x2, x1.size()[2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv2(x)
        return x

class DoubleConv3(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(DoubleConv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(p=dropout),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(p=dropout),
            nn.ELU(inplace=True)
        )


    def forward(self, x):
        x = self.conv(x)
        return x
