# Look at me and try to figure it out! https://arxiv.org/pdf/1505.04597.pdf
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):

    def __init__(self, n_channels, dropout):
        super(UNet, self).__init__()
        self.inconv = Input(n_channels, 64, dropout)
        self.down1 = Down(64, 128, dropout)
        self.down2 = Down(128, 256, dropout)
        self.down3 = Down(256, 512, dropout)
        self.down4 = Down(512, 1024, dropout)
        self.up1 = Up(1024, 512, dropout)
        self.up2 = Up(512, 256, dropout)
        self.up3 = Up(256, 128, dropout)
        self.up4 = Up(128, 64, dropout)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x

class Input(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(Input, self).__init__()
        self.conv = DoubleConv(in_channel, out_channel, dropout)

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(DoubleConv, self).__init__()
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

class Down(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channel, out_channel, dropout)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.conv = DoubleConv(out_channel, out_channel, dropout)

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffX = x1.size()[2] - x2.size()[2]
        # diffY = x1.size()[3] - x2.size()[3]
        # x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
        #                 diffY // 2, int(diffY / 2)))
        print(x1.size()[1], x2.size()[1])
        x2 = self.center_crop(x2, x1.size()[1])
        print(x2.size()[1])
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
