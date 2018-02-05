# https://arxiv.org/pdf/1604.02677.pdf
# https://github.com/lisjin/dcan-tensorflow/blob/master/tf-dcan/bbbc006.py

# TODO Implement the double loss for contours and segments

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class DCAN(nn.Module):
    def __init__(self, n_channels, dropout=0.2):
        super(DCAN, self).__init__()

        self.layers = nn.Sequential(
            Layer(n_channels, 64, 2, dropout),
            Layer(64, 128, 2, dropout),
            Layer(128, 256, 2, dropout)
            )
        # Out 1
        self.conv4 = nn.Conv_3(256, 512, dropout)
        self.pool4 = nn.MaxPool2d(2, stride=1)
        # Out 2
        self.conv5 = Conv_3(512, 512, dropout)
        self.pool5 = nn.MaxPool2d(2, stride=1)
        # Out 3
        self.conv6 = nn.Conv_3(512, 1024, dropout)

        self.c_out1 = Out(512, 8, dropout)
        self.c_out2 = Out(512, 8, dropout)
        self.c_out3 = Out(1024, 8, dropout)

        self.s_out1 = Out(512, 8, dropout)
        self.s_out2 = Out(512, 8, dropout)
        self.s_out3 = Out(1024, 8, dropout)

    def forward(self, x):
        x1 = self.conv4(self.layers(x))
        x2 = self.conv5(self.pool4(x1))
        x3 = self.conv6(self.pool5(x2))

        contour_output = self.c_out1(x1) + self.c_out2(x2) + self.c_out3(x3)
        segment_output = self.s_out1(x1) + self.s_out2(x2) + self.s_out3(x3)

        return F.Softmax(contour_output), F.Softmax(segment_output)

class Layer(nn.Module):
    def __init__(self, in_channel, out_channel, stride, dropout):
        super(Layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1)
            nn.Dropout2d(p=dropout)
            nn.ELU(inplace=True)
            nn.MaxPool2d(2, stride=stride)
        )

    def forward(self, x):
        return self.conv(x)

class Out(nn.Module):
    def __init__(self, in_channel, stride, dropout):
        super(Out, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, 1, 3, stride=stride, padding=1),
            nn.Dropout2d(p=dropout),
            nn.ELU(inplace=True),
            nn.Conv2d(1,1,1, padding=1),
            nn.Dropout2d(p=dropout),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Conv_3(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding = 1),
            nn.Dropout2d(p=dropout),
            nn.ELU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
