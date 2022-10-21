import numpy as np
import torch
from torch import max_pool2d, nn, relu, optim

class ConvBlock(nn.Module):
    def __init__(self, input_feats, output_feats, kernel, stride, padding) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_feats, out_channels=output_feats, kernel_size=kernel, stride=stride, padding=padding),
            nn.ReLU()
        )
    def forward(self, inp):
        x = self.conv(inp)
        return x