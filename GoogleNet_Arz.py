from turtle import forward
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

class ReduceConvBlock(nn.Module):
    def __init__(self, input_feats, output_feats1, output_feats2, kernel, stride=1, padding=0) -> None:
        super(ReduceConvBlock, self).__init__()
        self.redConv = nn.Sequential(
            nn.Conv2d(in_channels=input_feats, out_channels=output_feats1, kernel_size=1, stride=stride, padding=padding), # Channel Reduction
            nn.ReLU(),
            nn.Conv2d(in_channels=input_feats, out_channels=output_feats2, kernel_size=kernel, stride=stride, padding=padding),
            nn.ReLU()
        )
    def forward(self, inp):
        x = self.redConv(inp)
        return inp

class AuxilaryClassifier(nn.Module):
    def __init__(self, input_feats, classes) -> None:
        super(AuxilaryClassifier, self).__init__()
        self.avgpooling = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels=input_feats, out_channels=128, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=4*4*128, out_features=1024)
        self.dropout = nn.Dropout(p=0.7)
        self.classifier = nn.Linear(in_features=1024, out_features=classes)
    def forward(self, inp):
        N = inp.shape[0]
        x = self.avgpooling(inp)
        x = self.conv(x)
        x = self.relu(x)
        x = x.reshape(N,-1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

class Inception(nn.Module):
    def __init__(self, current_input, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, pool_proj_out) -> None:
        super(Inception, self).__init__()
        self.conv1 = ConvBlock(input_feats=current_input, output_feats=out_1x1, kernel=1, stride=1, padding=0)
        self.conv2 = ReduceConvBlock(input_feats=current_input, output_feats1=out_3x3_reduce, output_feats2=out_3x3, kernel=3, padding=1)
        self.conv3 = ReduceConvBlock(input_feats=current_input, output_feats1=out_5x5_reduce, output_feats2=out_5x5, kernel=5, padding=2)

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Conv2d(in_channels=current_input, out_channels=pool_proj_out, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, inp):
        out1 = self.conv1(inp)
        out2 = self.conv2(inp)
        out3 = self.conv3(inp)
        out4 = self.pool_proj(inp)

        x = torch.cat([out1, out2, out3, out4], dim=1)
        return x

class GoogleNet(nn.Module):
    def __init__(self, input_feats = 3, class_num=1000) -> None:
        super(GoogleNet, self).__init__()

        # Conv blocks
        self.conv1 = ConvBlock(input_feats=input_feats, output_feats=64, kernel=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            ConvBlock(input_feats=64, output_feats=64, kernel=1, stride=1, padding=0),
            ConvBlock(input_feats=64, output_feats=192, kernel=3, stride=1, padding=1)
        )

        # Inception blocks
        self.inception_3a = Inception(current_input=192, out_1x1=64, out_3x3_reduce=96, out_3x3=128, out_5x5_reduce=16, out_5x5=32, pool_proj_out=32)
        self.inception_3b = Inception(current_input=256, out_1x1=128, out_3x3_reduce=128, out_3x3=192, out_5x5_reduce=32, out_5x5=96, pool_proj_out=64)
        self.inception_4a = Inception(current_input=480, out_1x1=192, out_3x3_reduce=96, out_3x3=208, out_5x5_reduce=16, out_5x5=48, pool_proj_out=64)
        self.inception_4b = Inception(current_input=512, out_1x1=160, out_3x3_reduce=112, out_3x3=224, out_5x5_reduce=24, out_5x5=64, pool_proj_out=64)
        self.inception_4c = Inception(current_input=512, out_1x1=128, out_3x3_reduce=128, out_3x3=256, out_5x5_reduce=24, out_5x5=64, pool_proj_out=64)
        self.inception_4d = Inception(current_input=512, out_1x1=112, out_3x3_reduce=114, out_3x3=288, out_5x5_reduce=32, out_5x5=64, pool_proj_out=64)
        self.inception_4e = Inception(current_input=528, out_1x1=256, out_3x3_reduce=160, out_3x3=320, out_5x5_reduce=32, out_5x5=128, pool_proj_out=128)
        self.inception_5a = Inception(current_input=832, out_1x1=256, out_3x3_reduce=160, out_3x3=320, out_5x5_reduce=32, out_5x5=128, pool_proj_out=128)
        self.inception_5b = Inception(current_input=832, out_1x1=384, out_3x3_reduce=192, out_3x3=384, out_5x5_reduce=48, out_5x5=128, pool_proj_out=128)

        # Classifers
        self.aux_classifier1 = AuxilaryClassifier(input_feats=512, classes=class_num)
        self.aux_classifier2 = AuxilaryClassifier(input_feats=528, classes=class_num)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=7)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024*7*7, out_features=class_num)
        )
    
    def forward(self, inp):
        N = inp.shape[0]
        x = self.conv1(inp)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool1(x)
        x = self.inception_4a(x)
        out1 = self.aux_classifier1(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        out2 = self.aux_classifier2(x)
        x = self.inception_4e(x)
        x = self.maxpool1(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.classifier(x)
        if self.training == True:
            return [x, out1, out2]
        else:
            return x