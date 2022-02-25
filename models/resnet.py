import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from models.common import *
from torchsummary import summary

class ResBlock(Module):
    expansion = 1  # out_ch expansion from in_ch

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        expans_ch = out_ch * self.expansion
        self.conv1 = BnConv(in_ch, out_ch, stride=stride)
        self.conv2 = BnConv(out_ch, out_ch)

        if stride != 1 or in_ch != expans_ch:
            self.residual = BnConv(in_ch, expans_ch, kernel_size=1,
                                   stride=stride, padding=0)
        else:
            self.residual = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + self.residual(x)

class ResBottleNeckBlock(Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        expans_ch = out_ch * self.expansion
        self.conv1 = BnConv(in_ch, out_ch, kernel_size=1, padding=0)
        self.conv2 = BnConv(out_ch, out_ch, kernel_size=3,stride=stride)
        self.conv3 = BnConv(out_ch, expans_ch, kernel_size=1, padding=0)

        if stride !=1 or in_ch != expans_ch:
            self.residual = BnConv(in_ch, expans_ch, kernel_size=1, stride=stride, padding=0)
        else:
            self.residual = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + self.residual(x)

class ResNet(Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.block = block # Type of Block
        first_ch = 64
        self.tracking_ch = first_ch

        '''Stem'''
        self.conv1 = nn.Conv2d(3, first_ch, kernel_size=7, stride=1, padding=3)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1) # 8x8

        '''body'''
        self.blocks1 = self.make_blocks(first_ch, num_blocks[0], stride=1)
        self.blocks2 = self.make_blocks(first_ch*2, num_blocks[1], stride=1)
        self.blocks3 = self.make_blocks(first_ch*4, num_blocks[2], stride=2)
        self.blocks4 = self.make_blocks(first_ch*8, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(self.tracking_ch)

        '''fc layer'''
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.tracking_ch, num_classes)

    def make_blocks(self, out_ch, num_block, stride=1):
        layers = []
        layers.append(self.block(self.tracking_ch, out_ch, stride=stride)) # first layer
        self.tracking_ch = out_ch * self.block.expansion
        for _ in range(num_block - 1):
            layers.append(self.block(self.tracking_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''stem'''
        x = self.conv1(x)
        #x = self.pool1(x)

        '''body'''
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = F.relu(self.bn(x))

        '''head'''
        gap = self.gap(x)
        flat = gap.view(gap.size(0), -1)
        out = self.fc(flat)
        return out

def Resnet18():
    return ResNet(ResBlock, [2, 2, 2, 2])

def Resnet34():
    return ResNet(ResBlock, [3, 4, 6, 3])

def Resnet50():
    return ResNet(ResBottleNeckBlock, [3, 4, 6, 3])

def Resnet101():
    return ResNet(ResBottleNeckBlock, [3, 4, 23, 3])

def Resnet152():
    return ResNet(ResBottleNeckBlock, [3, 8, 36, 3])

if __name__ == "__main__":
    summary(Resnet18().cuda(), input_size=(3, 32, 32))