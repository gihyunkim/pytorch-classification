import torch.nn as nn
import torch.nn.functional as F
import torch
from models.common import ConvBn
from torchsummary import summary

'''VGGNET for CIFAR100'''
class VGG(nn.Module):
    def __init__(self, num_blocks, num_classes=100):
        super().__init__()
        self.block1 = self.make_block(3, 64, num_blocks[0])# 32x32
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = self.make_block(64, 128, num_blocks[1])# 16x16
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = self.make_block(128, 256, num_blocks[2])# 8x8
        # self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = self.make_block(256, 512, num_blocks[3])
        # self.pool4 = nn.MaxPool2d(2, 2)
        self.block5 = self.make_block(512, 512, num_blocks[4])
        # self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8*8*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.out = nn.Linear(4096, num_classes)

    def make_block(self, in_ch, out_ch, num_block):
        layers = []
        layers.append(ConvBn(in_ch, out_ch)) # first layer
        for _ in range(num_block-1): # remainder layer
            layers.append(ConvBn(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x) # first block
        x = self.pool1(x)
        x = self.block2(x) # second block
        x = self.pool2(x)
        x = self.block3(x) # third block
        x = self.block4(x) # forth block
        x = self.block5(x) # fifth block
        x_flat = x.view(-1, 8*8*512) # to 1 dim vector
        x_fc = self.fc1(x_flat) # fc layer
        x_fc = self.fc2(x_fc)
        out = self.out(x_fc)
        return out

def VGG16():
    return VGG([2, 2, 3, 3, 3])

def VGG19():
    return VGG([2, 2, 4, 4, 4])

if __name__ == "__main__":
    vgg = VGG
    summary(VGG16(), input_size=(3, 32, 32), batch_size=1)