import torch.nn as nn
import torch.nn.functional as F

'''Base Convolution Block: Conv + Bn + Relu'''
class ConvBn(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


'''Base Convolution Block: Bn + Relu + Conv'''
class BnConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        return self.conv(F.relu(self.bn(x)))
