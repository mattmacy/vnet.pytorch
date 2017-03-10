import torch
import torch.nn as nn
import torch.nn.functional as F


class InputTransition(nn.Module):
    def __init__(self, outChannels):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, outChannels, kernel_size=5, padding=2)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.conv1(x)
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 1)
        out = F.prelu(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChannels, nConvs):
        super(DownTransition, self).__init__()
        outChannels = 2*inChannels
        self.down_conv = nn.Conv3d(inChannels, outChannels, kernel_size=2, stride=2)
        self.convs = []
        for _ in range(nConvs):
            self.convs.append(nn.Conv3d(outChannels, outChannels, kernel_size=5, padding=2))

    def forward(self, x):
        out = F.prelu(self.down_conv(x))
        for conv in self.convs:
            out = F.prelu(conv(out))
        out = F.prelu(torch.add(out, x))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.convs = []
        for _ in range(nConvs):
            self.convs.append(nn.Conv3d(outChans, outChans, kernel_size=5, padding=2))

    def forward(self, x, skipx):
        out = F.prelu(self.up_conv(x))
        out = xcat = torch.cat((out, skipx), 1)
        for conv in self.convs:
            out = F.prelu(conv(out))
        out = F.prelu(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChannels):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChannels, 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = F.prelu(self.conv1(x))
        # do softmax of 1x1 convolution
        out = F.softmax(self.conv2(out))
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16)
        self.down_tr32 = DownTransition(16, 1)
        self.down_tr64 = DownTransition(32, 2)
        self.down_tr128 = DownTransition(64, 3)
        self.down_tr256 = DownTransition(128, 3)
        self.up_tr256 = UpTransition(256, 256, 3)
        self.up_tr128 = UpTransition(256, 128, 2)
        self.up_tr64 = UpTransition(128, 64, 1)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = OutputTransition(32)

    # The network topology as described in the diagram
    # in the VNet paper
    # def __init__(self):
    #     super(VNet, self).__init__()
    #     self.in_tr =  InputTransition(16)
    #     # the number of convolutions in each layer corresponds
    #     # to what is in the actual prototxt, not the intent
    #     self.down_tr32 = DownTransition(16, 2)
    #     self.down_tr64 = DownTransition(32, 3)
    #     self.down_tr128 = DownTransition(64, 3)
    #     self.down_tr256 = DownTransition(128, 3)
    #     self.up_tr256 = UpTransition(256, 3)
    #     self.up_tr128 = UpTransition(128, 3)
    #     self.up_tr64 = UpTransition(64, 2)
    #     self.up_tr32 = UpTransition(32, 1)
    #     self.out_tr = OutputTransition(16)
    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr128(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
