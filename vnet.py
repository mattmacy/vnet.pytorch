import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x

class LUConv(nn.Module):
    def __init__(self, nchan, inplace):
        super(LUConv, self).__init__()
        if inplace:
            self.prelu = nn.ReLU(inplace=inplace)
        else:
            self.prelu = nn.PReLU()
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(nchan)

    def forward(self, x):
        out = self.prelu(self.bn1(self.conv(x)))
        return out


def _make_nConv(nchan, depth, inplace):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, inplace))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, inplace):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(16)
        if inplace:
            self.relu1 = nn.ReLU(inplace=inplace)
        else:
            self.relu1 = nn.PReLU()

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, inplace, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans)
        self.do1 = passthrough

        if inplace:
            self.relu1 = nn.ReLU(inplace=inplace)
            self.relu2 = nn.ReLU(inplace=inplace)
        else:
            self.relu1 = nn.PReLU()
            self.relu2 = nn.PReLU()
        if dropout:
            self.do1 = nn.Dropout3d(p=0.2)
        self.ops = _make_nConv(outChans, nConvs, inplace)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, inplace, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.do1 = passthrough
        if inplace:
            self.relu1 = nn.ReLU(inplace=inplace)
            self.relu2 = nn.ReLU(inplace=inplace)
        else:
            self.relu1 = nn.PReLU()
            self.relu2 = nn.PReLU()
        if dropout:
            self.do1 = nn.Dropout3d(p=0.2)
        self.ops = _make_nConv(outChans, nConvs, inplace)

    def forward(self, x, skipx):
        out = self.do1(x)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, batchSize, inplace, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.batchSize = batchSize
        if inplace:
            self.relu1 = nn.ReLU(inplace=inplace)
        else:
            self.relu1 = nn.PReLU()
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax


    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // 2, 2)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, batchSize, inplace=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, inplace)
        self.down_tr32 = DownTransition(16, 1, inplace)
        self.down_tr64 = DownTransition(32, 2, inplace)
        self.down_tr128 = DownTransition(64, 3, inplace, dropout=True)
        self.down_tr256 = DownTransition(128, 3, inplace, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 3, inplace, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, inplace, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, inplace)
        self.up_tr32 = UpTransition(64, 32, 1, inplace)
        self.out_tr = OutputTransition(32, batchSize, inplace, nll)


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
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out
