import torch
import torch.nn as nn
import torch.nn.functional as F



class PReLUConv(nn.Module):
    def __init__(self, nchan, inplace):
        super(PReLUConv, self).__init__()
        self.prelu = nn.ReLU(inplace=inplace)
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)

    def forward(self, x):
        out = self.prelu(self.conv(x))
        return out

def _make_nConv(nchan, depth, inplace):
    layers = []
    for _ in range(depth):
        layers.append(PReLUConv(nchan, inplace))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, inplace):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=inplace)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.conv1(x)
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,
                         x, x, x, x, x, x, x, x), 0)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, inplace):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.ops = _make_nConv(outChans, nConvs, inplace)

    def forward(self, x):
        down = self.relu1(self.down_conv(x))
        out = self.ops(down)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, inplace):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=2, stride=2)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.ops = _make_nConv(outChans, nConvs, inplace)

    def forward(self, x, skipx):
        out = self.relu1(self.up_conv(x))
        xcat = torch.cat((out, skipx), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, batchSize, inplace):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=inplace)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        # chan0 = torch.cat((out[0][0], out[1][0]), 0)
        # chan0 = chan0.view(chan0.numel()).unsqueeze(0)
        # chan1 = torch.cat((out[1][0], out[1][1]), 0)
        # chan1 = chan1.view(chan1.numel()).unsqueeze(0)
        # print(chan1.size())
        # out = torch.cat((chan0, chan1), 0)
        print(out.size())
        out = F.softmax(out.squeeze())
        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, batchSize, inplace=True):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, inplace)
        self.down_tr32 = DownTransition(16, 1, inplace)
        self.down_tr64 = DownTransition(32, 2, inplace)
        self.down_tr128 = DownTransition(64, 3, inplace)
        self.down_tr256 = DownTransition(128, 3, inplace)
        self.up_tr256 = UpTransition(256, 256, 3, inplace)
        self.up_tr128 = UpTransition(256, 128, 2, inplace)
        self.up_tr64 = UpTransition(128, 64, 1, inplace)
        self.up_tr32 = UpTransition(64, 32, 1, inplace)
        self.out_tr = OutputTransition(32, batchSize, inplace)

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
