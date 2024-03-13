import torch
from torch import nn
import math

class ResBlock(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        # self.mp = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        if self.downsample:
            residual = self.conv_downsample(residual)
        x = x + residual
        x = self.prelu2(x)
        # return self.mp(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.PReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.PReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA(nn.Module):
    def __init__(self,input_channel,hidden_size):
        super(ECAPA, self).__init__()
        self.input_layer=nn.Sequential(
            nn.Conv1d(input_channel, hidden_size, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_size)
        )
        self.layer1 = Bottle2neck(hidden_size, hidden_size, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(hidden_size, hidden_size, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(hidden_size, hidden_size, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * hidden_size, hidden_size, kernel_size=1)
        self.relu=nn.PReLU()

    def forward(self,x):
        x=self.input_layer(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        return x


if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    net=ECAPA(512,512)
    macs, params = get_model_complexity_info(net, (512,1000), as_strings=True, print_per_layer_stat=True)
    print(macs)
    print(params)