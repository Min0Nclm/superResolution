import torch
import torch.nn as nn

class BSConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(BSConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class ESA(nn.Module):

    def __init__(self, num_feat):
        super(ESA, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)

    def forward(self, x):

        attn = self.conv1(x)
        attn = self.sigmoid(attn)

        attn = x * attn

        attn = self.conv2(attn)
        return attn
