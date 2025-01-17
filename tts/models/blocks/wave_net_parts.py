import torch
import torch.nn as nn


class CasualConvBlock(nn.Module):
    def __init__(self):
        super().__init__(in_channels, out_channels, kernel_size)
        # self.deconv = nn.ConvTranspose1d(in_channels,
        #                                  out_channels,
        #                                  kernel_size=)

    def forward(self, x):
        raise NotImplementedError


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class ResudialDilatedBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dilated_conv = nn.Conv1d(in_channels=in_channels,
                                      out_channels=in_channels,
                                      kernel_size=2,
                                      dilation=None,
                                      padding=0,
                                      bias=False)
    
    def forward(self, x):
        x = self.dilated_conv(x)

