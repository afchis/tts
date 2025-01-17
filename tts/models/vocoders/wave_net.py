import torch
import torch.nn as nn

from .blocks.wave_net_parts import CasualConvBlock, ResudialDilatedBlock


class WaveNet(nn.Module):
    def __init__(
        self,
        num_features=256,
        num_blocks=30,
        casual_kernel_size=64,
        dilated_kernel_size=3,
        skip_channels_size=64
    ):
        super().__init__()
        self.casual_deconv = nn.ConvTranspose1d(
            in_channels=num_features,
            out_channels=skip_channels_size,
            kernel_size=casual_kernel_size,
            stride=casual_kernel_size
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.casual_deconv(x)
        raise NotImplementedError("class WaveNet")

