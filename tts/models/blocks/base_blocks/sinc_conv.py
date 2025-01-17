import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=80,
                 kernel_size=251,
                 sample_rate=16000,
                 window="hamming_window",
                 min_freq=20.,
                 min_band=50.):
        super().__init__()
        self.min_freq = min_freq
        self.min_band = min_band
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size // 4
        self.padding = 0
        self.sample_rate = sample_rate
        self._init()

    @staticmethod
    def _to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def _to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def _get_freq_points(self, sample_rate, num_filters):
        low_mel = self._to_mel(self.min_freq)
        high_mel = self._to_mel(sample_rate // 2)
        mel_points = np.linspace(low_mel, high_mel, num_filters+1)
        freq_points = self._to_hz(mel_points) / self.sample_rate
        return freq_points

    def _init(self):
        freq_points = self._get_freq_points(sample_rate=self.sample_rate,
                                           num_filters=self.out_channels)
        self.bands_low_freq = nn.Parameter(torch.Tensor(freq_points[:-1].reshape(-1, 1)))
        self.bands = nn.Parameter(torch.Tensor(np.diff(freq_points).reshape(-1, 1)))
        window = torch.hamming_window(self.kernel_size)
        self.register_buffer("window", window)
        n = (self.kernel_size - 1) / 2
        n = 2 * math.pi * torch.arange(-n, n+1).view(1, -1)
        self.register_buffer("n", n)

    def sinc(self, x):
        sinc = torch.sinc(x)
        sinc[:, self.kernel_size // 2] = 1.
        return sinc

    def _get_weight_from_bands(self):
        weight = list()
        for low_freq, band in zip(self.bands_low_freq, self.bands):
            f1 = torch.abs(low_freq)
            f2 = torch.abs(low_freq) + torch.abs(band)
            kernel = 2 * f2 * self.sinc(2 * math.pi * f2 * self.n * self.sample_rate) \
                   - 2 * f1 * self.sinc(2 * math.pi * f1 * self.n * self.sample_rate)
            kernel = kernel / kernel.max()
            weight.append(kernel)
        weight = torch.stack(weight) * self.window
        return weight

    def forward(self, x):
        weight = self._get_weight_from_bands()
        out = F.conv1d(x, weight, bias=None, stride=self.stride, padding=self.padding)
        return out


if __name__ == "__main__":
    x = torch.rand([1, 1, 32000])
    model = SincConv()
    out = model(x)


