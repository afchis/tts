import typing as t

import torch
import torch.nn as nn

from .base_blocks import MultiHeadAttention, FeedForward


class FFTBlock(nn.Module):
    def __init__(self, in_features, hid_features, num_heads):
        super().__init__()
        self.multi_head = MultiHeadAttention(in_features=in_features,
                                             num_heads=num_heads)
        self.feed_forward = FeedForward(in_features=in_features,
                                        hid_features=hid_features,
                                        conv=True)
        self.linear_norm = nn.LayerNorm(normalized_shape=in_features)

    def forward(self, x):
        resudial = x
        x = self.multi_head(x)
        x = self.linear_norm(resudial + x)
        resudial = x
        x = self.feed_forward(x)
        x = self.linear_norm(resudial + x)
        return x


class FastSpeech2Predictor(nn.Module):
    def __init__(self, in_features, out_features, target=None):
        super().__init__()
        self.target = target
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_features,
                      out_channels=in_features*4,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.InstanceNorm1d(num_features=in_features*4),
            nn.Dropout(0.1)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=in_features*4,
                      out_channels=in_features,
                      kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.InstanceNorm1d(num_features=in_features),
            nn.Dropout(0.1)
            )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.squeeze(-1)
        x = x.transpose(1, 2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        if self.target == "duration":
            x = torch.clip(x, min=1).long()
        return x


class LenghtRegulator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def repeat_sequence_from_duration(self, att_out, duration):
        out = list()
        i = 0
        for batch_att_out, batch_duration in zip(att_out, duration):
            a = batch_att_out
            d = batch_duration
            batch = [a_i.repeat(torch.round(d_i), 1) for a_i, d_i in zip(a, d)]
            batch = torch.cat(batch, dim=0)
            out.append(batch)
            i += 1
        out = nn.utils.rnn.pad_sequence(out, batch_first=True)
        return out
    
    def forward(self, att_out, duration, alpha=1):
        """
        args:
            att_out: Hidden state of phoneme after encoder
            duration: Out after duration predictor
            alpha: Speed-control parameter
        """
        duration = torch.mul(duration, alpha)
        out = self.repeat_sequence_from_duration(att_out, duration)
        return out


class VarianceAdaptor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.lenght_regulator = LenghtRegulator()
        self.predictor_duration = FastSpeech2Predictor(in_features=in_features,
                                                       out_features=1,
                                                       target="duration")
        self.predictor_pitch = FastSpeech2Predictor(in_features=in_features,
                                                    out_features=in_features)
        self.predictor_energy = FastSpeech2Predictor(in_features=in_features,
                                                     out_features=in_features)
        self.predictor_last = FastSpeech2Predictor(in_features=in_features,
                                                   out_features=in_features)

    def forward(self, att_out):
        duration = self.predictor_duration(att_out).squeeze(-1)
        out = self.lenght_regulator(att_out, duration, alpha=1)
        pitch = self.predictor_pitch(out)
        energy = self.predictor_energy(out)
        last = self.predictor_last(out)
        for item in [pitch, energy, last]:
            out += item
        # out = last
        return out

