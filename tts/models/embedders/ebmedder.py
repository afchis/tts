import torch
import torch.nn as nn

from tts.models.blocks.base_blocks.sinc_conv import SincConv


class EmbedderWave(nn.Module):
    def __init__(self,):
        super().__init__()
        self.sinc_conv = SincConv(in_channels=1,
                                  out_channels=80,
                                  kernel_size=251,
                                  sample_rate=16000)
        self.batch_norm = nn.BatchNorm1d(80)
        self.lstm = nn.LSTM(input_size=80, hidden_size=768, num_layers=3, batch_first=True)
        self.embedding = nn.Linear(in_features=768, out_features=256)

    def forward(self, wave):
        spec = self.sinc_conv(wave)
        spec = self.batch_norm(spec).transpose(1, 2)
        spec = torch.log(100*torch.abs(spec) + 1)
        lstm_out, _ = self.lstm(spec)
        out = self.embedding(lstm_out[:, -1, :])
        out = out / torch.norm(out, dim=-1, keepdim=True)
        return out


class EmbedderMel(nn.Module):
    def __init__(self,):
        super().__init__()
        self.lstm = nn.LSTM(input_size=80, hidden_size=768, num_layers=3, batch_first=True)
        self.embedding = nn.Linear(in_features=768, out_features=256)

    def forward(self, mel):
        lstm_out, _ = self.lstm(mel)
        out = self.embedding(lstm_out[:, -1, :])
        out = out / torch.norm(out, dim=-1, keepdim=True)
        return out


if __name__ == "__main__":
    x = torch.rand([64, 1, 32000])
    model = EmbedderWave()
    out = model(x)
    print(out.shape)


