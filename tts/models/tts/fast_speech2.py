import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.models.blocks.fast_speech_parts import FFTBlock, VarianceAdaptor
from tts.models.blocks.base_blocks import PositionalEncoding
from tts.models.wave_net import WaveNet


class FastSpeech2(nn.Module):
    __input_example__ = [1, 86]
    __output_example__ = [1, 80, 138]

    def __init__(
        self,
        num_embeddings=83,
        fft_blocks=4,
        fft_num_heads=2,
    ):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=256)
        self.encoder = nn.ModuleList(
            [FFTBlock(in_features=256,
                      hid_features=2048,
                      num_heads=2)
            for _ in range(fft_blocks)]
        )
        self.variance_adaptor = VarianceAdaptor(in_features=256)
        self.positional_encoder_1 = PositionalEncoding(dim=1)
        self.positional_encoder_2 = PositionalEncoding(dim=1)
        self.mel_decoder = nn.ModuleList(
            [FFTBlock(in_features=256,
                      hid_features=256,
                      num_heads=2)
            for _ in range(fft_blocks)]
        )
        self.mel_decoder_final = nn.Linear(in_features=256, out_features=80)

    def forward(self, phoneme):
        if self.training:
            phoneme = nn.utils.rnn.pad_sequence(phoneme, batch_first=True)
        emb = self.embedder(phoneme) 
        emb = self.positional_encoder_1(emb)
        for fft_block in self.encoder:
            emb = fft_block(emb)
        att_out = self.positional_encoder_2(emb)
        out = self.variance_adaptor(att_out)
        for  fft_block in self.mel_decoder:
            out = fft_block(out)
        out = self.mel_decoder_final(out).transpose(1, 2)
        return out


if __name__ == "__main__":
    phoneme = torch.randint(low=0, high=82, size=FastSpeech2.__input_example__)
    model = FastSpeech2()
    out = model(phoneme)

    