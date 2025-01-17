import torch
import torch.nn as nn

from tts.models.blocks.tacotron_parts import Encoder, Decoder, PostNet


class Tacotron2(nn.Module):
    __input_example__ = [2, 86]
    __output_example__ = [2, 80, 138]

    def __init__(self, num_embeddings=83, embedding_dim=128, mel_dim=80):
        super().__init__()
        self.inference = False
        self.embedder = nn.Embedding(num_embeddings=num_embeddings,
                                     embedding_dim=embedding_dim)
        self.encoder_lstm = Encoder(embedding_dim=embedding_dim,
                                    num_convs=3)
        self.decoder = Decoder(mel_dim=mel_dim,
                               embedding_dim=embedding_dim)
        self.post_net = PostNet(mel_dim=mel_dim,
                                post_net_dim=512,
                                num_convs=3,
                                kernel_size=5)

    def _forward(self, text_or_phoneme, target_mels=None):
        emb = self.embedder(text_or_phoneme)
        encoder_output = self.encoder_lstm(emb)
        decoder_output, stop_values = self.decoder(encoder_output, target_mels)
        post_output = self.post_net(decoder_output)
        post_output += decoder_output
        return decoder_output, post_output, stop_values

    def forward(self, text_or_phoneme, target_mels=None):
        if self.inference:
            if len(text_or_phoneme) > 1:
                raise Exception("Размерность батча при инференсе не должна быть больше (1)")
            text_or_phoneme = torch.stack(text_or_phoneme, dim=0)
            _, post_output, _, _ = self._forward(text_or_phoneme, target_mels=None)
            return post_output
        else:
            text_or_phoneme = nn.utils.rnn.pad_sequence(text_or_phoneme, batch_first=True)
            if not target_mels is None:
                target_mels = nn.utils.rnn.pad_sequence(target_mels, batch_first=True)
            decoder_output, post_output, stop_values = self._forward(text_or_phoneme, target_mels=target_mels)
            return decoder_output, post_output, stop_values


if __name__ == "__main__":
    text = torch.randint(low=0, high=82, size=Tacotron2.__input_example__)
    model = Tacotron2()
    # model.eval()
    out = model(text)
