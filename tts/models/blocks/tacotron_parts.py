import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_blocks.transformer import LocationSensetiveAttention


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0.5, activation="relu"):
        super().__init__()
        self.activations = {"relu": nn.ReLU,
                            "tanh": nn.Tanh,
                            "sigmoid": nn.Sigmoid}
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=int(kernel_size/2)),
            nn.BatchNorm1d(out_channels),
            self.activations[activation](),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_convs=3):
        super().__init__()
        conv_blocks = list()
        for i in range(num_convs):
            block = ConvBlock(in_channels=embedding_dim,
                              out_channels=embedding_dim,
                              kernel_size=5,
                              stride=1,
                              dropout=0.5,
                              activation="relu")
            conv_blocks.append(block)
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.bi_lstm = nn.LSTM(input_size=embedding_dim,
                               hidden_size=int(embedding_dim/2),
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)

    def _conv_block_forward(self, emb):
        emb = emb.transpose(1, 2)
        for conv_block in self.conv_blocks:
            emb = conv_block(emb)
        emb = emb.transpose(1, 2)
        return emb

    def _forward_train(self, emb):
        out = self._conv_block_forward(emb)
        out, _ = self.bi_lstm(out)
        return out

    def _forward_inference(self, emb):
        out = self._conv_block_forward(emb)
        out, _ = self.bi_lstm(out)
        return out

    def forward(self, emb):
        if self.training:
            return self._forward_train(emb)
        else:
            return self._forward_inference(emb)


class PreNet(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_block_1 = nn.Sequential(nn.Linear(in_features=in_features,
                                                      out_features=out_features),
                                            nn.ReLU())
        self.linear_block_2 = nn.Sequential(nn.Linear(in_features=out_features,
                                                      out_features=out_features),
                                            nn.ReLU())

    def forward(self, x):
        x = F.dropout(self.linear_block_1(x), p=0.5, training=True)
        x = F.dropout(self.linear_block_2(x), p=0.5, training=True)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 mel_dim=80,
                 embedding_dim=256,
                 pre_net_dim=256,
                 rnn_dim=1024,
                 attention_dim=128,
                 max_decoder_steps=1000,
                 rnn_dropout=0.1):
        super().__init__()
        self.max_decoder_steps = max_decoder_steps
        self.mel_dim = mel_dim
        self.embedding_dim = embedding_dim
        self.pre_net = PreNet(in_features=mel_dim,
                              out_features=pre_net_dim)
        self.dropout = nn.Dropout(p=rnn_dropout)
        self.pre_rnn = nn.LSTMCell(input_size=embedding_dim+pre_net_dim,
                                   hidden_size=rnn_dim,
                                   bias=True)
        self.loc_sens_att = LocationSensetiveAttention(rec_hidden_dim=rnn_dim,
                                                       value_dim=embedding_dim,
                                                       attention_dim=attention_dim)
        self.post_rnn = nn.LSTMCell(input_size=rnn_dim+embedding_dim,
                                    hidden_size=rnn_dim,
                                    bias=True)
        self.mel_proj = nn.Linear(in_features=rnn_dim+embedding_dim,
                                  out_features=mel_dim)
        self.stop_proj = nn.Linear(in_features=rnn_dim+embedding_dim,
                                   out_features=1)
        self.stop_value_threshold = 0.5

    @property
    def device(self):
        return next(self.parameters()).device

    def init_decoder(self, embedding, target_mels=None):
        device = self.device
        batch_size = embedding.size(0)
        seq_len = embedding.size(1)
        self.context_frame = torch.zeros([batch_size, self.embedding_dim]).to(device)
        self.c_pre = torch.zeros([batch_size, self.pre_rnn.hidden_size]).to(device)
        self.h_pre = torch.zeros([batch_size, self.pre_rnn.hidden_size]).to(device)
        self.c_post = torch.zeros([batch_size, self.post_rnn.hidden_size]).to(device)
        self.h_post = torch.zeros([batch_size, self.post_rnn.hidden_size]).to(device)
        self.alignment_weights = torch.zeros([batch_size, seq_len]).to(device)
        self.alignment_sum = torch.zeros([batch_size, seq_len]).to(device)
        if not target_mels is None:
            self.max_decoder_steps = target_mels.size(1)
        mel_go_frame = torch.zeros([batch_size, self.mel_dim]).to(device)
        return mel_go_frame

    def forward_step(self, embedding, mel_frame_prev):
        pre_net_out = self.pre_net(mel_frame_prev) # [batch, num_mel] --> [batch, pre_out_size]
        pre_rnn_input = torch.cat([pre_net_out, self.context_frame], dim=-1) # [batch, pre_rnn_input_size]
        self.c_pre, self.h_pre = self.pre_rnn(pre_rnn_input, (self.c_pre, self.h_pre))
        self.c_pre = self.dropout(self.c_pre)
        alignment_cat = torch.stack([self.alignment_sum, self.alignment_weights], dim=1) # [batch, 2, seq_len]
        self.context_frame, self.alignment_weights = self.loc_sens_att(self.h_pre, embedding, alignment_cat)
        self.alignment_sum += self.alignment_weights
        post_rnn_input = torch.cat([self.h_pre, self.context_frame], dim=-1)
        self.c_post, self.h_post = self.post_rnn(post_rnn_input, (self.c_post, self.h_post))
        self.c_post = self.dropout(self.c_post)
        output = torch.cat([self.h_post, self.context_frame], dim=-1)
        mel_output = torch.sigmoid(self.mel_proj(output))
        stop_value = torch.sigmoid(self.stop_proj(output))
        return mel_output, stop_value

    def forward(self, embedding, target_mels=None):
        mel_frames = list()
        stop_values = list()
        mel_frame_prev = self.init_decoder(embedding, target_mels)
        for idx in range(self.max_decoder_steps):
            if not target_mels is None and idx > 0: # Force teaching method. Only for train.
                mel_frame_prev = target_mels[:, idx-1]
            mel_output, stop_value = self.forward_step(embedding, mel_frame_prev)
            mel_frames.append(mel_output)
            stop_values.append(stop_value)
            mel_frame_prev = mel_output
        mel_frames = torch.stack(mel_frames, dim=1)
        stop_values = torch.stack(stop_values, dim=1)
        return mel_frames, stop_values

    def inference(self, embedding, target_mels=None):
        mel_frames = list()
        mel_frame_prev = self.init_decoder(embedding, target_mels=None)
        for i in range(self.max_decoder_steps):
            mel_output, stop_value = self.forward_step(embedding, mel_frame=mel_frame_prev)
            self.mel_frame = mel_output
            mel_frames.append(mel_output)
            if stop_value.item() > self.stop_value_threshold:
                break
        mel_frames = torch.stack(mel_frames, dim=1)
        return mel_frames, None


class PostNet(nn.Module):
    def __init__(self, mel_dim, post_net_dim, num_convs, kernel_size):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ConvBlock(in_channels=mel_dim,
                                          out_channels=post_net_dim,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          dropout=0.5,
                                          activation="tanh"))
        for _ in range(num_convs):
            self.conv_blocks.append(ConvBlock(in_channels=post_net_dim,
                                              out_channels=post_net_dim,
                                              kernel_size=kernel_size,
                                              stride=1,
                                              dropout=0.5,
                                              activation="tanh"))
        self.conv_blocks.append(ConvBlock(in_channels=post_net_dim,
                                          out_channels=mel_dim,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          dropout=0.5,
                                          activation="tanh"))

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = x.transpose(1, 2)
        return x

