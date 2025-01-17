import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        positions = torch.arange(0, self.max_len).unsqueeze(1)
        denominator = 10000**(torch.arange(0, dim, 2) / self.dim)
        self.pe = torch.zeros([max_len, 1, dim])
        self.pe[:, 0, 0::2] = torch.sin(positions / denominator)
        self.pe[:, 0, 1::2] = torch.cos(positions / denominator)

    def forward(self, x):
        out = x + self.pe[:x.size(0)].to(x.device)
        return out


class MultiHeadScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        """ input shape  --> [batch, num_heads, seq_len, head_dim]
            output shape --> [batch, num_heads, seq_len, head_dim]
        """
        # inputs shape --> [batch, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = query.size()
        q_to_k_sim_scores = query.matmul(key.transpose(-2, -1))
        q_to_k_sim_scores = q_to_k_sim_scores / math.sqrt(head_dim)
        attention = F.softmax(q_to_k_sim_scores, dim=-1)
        output = attention.matmul(value)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, num_heads):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.linear_q = nn.Linear(in_features, in_features)
        self.linear_k = nn.Linear(in_features, in_features)
        self.linear_v = nn.Linear(in_features, in_features)
        self.linear_o = nn.Linear(in_features, in_features)
        self.scale_dot_prodict = MultiHeadScaledDotProduct()

    def _slice_before_heads(self, query, key, value):
        """ inputs shape  --> [batch, seq_len, feature_dim]
            outputs shape --> [batch, num_heads, seq_len, head_dim]
        """
        batch, seq_len, feature_dim = query.size()
        head_dim = feature_dim // self.num_heads
        query = query.reshape(batch, seq_len, self.num_heads, head_dim)
        key = key.reshape(batch, seq_len, self.num_heads, head_dim)
        value = value.reshape(batch, seq_len, self.num_heads, head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        return query, key, value

    def _stack_after_heads(self, att_out):
        """ input shape  --> [batch, num_heads, seq_len, head_dim]
            output shape --> [batch, seq_len, num_heads * head_dim]
        """
        batch, num_heads, seq_len, head_dim = att_out.size()
        att_out = att_out.permute(0, 2, 1, 3)
        att_out = att_out.reshape(batch, seq_len, num_heads*head_dim)
        return att_out

    def forward(self, x):
        query, key, value = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        query, key, value = self._slice_before_heads(query, key, value)
        output = self.scale_dot_prodict(query, key, value)
        output = self._stack_after_heads(output)
        output = self.linear_o(output)
        return output


class FeedForward(nn.Module):
    def __init__(self,
        in_features=256,
        hid_features=2048,
        dropout=0.2,
        conv=False
    ):
        super().__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout)
        self.conv = conv
        if self.conv:
            self.layer1 = nn.Conv1d(in_channels=in_features,
                                    out_channels=hid_features,
                                    kernel_size=9,
                                    stride=1,
                                    padding=4)
            self.layer2 = nn.Conv1d(in_channels=hid_features,
                                    out_channels=in_features,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        else:
            self.layer1 = nn.Linear(in_features, hid_features)
            self.layer2 = nn.Linear(hid_features, in_features)

    def forward(self, x):
        if self.conv:
            x = x.transpose(1, 2)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        if self.conv:
            x = x.transpose(1, 2)
        return x


class LocationSensetiveAttention(nn.Module):
    def __init__(self,
                 rec_hidden_dim=512,
                 value_dim=256,
                 attention_dim=128,
                 location_conv_filter_size=32,
                 location_conv_kernel_size=31):
        super().__init__()
        self.query_proj = nn.Linear(rec_hidden_dim, attention_dim, bias=False)
        self.value_proj = nn.Linear(value_dim, attention_dim, bias=False)
        self.aligin_proj = nn.Linear(attention_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.rand(attention_dim).uniform_(-0.1, 0.1))
        self.location_conv = nn.Conv1d(in_channels=2,
                                       out_channels=location_conv_filter_size,
                                       kernel_size=location_conv_kernel_size,
                                       padding=int(location_conv_kernel_size/2),
                                       bias=False)
        self.location_proj = nn.Linear(location_conv_filter_size,
                                       attention_dim,
                                       bias=False)

    def forward(self, query, value, last_alignment_weghts, mask=None):
        last_alignment_weghts = self.location_conv(last_alignment_weghts).transpose(1, 2)
        last_alignment_weghts = self.location_proj(last_alignment_weghts)
        alignment_scores = self.aligin_proj(
            torch.tanh(
                self.query_proj(query).unsqueeze(1)
                + self.value_proj(value)
                + last_alignment_weghts
                + self.bias)
        ).squeeze(-1)
        if mask is not None:
            alignment_scores.masked_fill_(mask, -float("inf"))
        alignment_weights = F.softmax(alignment_scores, dim=1)
        context = torch.bmm(alignment_weights.unsqueeze(1), value).squeeze(1)
        return context, alignment_weights


