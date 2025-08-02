import math
import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)  # [batch_size, seq_length, input_dim]
        self.key = nn.Linear(input_dim, input_dim)  # [batch_size, seq_length, input_dim]
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = self.softmax(attn_scores)
        weighted_values = torch.matmul(attn_probs, v)
        return weighted_values


def cal_attention(query, key, value):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = scores.softmax(dim=-1)

    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
        ]

    def forward(self, x):
        nbatches = x.size(0)

        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (x, x, x))
        ]

        x = cal_attention(
            query, key, value
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        return self.linears[-1](x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(CrossAttention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query = nn.Linear(query_dim, query_dim, bias=False)
        self.key = nn.Linear(key_dim, query_dim, bias=False)
        self.value = nn.Linear(value_dim, value_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, kv):
        q = self.query(query)
        k = self.key(kv)
        v = self.value(kv)

        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_probs = self.softmax(attn_scores)
        weighted_values = torch.matmul(attn_probs, v)
        return weighted_values
