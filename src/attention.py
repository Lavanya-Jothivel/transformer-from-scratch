import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query,
    key,
    value,
    mask=None
):
    d_k = query.size(-1)

    scores = torch.matmul(
        query,
        key.transpose(-2, -1)
    )

    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

    attention_weights = F.softmax(
        scores,
        dim=-1
    )

    attention_output = torch.matmul(
        attention_weights,
        value
    )

    return attention_output, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(
            d_model,
            d_model
        )

        self.W_k = nn.Linear(
            d_model,
            d_model
        )

        self.W_v = nn.Linear(
            d_model,
            d_model
        )

        self.W_o = nn.Linear(
            d_model,
            d_model
        )

    def split_heads(self, x):
        batch_size = x.size(0)

        x = x.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size = x.size(0)

        x = x.transpose(
            1,
            2
        ).contiguous()

        return x.view(
            batch_size,
            -1,
            self.d_model
        )

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output, attention_weights = (
            scaled_dot_product_attention(
                Q,
                K,
                V,
                mask
            )
        )

        attention_output = self.combine_heads(
            attention_output
        )

        output = self.W_o(
            attention_output
        )

        return output, attention_weights