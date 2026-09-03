import copy

import torch.nn as nn

from .attention import MultiHeadAttention
from .embeddings import TokenEmbedding, PositionalEncoding


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(
            self.activation(
                self.linear1(x)
            )
        )


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        mask=None
    ):
        attention_output, attention_weights = (
            self.self_attention(
                x,
                x,
                x,
                mask
            )
        )

        x = self.norm1(
            x
            + self.dropout1(
                attention_output
            )
        )

        ffn_output = self.feed_forward(
            x
        )

        x = self.norm2(
            x
            + self.dropout2(
                ffn_output
            )
        )

        return x, attention_weights


class Encoder(nn.Module):

    def __init__(
        self,
        encoder_layer,
        num_layers
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    encoder_layer
                )
                for _ in range(
                    num_layers
                )
            ]
        )

    def forward(
        self,
        x,
        mask=None
    ):
        attention_maps = []

        for layer in self.layers:
            x, attention_weights = (
                layer(
                    x,
                    mask
                )
            )

            attention_maps.append(
                attention_weights
            )

        return x, attention_maps


class EncoderModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = TokenEmbedding(
            vocab_size,
            d_model
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model,
                max_len
            )
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        base_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder = Encoder(
            base_layer,
            num_layers
        )

    def forward(
        self,
        src,
        mask=None
    ):
        x = self.embedding(
            src
        )

        x = self.positional_encoding(
            x
        )

        x = self.embedding_dropout(
            x
        )

        return self.encoder(
            x,
            mask
        )