import copy

import torch.nn as nn

from .attention import MultiHeadAttention
from .embeddings import TokenEmbedding, PositionalEncoding
from .encoder import FeedForward


class DecoderLayer(nn.Module):

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

        self.cross_attention = MultiHeadAttention(
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

        self.norm3 = nn.LayerNorm(
            d_model
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

        self.dropout3 = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        src_mask=None
    ):
        self_attention_output, self_attention_weights = (
            self.self_attention(
                x,
                x,
                x,
                tgt_mask
            )
        )

        x = self.norm1(
            x
            + self.dropout1(
                self_attention_output
            )
        )

        cross_attention_output, cross_attention_weights = (
            self.cross_attention(
                x,
                encoder_output,
                encoder_output,
                src_mask
            )
        )

        x = self.norm2(
            x
            + self.dropout2(
                cross_attention_output
            )
        )

        ffn_output = self.feed_forward(
            x
        )

        x = self.norm3(
            x
            + self.dropout3(
                ffn_output
            )
        )

        return (
            x,
            self_attention_weights,
            cross_attention_weights
        )


class Decoder(nn.Module):

    def __init__(
        self,
        decoder_layer,
        num_layers
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    decoder_layer
                )
                for _ in range(
                    num_layers
                )
            ]
        )

    def forward(
        self,
        x,
        encoder_output,
        tgt_mask=None,
        src_mask=None
    ):
        self_attention_maps = []
        cross_attention_maps = []

        for layer in self.layers:

            (
                x,
                self_attention_weights,
                cross_attention_weights
            ) = layer(
                x,
                encoder_output,
                tgt_mask,
                src_mask
            )

            self_attention_maps.append(
                self_attention_weights
            )

            cross_attention_maps.append(
                cross_attention_weights
            )

        return (
            x,
            self_attention_maps,
            cross_attention_maps
        )


class DecoderModel(nn.Module):

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

        base_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.decoder = Decoder(
            base_layer,
            num_layers
        )

    def forward(
        self,
        tgt,
        encoder_output,
        tgt_mask=None,
        src_mask=None
    ):
        x = self.embedding(
            tgt
        )

        x = self.positional_encoding(
            x
        )

        x = self.embedding_dropout(
            x
        )

        return self.decoder(
            x,
            encoder_output,
            tgt_mask,
            src_mask
        )