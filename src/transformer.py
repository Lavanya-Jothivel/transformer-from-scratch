import torch.nn as nn

from .encoder import EncoderModel
from .decoder import DecoderModel


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()

        self.encoder = EncoderModel(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            max_len=max_len,
            dropout=dropout
        )

        self.decoder = DecoderModel(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            max_len=max_len,
            dropout=dropout
        )

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None
    ):
        (
            encoder_output,
            encoder_attention_maps
        ) = self.encoder(
            src,
            src_mask
        )

        (
            decoder_output,
            decoder_self_attention_maps,
            decoder_cross_attention_maps
        ) = self.decoder(
            tgt,
            encoder_output,
            tgt_mask,
            src_mask
        )

        logits = self.output_projection(
            decoder_output
        )

        return (
            logits,
            encoder_attention_maps,
            decoder_self_attention_maps,
            decoder_cross_attention_maps
        )