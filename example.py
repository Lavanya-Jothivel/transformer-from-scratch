import torch

from src.transformer import Transformer
from src.masks import create_padding_mask, create_target_mask


def main():
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=120,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,
    )

    src = torch.tensor([
        [1, 10, 20, 30, 2, 0]
    ])

    tgt = torch.tensor([
        [1, 40, 50, 60]
    ])

    src_mask = create_padding_mask(
        src,
        pad_id=0,
    )

    tgt_mask = create_target_mask(
        tgt,
        pad_id=0,
    )

    logits, _, _, _ = model(
        src,
        tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
    )

    print("Transformer ran successfully!")
    print("Output shape:", logits.shape)


if __name__ == "__main__":
    main()