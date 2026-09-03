import torch

from src.transformer import Transformer
from src.attention import scaled_dot_product_attention
from src.masks import (
    create_padding_mask,
    create_target_mask,
)


def test_attention():
    batch_size = 2
    num_heads = 4
    seq_len = 5
    d_k = 8

    Q = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        d_k
    )

    K = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        d_k
    )

    V = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        d_k
    )

    output, weights = (
        scaled_dot_product_attention(
            Q,
            K,
            V
        )
    )

    assert output.shape == (
        batch_size,
        num_heads,
        seq_len,
        d_k
    )

    assert weights.shape == (
        batch_size,
        num_heads,
        seq_len,
        seq_len
    )

    print("Attention test passed!")


def test_masks():
    tokens = torch.tensor([
        [1, 5, 8, 2, 0, 0]
    ])

    padding_mask = create_padding_mask(
        tokens,
        pad_id=0
    )

    target_mask = create_target_mask(
        tokens,
        pad_id=0
    )

    assert padding_mask.shape == (
        1, 1, 1, 6
    )

    assert target_mask.shape == (
        1, 1, 6, 6
    )

    # Future position must be blocked.
    assert target_mask[
        0, 0, 0, 1
    ].item() is False

    print("Mask tests passed!")


def test_complete_transformer():
    torch.manual_seed(42)

    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=120,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=50,
        dropout=0.0
    )

    src = torch.tensor([
        [1, 10, 20, 30, 2, 0],
        [1, 15, 25, 2, 0, 0]
    ])

    tgt = torch.tensor([
        [1, 40, 50, 60, 0],
        [1, 45, 55, 0, 0]
    ])

    src_mask = create_padding_mask(
        src,
        pad_id=0
    )

    tgt_mask = create_target_mask(
        tgt,
        pad_id=0
    )

    (
        logits,
        encoder_maps,
        decoder_self_maps,
        decoder_cross_maps
    ) = model(
        src,
        tgt,
        src_mask=src_mask,
        tgt_mask=tgt_mask
    )

    assert logits.shape == (
        2,
        5,
        120
    )

    assert len(encoder_maps) == 2
    assert len(decoder_self_maps) == 2
    assert len(decoder_cross_maps) == 2

    assert not torch.isnan(
        logits
    ).any()

    print(
        "Complete Transformer test passed!"
    )


def test_gradients():
    model = Transformer(
        src_vocab_size=100,
        tgt_vocab_size=120,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0
    )

    src = torch.randint(
        1, 100, (2, 6)
    )

    tgt = torch.randint(
        1, 120, (2, 5)
    )

    logits, _, _, _ = model(
        src,
        tgt
    )

    loss = logits.mean()

    loss.backward()

    gradients_exist = any(
        p.grad is not None
        for p in model.parameters()
    )

    assert gradients_exist

    print("Gradient test passed!")