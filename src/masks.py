import torch


def create_padding_mask(
    tokens,
    pad_id
):
    mask = (
        tokens != pad_id
    )

    return (
        mask
        .unsqueeze(1)
        .unsqueeze(2)
    )


def create_causal_mask(
    sequence_length,
    device
):
    return torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            device=device,
            dtype=torch.bool
        )
    )


def create_target_mask(
    target_tokens,
    pad_id
):
    _, sequence_length = (
        target_tokens.shape
    )

    padding_mask = (
        target_tokens != pad_id
    )

    padding_mask = (
        padding_mask
        .unsqueeze(1)
        .unsqueeze(2)
    )

    causal_mask = create_causal_mask(
        sequence_length,
        target_tokens.device
    )

    causal_mask = (
        causal_mask
        .unsqueeze(0)
        .unsqueeze(1)
    )

    return (
        padding_mask
        & causal_mask
    )