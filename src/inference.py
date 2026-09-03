import torch

from .masks import (
    create_padding_mask,
    create_target_mask,
)


def prepare_source_sentence(
    sentence,
    tokenizer,
    max_len,
    bos_id,
    eos_id,
    pad_id,
    device,
):
    token_ids = tokenizer.encode(
        sentence
    ).ids

    token_ids = token_ids[
        :max_len - 2
    ]

    token_ids = (
        [bos_id]
        + token_ids
        + [eos_id]
    )

    padding_length = (
        max_len
        - len(token_ids)
    )

    token_ids = (
        token_ids
        + [pad_id] * padding_length
    )

    return torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )


def greedy_translate(
    model,
    sentence,
    src_tokenizer,
    tgt_tokenizer,
    max_len,
    src_bos_id,
    src_eos_id,
    src_pad_id,
    tgt_bos_id,
    tgt_eos_id,
    tgt_pad_id,
    device,
):
    """
    Greedy autoregressive translation.

    The encoder is computed once.
    The decoder then predicts one token
    at a time until EOS or max_len.
    """

    model.eval()

    src = prepare_source_sentence(
        sentence=sentence,
        tokenizer=src_tokenizer,
        max_len=max_len,
        bos_id=src_bos_id,
        eos_id=src_eos_id,
        pad_id=src_pad_id,
        device=device,
    )

    src_mask = create_padding_mask(
        src,
        src_pad_id,
    )

    generated = torch.tensor(
        [[tgt_bos_id]],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        # Compute encoder only once
        encoder_output, _ = model.encoder(
            src,
            src_mask,
        )

        for _ in range(
            max_len - 1
        ):

            tgt_mask = create_target_mask(
                generated,
                tgt_pad_id,
            )

            (
                decoder_output,
                _,
                _,
            ) = model.decoder(
                generated,
                encoder_output,
                tgt_mask,
                src_mask,
            )

            logits = model.output_projection(
                decoder_output
            )

            next_token_logits = logits[
                :, -1, :
            ]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
            ).unsqueeze(1)

            generated = torch.cat(
                [
                    generated,
                    next_token,
                ],
                dim=1,
            )

            if (
                next_token.item()
                == tgt_eos_id
            ):
                break

    generated_ids = (
        generated[0]
        .detach()
        .cpu()
        .tolist()
    )

    # Remove BOS
    generated_ids = (
        generated_ids[1:]
    )

    if tgt_eos_id in generated_ids:
        eos_position = (
            generated_ids.index(
                tgt_eos_id
            )
        )

        generated_ids = (
            generated_ids[
                :eos_position
            ]
        )

    translated_text = (
        tgt_tokenizer.decode(
            generated_ids
        )
    )

    return (
        translated_text,
        generated_ids,
    )