import torch

from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """
    Dataset wrapper for English-to-German translation.

    Produces:
    - encoder_input
    - decoder_input
    - labels
    - source text
    - target text
    """

    def __init__(
        self,
        dataset,
        src_tokenizer,
        tgt_tokenizer,
        max_len,
        src_bos_id,
        src_eos_id,
        src_pad_id,
        tgt_bos_id,
        tgt_eos_id,
        tgt_pad_id,
        src_language="en",
        tgt_language="de",
    ):
        self.dataset = dataset

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.max_len = max_len

        self.src_bos_id = src_bos_id
        self.src_eos_id = src_eos_id
        self.src_pad_id = src_pad_id

        self.tgt_bos_id = tgt_bos_id
        self.tgt_eos_id = tgt_eos_id
        self.tgt_pad_id = tgt_pad_id

        self.src_language = src_language
        self.tgt_language = tgt_language

    def __len__(self):
        return len(
            self.dataset
        )

    def __getitem__(
        self,
        idx
    ):
        item = self.dataset[
            idx
        ]

        src_text = item[
            "translation"
        ][
            self.src_language
        ]

        tgt_text = item[
            "translation"
        ][
            self.tgt_language
        ]

        src_ids = (
            self.src_tokenizer
            .encode(
                src_text
            )
            .ids
        )

        tgt_ids = (
            self.tgt_tokenizer
            .encode(
                tgt_text
            )
            .ids
        )

        # Reserve space for:
        # <bos> source <eos>
        src_ids = src_ids[
            : self.max_len - 2
        ]

        # Reserve space for target EOS
        tgt_ids = tgt_ids[
            : self.max_len - 1
        ]

        # Encoder input:
        # <bos> tokens <eos> <pad>...
        encoder_input = (
            [self.src_bos_id]
            + src_ids
            + [self.src_eos_id]
        )

        encoder_padding = (
            self.max_len
            - len(
                encoder_input
            )
        )

        encoder_input = (
            encoder_input
            + [self.src_pad_id]
            * encoder_padding
        )

        # Decoder input:
        # <bos> token1 token2 ...
        decoder_input = (
            [self.tgt_bos_id]
            + tgt_ids
        )

        decoder_padding = (
            self.max_len
            - len(
                decoder_input
            )
        )

        decoder_input = (
            decoder_input
            + [self.tgt_pad_id]
            * decoder_padding
        )

        # Labels:
        # token1 token2 ... <eos>
        labels = (
            tgt_ids
            + [self.tgt_eos_id]
        )

        label_padding = (
            self.max_len
            - len(
                labels
            )
        )

        labels = (
            labels
            + [self.tgt_pad_id]
            * label_padding
        )

        return {
            "encoder_input":
                torch.tensor(
                    encoder_input,
                    dtype=torch.long
                ),

            "decoder_input":
                torch.tensor(
                    decoder_input,
                    dtype=torch.long
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long
                ),

            "src_text":
                src_text,

            "tgt_text":
                tgt_text,
        }