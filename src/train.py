import torch
from tqdm import tqdm


def transformer_learning_rate(
    step,
    d_model,
    warmup_steps
):
    """
    Learning-rate schedule from
    Attention Is All You Need.
    """

    step = max(step, 1)

    return (
        d_model ** -0.5
        * min(
            step ** -0.5,
            step * warmup_steps ** -1.5
        )
    )


def update_learning_rate(
    optimizer,
    step,
    d_model,
    warmup_steps
):
    lr = transformer_learning_rate(
        step,
        d_model,
        warmup_steps
    )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
    device,
    src_pad_id,
    tgt_pad_id,
    create_padding_mask,
    create_target_mask,
    d_model,
    warmup_steps,
    global_step=0
):
    model.train()

    total_loss = 0.0

    progress = tqdm(
        data_loader,
        desc="Training"
    )

    for batch in progress:

        src = batch[
            "encoder_input"
        ].to(device)

        tgt = batch[
            "decoder_input"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        src_mask = create_padding_mask(
            src,
            src_pad_id
        )

        tgt_mask = create_target_mask(
            tgt,
            tgt_pad_id
        )

        optimizer.zero_grad()

        logits, _, _, _ = model(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1)
            ),
            labels.reshape(-1)
        )

        global_step += 1

        lr = update_learning_rate(
            optimizer,
            global_step,
            d_model,
            warmup_steps
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{lr:.7f}"
        )

    average_loss = (
        total_loss
        / len(data_loader)
    )

    return average_loss, global_step