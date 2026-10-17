"""Full-vocabulary causal LM loss with bounded FP32 cross-entropy intermediates."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def chunked_causal_lm_loss(
    logits,
    labels,
    vocab_size,
    num_items_in_batch=None,
    ignore_index=-100,
    shift_labels=None,
    *,
    chunk_size=128,
    **kwargs,
):
    """Match HF ForCausalLMLoss, recomputing each token chunk during backward.

    The model still produces full logits in its original dtype. Only the loss
    upcast/softmax is chunked; every vocabulary entry and supervised token is
    retained. Checkpointing is essential: a loop alone would retain all FP32
    softmax intermediates until backward. No model parameters or distributed
    collectives are accessed inside the checkpointed function.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if logits.shape[-1] != vocab_size:
        raise ValueError("logits vocabulary dimension does not match vocab_size")
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    targets = shift_labels.reshape(-1).to(logits.device)
    flat_logits = logits.reshape(-1, vocab_size)
    if flat_logits.shape[0] != targets.numel():
        raise ValueError("logits and shifted labels must have the same token count")

    def chunk_loss(scores, target):
        return F.cross_entropy(
            scores.float(), target, ignore_index=ignore_index, reduction="sum"
        )

    # One SplitBackward assembles the logits gradient once. Independent slices
    # would each allocate a full-sized zero-padded gradient in SliceBackward.
    losses = []
    for scores, target in zip(
        flat_logits.split(chunk_size, dim=0), targets.split(chunk_size, dim=0)
    ):
        if torch.is_grad_enabled() and scores.requires_grad:
            loss = checkpoint(
                chunk_loss, scores, target, use_reentrant=False, preserve_rng_state=False
            )
        else:
            loss = chunk_loss(scores, target)
        losses.append(loss)
    numerator = torch.stack(losses).sum()
    denominator = num_items_in_batch
    if denominator is None:
        denominator = targets.ne(ignore_index).sum()
    elif torch.is_tensor(denominator):
        denominator = denominator.to(numerator.device)
    # Normalize once across all chunks, including an uneven final chunk. This
    # also preserves HF's all-ignored mean (NaN) and externally normalized sum.
    return numerator / denominator
