import torch
from nforge.activations.batches import cache_batch


def make_induction_batch(model, n=32, seq_len=50, seed=0):
    """Random tokens repeated twice. The corrupted run replaces the first copy, so there is nothing to copy."""
    g = torch.Generator().manual_seed(seed)
    dev = model.cfg.device
    bos = torch.full((n, 1), model.tokenizer.bos_token_id)
    r = torch.randint(1000, 30000, (n, seq_len), generator=g)
    other = torch.randint(1000, 30000, (n, seq_len), generator=g)
    clean = torch.cat([bos, r, r], 1).to(dev)
    corr = torch.cat([bos, other, r], 1).to(dev)
    target = slice(seq_len + 1, 2 * seq_len)

    def metric(logits):
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, clean[:, 1:, None]).squeeze(-1)
        return lp[:, target].mean(1)

    return cache_batch(model, dict(clean=clean, corr=corr, metric=metric))