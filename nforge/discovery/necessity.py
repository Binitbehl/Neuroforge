import torch
from nforge.patching.ablation import per_example


def necessity_scan(model, batches):
    """Resample-ablate each head alone on the clean prompts.
    Returns a [layers, heads] grid: 1 = no effect, below 1 = removal hurts, above 1 = removal helps."""
    c = per_example(model, batches, "clean", [], "clean_cache").mean().item()
    x = per_example(model, batches, "corr", [], "corr_cache").mean().item()
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    kept = torch.zeros(n_l, n_h)
    for l in range(n_l):
        for h in range(n_h):
            v = per_example(model, batches, "clean", [(l, h)], "corr_cache").mean().item()
            kept[l, h] = (v - x) / (c - x)
    return kept