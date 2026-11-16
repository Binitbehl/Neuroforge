import torch
from nforge.activations.batches import only_z, logit_diff


def attribution_scores(model, batches, clean_ld, corr_ld):
    """First-order estimate of each head's patching effect (same scale as exact patching)."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    attr = torch.zeros(n_layers, n_heads)
    n = 0
    for b in batches:
        grads = {}

        def save_grad(g, hook):
            grads[hook.name] = g.detach()

        with model.hooks(bwd_hooks=[(only_z, save_grad)]):
            logit_diff(model(b["corr"]), b).sum().backward()
        for l in range(n_layers):
            name = f"blocks.{l}.attn.hook_z"
            delta = b["clean_cache"][name] - b["corr_cache"][name]
            attr[l] += (delta * grads[name]).sum(dim=(1, 3)).sum(0).cpu()
        n += b["clean"].shape[0]
    return attr / n / (clean_ld - corr_ld)