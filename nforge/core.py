import torch
from collections import defaultdict

only_z = lambda n: n.endswith("attn.hook_z")


def build_batches(model, pairs):
    """Group (clean, corrupted, correct_id, wrong_id) pairs by token length into batches."""
    dev = model.cfg.device
    groups = defaultdict(list)
    for clean, corr, c, w in pairs:
        ct, xt = model.to_tokens(clean), model.to_tokens(corr)
        groups[ct.shape[1]].append((ct[0], xt[0], c, w))
    batches = []
    for items in groups.values():
        b = dict(
            clean=torch.stack([i[0] for i in items]),
            corr=torch.stack([i[1] for i in items]),
            correct=torch.tensor([i[2] for i in items], device=dev),
            wrong=torch.tensor([i[3] for i in items], device=dev),
        )
        with torch.no_grad():
            _, b["clean_cache"] = model.run_with_cache(b["clean"], names_filter=only_z)
            _, b["corr_cache"] = model.run_with_cache(b["corr"], names_filter=only_z)
        batches.append(b)
    return batches


def logit_diff(logits, b):
    idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[idx, -1, b["correct"]] - logits[idx, -1, b["wrong"]]


def run_with_overwrite(model, b, tokens, heads, source_cache):
    """Run `tokens`, overwriting the given (layer, head) outputs with values from source_cache."""
    by_layer = defaultdict(list)
    for l, h in heads:
        by_layer[l].append(h)
    hooks = []
    for l, hs in by_layer.items():
        name = f"blocks.{l}.attn.hook_z"

        def hook(z, hook, hs=hs, name=name):
            z[:, :, hs, :] = source_cache[name][:, :, hs, :]
            return z

        hooks.append((name, hook))
    with torch.no_grad():
        return logit_diff(model.run_with_hooks(tokens, fwd_hooks=hooks), b)


def per_example(model, batches, tokens_key, heads, cache_key):
    return torch.cat(
        [run_with_overwrite(model, b, b[tokens_key], heads, b[cache_key]) for b in batches]
    ).cpu()
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