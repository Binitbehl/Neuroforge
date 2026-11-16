import torch
from collections import defaultdict
from nforge.activations.batches import logit_diff


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