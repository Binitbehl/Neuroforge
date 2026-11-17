import gc
import os
import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("HF_TOKEN", "placeholder")

MODEL = "pythia-70m"
STEPS = [1000, 2000, 4000, 16000, 143000]
SEQ_LEN, BATCH, SEED = 50, 32, 1
TARGET = slice(SEQ_LEN + 1, 2 * SEQ_LEN)
SOURCE = (2, 1)
READER_LAYERS = [3, 4, 5]


def make_batch(model, seed):
    g = torch.Generator().manual_seed(seed)
    bos = torch.full((BATCH, 1), model.tokenizer.bos_token_id)
    r = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    other = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    dev = model.cfg.device
    return torch.cat([bos, r, r], 1).to(dev), torch.cat([bos, other, r], 1).to(dev)


def metric(logits, clean):
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, clean[:, 1:, None]).squeeze(-1)
    return lp[:, TARGET].mean(1)


for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    model.set_use_split_qkv_input(True)
    n_h = model.cfg.n_heads
    clean, corrupt = make_batch(model, SEED)
    only_z = lambda n: n.endswith("attn.hook_z")
    with torch.no_grad():
        cl, clean_cache = model.run_with_cache(clean, names_filter=only_z)
        xl, corr_cache = model.run_with_cache(corrupt, names_filter=only_z)
    c, x = metric(cl, clean).mean().item(), metric(xl, clean).mean().item()

    sl, sh = SOURCE
    name = f"blocks.{sl}.attn.hook_z"
    delta = (corr_cache[name][:, :, sh, :] - clean_cache[name][:, :, sh, :]) @ model.W_O[sl, sh]

    def key_patch(readers):
        by_layer = {}
        for l, h in readers:
            by_layer.setdefault(l, []).append(h)
        hooks = []
        for l, hs in by_layer.items():
            def hook(act, hook, hs=hs):
                act = act.clone()
                for h in hs:
                    act[:, :, h, :] += delta
                return act
            hooks.append((f"blocks.{l}.hook_k_input", hook))
        return hooks

    def score(readers):
        with torch.no_grad():
            v = metric(model.run_with_hooks(clean, fwd_hooks=key_patch(readers)), clean).mean().item()
        return (v - x) / (c - x)

    singles = {(l, h): score([(l, h)]) for l in READER_LAYERS for h in range(n_h)}
    worst = sorted(singles, key=singles.get)

    print(f"\n=== step {step} === key-path patching of {SOURCE}")
    for l in READER_LAYERS:
        print(f"  all heads of layer {l}: {score([(l, h) for h in range(n_h)]):.2f}")
    print(f"  top-4 readers jointly {worst[:4]}: {score(worst[:4]):.2f}")
    print("  single readers, 8 lowest 'kept':", [(hd, round(singles[hd], 2)) for hd in worst[:8]], flush=True)
    del model
    gc.collect()

print("\nDone.")