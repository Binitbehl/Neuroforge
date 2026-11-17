import gc
import os
import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("HF_TOKEN", "placeholder")   # workaround for the empty-token download error

MODEL = "pythia-70m"
STEPS = [16000, 143000]
SEQ_LEN, BATCH = 50, 32
TARGET = slice(SEQ_LEN + 1, 2 * SEQ_LEN)      # positions predicting the repeated tokens
N_RANDOM = 20


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


def run_overwrite(model, tokens, clean, heads, cache):
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    hooks = []
    for l, hs in by_layer.items():
        name = f"blocks.{l}.attn.hook_z"

        def hook(z, hook, hs=hs, name=name):
            z[:, :, hs, :] = cache[name][:, :, hs, :]
            return z

        hooks.append((name, hook))
    with torch.no_grad():
        return metric(model.run_with_hooks(tokens, fwd_hooks=hooks), clean)


def prepare(model, seed):
    clean, corrupt = make_batch(model, seed)
    only_z = lambda n: n.endswith("attn.hook_z")
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean, names_filter=only_z)
        corr_logits, corr_cache = model.run_with_cache(corrupt, names_filter=only_z)
    c = metric(clean_logits, clean).mean().item()
    x = metric(corr_logits, clean).mean().item()
    return dict(clean=clean, corrupt=corrupt, clean_cache=clean_cache, corr_cache=corr_cache, c=c, x=x)


def patch_scores(model, d):
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    s = torch.zeros(n_l, n_h)
    for l in range(n_l):
        for h in range(n_h):
            p = run_overwrite(model, d["corrupt"], d["clean"], [(l, h)], d["clean_cache"]).mean().item()
            s[l, h] = (p - d["x"]) / (d["c"] - d["x"])
    return s


def induction_attention(model, clean):
    """Mean attention from each repeated token to the token AFTER its earlier occurrence."""
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    with torch.no_grad():
        _, cache = model.run_with_cache(clean, names_filter=lambda n: n.endswith("attn.hook_pattern"))
    out = torch.zeros(n_l, n_h)
    for l in range(n_l):
        pat = cache[f"blocks.{l}.attn.hook_pattern"]                          # [batch, head, q, k]
        diag = torch.diagonal(pat, offset=-(SEQ_LEN - 1), dim1=-2, dim2=-1)   # q - k = SEQ_LEN - 1
        out[l] = diag[..., 2:].mean(dim=(0, 2)).cpu()                         # keep queries in the second copy
    return out


def top(scores, k):
    n_h = scores.shape[1]
    return [(i.item() // n_h, i.item() % n_h) for i in scores.flatten().argsort(descending=True)[:k]]


def rank_corr(a, b):
    ra, rb = a.flatten().argsort().argsort().float(), b.flatten().argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


print(f"Settings: {SEQ_LEN}-token random sequences, batch {BATCH}, ranking on seed 0, necessity on seed 1\n")
for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    A, B = prepare(model, 0), prepare(model, 1)

    sA, sB = patch_scores(model, A), patch_scores(model, B)
    ind = induction_attention(model, A["clean"])
    top3 = top(sA, 3)

    def kept(heads):
        v = run_overwrite(model, B["clean"], B["clean"], heads, B["corr_cache"]).mean().item()
        return (v - B["x"]) / (B["c"] - B["x"])        # 1 = behavior fully kept, 0 = as bad as corrupted

    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    pool = [(l, h) for l in range(n_l) for h in range(n_h) if (l, h) not in top3]
    rand = []
    for s in range(N_RANDOM):
        g = torch.Generator().manual_seed(100 + s)
        rand.append(kept([pool[i] for i in torch.randperm(len(pool), generator=g)[:3]]))
    rt = torch.tensor(rand)

    print(f"=== step {step} (gap {A['c'] - A['x']:.2f}) ===")
    print("  top-5 by patching (seed 0):   ", [(hd, round(sA[hd].item(), 2)) for hd in top(sA, 5)])
    print("  top-5 by induction attention: ", [(hd, round(ind[hd].item(), 2)) for hd in top(ind, 5)])
    print(f"  seed 0 vs seed 1 rank corr of patching scores: {rank_corr(sA, sB):.2f}")
    print(f"  top-3 patching heads {top3} ablated on held-out sequences: {kept(top3):.2f} of behavior kept")
    print(f"  random 3-head sets: {rt.mean():.2f} ± {rt.std():.2f} (min {rt.min():.2f})", flush=True)
    del model
    gc.collect()

print("\nDone.")