import time
import torch
from collections import defaultdict
from transformer_lens import HookedTransformer
from nforge.data.ioi import make_pairs

# ---------------- Settings ----------------
N_PAIRS = 30          # pairs per set (raise to 60+ once it runs fast)
N_RANDOM_DRAWS = 10   # random baselines per k (raise to 30 later)
KS = [1, 2, 3, 4, 5, 6, 8, 10]

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
DEVICE = model.cfg.device
N_LAYERS, N_HEADS = model.cfg.n_layers, model.cfg.n_heads
print(f"Running on: {DEVICE}", flush=True)

only_z = lambda n: n.endswith("attn.hook_z")


def build_batches(pairs):
    """Group pairs by token length so each group runs as one batch."""
    groups = defaultdict(list)
    for clean, corr, c, w in pairs:
        ct, xt = model.to_tokens(clean), model.to_tokens(corr)
        groups[ct.shape[1]].append((ct[0], xt[0], c, w))
    batches = []
    for items in groups.values():
        b = dict(
            clean=torch.stack([i[0] for i in items]),
            corr=torch.stack([i[1] for i in items]),
            correct=torch.tensor([i[2] for i in items], device=DEVICE),
            wrong=torch.tensor([i[3] for i in items], device=DEVICE),
        )
        with torch.no_grad():
            _, b["clean_cache"] = model.run_with_cache(b["clean"], names_filter=only_z)
            _, b["corr_cache"] = model.run_with_cache(b["corr"], names_filter=only_z)
        batches.append(b)
    return batches


def ld(logits, b):
    idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[idx, -1, b["correct"]] - logits[idx, -1, b["wrong"]]


def run(b, tokens, heads, source_cache):
    """Overwrite the given heads' z with values from source_cache."""
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
        return ld(model.run_with_hooks(tokens, fwd_hooks=hooks), b)


def per_example(batches, tokens_key, heads, cache_key):
    return torch.cat([run(b, b[tokens_key], heads, b[cache_key]) for b in batches]).cpu()


# ---------------- Data: discovery set and a disjoint held-out set ----------------
disc_pairs = make_pairs(model, N_PAIRS, seed=0)
seen = {p[0] for p in disc_pairs}
held_pairs = [p for p in make_pairs(model, 4 * N_PAIRS, seed=1) if p[0] not in seen][:N_PAIRS]
disc, held = build_batches(disc_pairs), build_batches(held_pairs)
print(f"Discovery pairs: {len(disc_pairs)} | Held-out pairs: {len(held_pairs)}", flush=True)

# ---------------- Discovery: rank heads by patching on the DISCOVERY set ----------------
clean_ld = per_example(disc, "clean", [], "clean_cache").mean().item()
corr_ld = per_example(disc, "corr", [], "corr_cache").mean().item()
print(f"Discovery means: clean {clean_ld:.2f} | corrupted {corr_ld:.2f}", flush=True)

t0 = time.time()
scores = {}
for l in range(N_LAYERS):
    for h in range(N_HEADS):
        patched = per_example(disc, "corr", [(l, h)], "clean_cache").mean().item()
        scores[(l, h)] = (patched - corr_ld) / (clean_ld - corr_ld)
    print(f"  patching: layer {l + 1}/{N_LAYERS} done, {time.time() - t0:.0f}s elapsed", flush=True)

ranked = sorted(scores, key=scores.get, reverse=True)
circuit = ranked[:10]
negatives = sorted(scores, key=scores.get)[:3]
print("\nTop 10 positive heads (discovery set):")
for hd in circuit:
    print(f"  {hd}: {scores[hd]:+.3f}")
print("Most negative heads:", [(hd, round(scores[hd], 3)) for hd in negatives], flush=True)

# ---------------- Evaluation on HELD-OUT prompts ----------------
held_clean = per_example(held, "clean", [], "clean_cache")
print(f"\nHeld-out clean logit diff: {held_clean.mean():.2f}", flush=True)


def ratio_ci(ablated, n_boot=1000):
    """Mean ablated / mean clean, with a bootstrap 95% CI over prompts."""
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(ablated), (n_boot, len(ablated)), generator=g)
    r = ablated[idx].mean(1) / held_clean[idx].mean(1)
    return (ablated.mean() / held_clean.mean()).item(), r.quantile(0.025).item(), r.quantile(0.975).item()


# Random baselines draw only from heads not in the circuit or the negatives
excluded = set(circuit) | set(negatives)
pool = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS) if (l, h) not in excluded]

print(f"\n{'k':>2} {'circuit (95% CI)':>26} {f'random mean ± std ({N_RANDOM_DRAWS} draws)':>34}")
for k in KS:
    r, lo, hi = ratio_ci(per_example(held, "clean", circuit[:k], "corr_cache"))
    rand = []
    for seed in range(N_RANDOM_DRAWS):
        g = torch.Generator().manual_seed(seed)
        pick = [pool[i] for i in torch.randperm(len(pool), generator=g)[:k]]
        rand.append((per_example(held, "clean", pick, "corr_cache").mean() / held_clean.mean()).item())
    rt = torch.tensor(rand)
    print(f"{k:>2} {r:>10.0%} [{lo:>4.0%}, {hi:>4.0%}] {rt.mean():>20.0%} ± {rt.std():.0%}", flush=True)

print("\nDone.")
