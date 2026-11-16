import torch
from transformer_lens import HookedTransformer
from nforge.data.ioi import make_pairs
from nforge.core import build_batches, per_example, attribution_scores

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
N_LAYERS, N_HEADS = model.cfg.n_layers, model.cfg.n_heads
ALL_HEADS = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
KS = [5, 10, 15, 20, 30, 40, 60, 80, 100, 144]
N_RANDOM_ORDERS = 5

# ---- Rank heads on the DISCOVERY set (fast attribution patching) ----
disc_pairs = make_pairs(model, 30, seed=0)
disc = build_batches(model, disc_pairs)
c_ld = per_example(model, disc, "clean", [], "clean_cache").mean().item()
x_ld = per_example(model, disc, "corr", [], "corr_cache").mean().item()
scores = attribution_scores(model, disc, c_ld, x_ld)
order = sorted(ALL_HEADS, key=lambda hd: scores[hd].item(), reverse=True)

# ---- Evaluate on HELD-OUT prompts ----
seen = {p[0] for p in disc_pairs}
held_pairs = [p for p in make_pairs(model, 240, seed=1) if p[0] not in seen][:60]
held = build_batches(model, held_pairs)
clean = per_example(model, held, "clean", [], "clean_cache").mean().item()
floor = per_example(model, held, "clean", ALL_HEADS, "corr_cache").mean().item()
print(f"Held-out: full model {clean:.2f} | all heads ablated {floor:.2f} ({floor / clean:.0%})\n", flush=True)


def kept_score(keep):
    """Mean logit diff on clean prompts with every head outside `keep` ablated."""
    keep = set(keep)
    ablate = [hd for hd in ALL_HEADS if hd not in keep]
    return per_example(model, held, "clean", ablate, "corr_cache").mean().item()


def normalised(x):
    return (x - floor) / (clean - floor)   # 0 = nothing kept, 1 = full model


print(f"{'k':>4} {'ranked: % of clean':>20} {'normalised':>11} {'random order: normalised':>26}")
for k in KS:
    ranked_val = kept_score(order[:k])
    rand = []
    for seed in range(N_RANDOM_ORDERS):
        g = torch.Generator().manual_seed(seed)
        perm = [ALL_HEADS[i] for i in torch.randperm(len(ALL_HEADS), generator=g)]
        rand.append(normalised(kept_score(perm[:k])))
    rt = torch.tensor(rand)
    print(f"{k:>4} {ranked_val / clean:>19.0%} {normalised(ranked_val):>11.2f} "
          f"{rt.mean():>18.2f} ± {rt.std():.2f}", flush=True)

print("\nDone.")
