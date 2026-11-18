import json
import time
import torch
from nforge.models.loading import load_model
from nforge.data.ioi import make_pairs
from nforge.activations.batches import build_batches
from nforge.patching.ablation import per_example
from nforge.discovery.attribution import attribution_scores
from nforge.stability.compare import rank_corr

N_DISC, N_EVAL = 30, 30

model = load_model("gpt2")
n_l, n_h = model.cfg.n_layers, model.cfg.n_heads

# Rank heads by attribution patching on the discovery set (seed 0)
disc_pairs = make_pairs(model, N_DISC, seed=0)
disc = build_batches(model, disc_pairs)
c0 = per_example(model, disc, "clean", [], "clean_cache").mean().item()
x0 = per_example(model, disc, "corr", [], "corr_cache").mean().item()
attr = attribution_scores(model, disc, c0, x0)
attr_rank = {(i.item() // n_h, i.item() % n_h): r + 1
             for r, i in enumerate(attr.flatten().argsort(descending=True))}

# Evaluation prompts disjoint from discovery
seen = {p[0] for p in disc_pairs}
eval_pairs = [p for p in make_pairs(model, 240, seed=1) if p[0] not in seen][:N_EVAL]
ev = build_batches(model, eval_pairs)
clean = per_example(model, ev, "clean", [], "clean_cache").mean().item()
corr = per_example(model, ev, "corr", [], "corr_cache").mean().item()
print(f"Eval pairs: {len(eval_pairs)} | clean {clean:.2f} | corrupted {corr:.2f}\n", flush=True)

# Single-head resample ablation on clean prompts
kept = torch.zeros(n_l, n_h)
t0 = time.time()
for l in range(n_l):
    for h in range(n_h):
        v = per_example(model, ev, "clean", [(l, h)], "corr_cache").mean().item()
        kept[l, h] = (v - corr) / (clean - corr)      # 1 = no effect, <1 = removal hurts, >1 = removal helps
    print(f"  layer {l + 1}/{n_l} done, {time.time() - t0:.0f}s elapsed", flush=True)

drop = 1 - kept
all_heads = [(l, h) for l in range(n_l) for h in range(n_h)]
ranked = sorted(all_heads, key=lambda hd: drop[hd].item(), reverse=True)

print("\n15 heads whose single removal hurts most:")
print(f"{'head':>9} {'kept':>6} {'attribution rank (of 144)':>27}")
for hd in ranked[:15]:
    print(f"{str(hd):>9} {kept[hd].item():>6.2f} {attr_rank[hd]:>27}")

print("\n5 heads whose removal helps most (suppressors):")
for hd in ranked[::-1][:5]:
    print(f"{str(hd):>9} {kept[hd].item():>6.2f} {attr_rank[hd]:>27}")

top10 = set(ranked[:10])
print("\nOf the 10 most necessary heads, how many sit in the attribution top-k (chance in brackets):")
for k in (10, 20, 40):
    atop = {hd for hd, r in attr_rank.items() if r <= k}
    print(f"  top-{k}: {len(top10 & atop)}/10   [about {10 * k / 144:.1f} by chance]")
print(f"Rank correlation, attribution score vs drop, all 144 heads: {rank_corr(attr, drop):.2f}")

json.dump({"kept": kept.tolist(), "attr": attr.tolist()}, open("experiments/gpt2_necessity_scan.json", "w"))
print("\nDone.")