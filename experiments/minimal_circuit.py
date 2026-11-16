import torch
from transformer_lens import HookedTransformer
from nforge.ioi_data import make_pairs
from nforge.core import build_batches, per_example, attribution_scores

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
N_LAYERS, N_HEADS = model.cfg.n_layers, model.cfg.n_heads
ALL_HEADS = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]

disc_pairs = make_pairs(model, 30, seed=0)
disc = build_batches(model, disc_pairs)
c_ld = per_example(model, disc, "clean", [], "clean_cache").mean().item()
x_ld = per_example(model, disc, "corr", [], "corr_cache").mean().item()
scores = attribution_scores(model, disc, c_ld, x_ld)
order = sorted(ALL_HEADS, key=lambda hd: scores[hd].item(), reverse=True)
NEGATIVES = order[-3:]

seen = {p[0] for p in disc_pairs}
held_pairs = [p for p in make_pairs(model, 240, seed=1) if p[0] not in seen][:60]
held = build_batches(model, held_pairs)

full = per_example(model, held, "clean", [], "clean_cache")
floor = per_example(model, held, "clean", ALL_HEADS, "corr_cache")
print(f"Full model: logit diff {full.mean():.2f}, accuracy {(full > 0).float().mean():.0%}")
print(f"All ablated: logit diff {floor.mean():.2f}, accuracy {(floor > 0).float().mean():.0%}\n")

print("Heads ranked 11-25 (what closed the gap):")
for i, hd in enumerate(order[10:25], start=11):
    print(f"  #{i:>2} {hd}: {scores[hd].item():+.3f}")
print("Lowest-ranked (negative) heads:", [(hd, round(scores[hd].item(), 3)) for hd in NEGATIVES], "\n")


def evaluate(keep):
    keep = set(keep)
    vals = per_example(model, held, "clean", [h for h in ALL_HEADS if h not in keep], "corr_cache")
    return vals.mean().item(), (vals > 0).float().mean().item()


print(f"{'k':>3} {'logit diff':>11} {'accuracy':>9} | {'+3 negatives: logit diff':>26} {'accuracy':>9}")
for k in [10, 12, 14, 16, 18, 20, 22, 25, 30, 40]:
    ld, acc = evaluate(order[:k])
    ld2, acc2 = evaluate(order[:k] + NEGATIVES)
    print(f"{k:>3} {ld:>11.2f} {acc:>9.0%} | {ld2:>26.2f} {acc2:>9.0%}", flush=True)

print("\nDone.")