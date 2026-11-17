import torch
from transformer_lens import HookedTransformer
from nforge.data.ioi import make_pairs
from nforge.activations.batches import build_batches, logit_diff, only_z
from nforge.patching.ablation import per_example
from nforge.discovery.attribution import attribution_scores

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
ALL_HEADS = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]

disc_pairs = make_pairs(model, 30, seed=0)
disc = build_batches(model, disc_pairs)
c_ld = per_example(model, disc, "clean", [], "clean_cache").mean().item()
x_ld = per_example(model, disc, "corr", [], "corr_cache").mean().item()
scores = attribution_scores(model, disc, c_ld, x_ld)
order = sorted(ALL_HEADS, key=lambda hd: scores[hd].item(), reverse=True)

seen = {p[0] for p in disc_pairs}
held_pairs = [p for p in make_pairs(model, 240, seed=1) if p[0] not in seen][:60]
held = build_batches(model, held_pairs)


def ld_keep(keep):
    keep = set(keep)
    ablate = [hd for hd in ALL_HEADS if hd not in keep]
    return per_example(model, held, "clean", ablate, "corr_cache").mean().item()


base = order[:16]
print(f"Top 16 kept: {ld_keep(base):.2f}\n")

print("Add ONE head to the top 16:")
for i, hd in enumerate(order[16:30], start=17):
    print(f"  + #{i:>2} {hd}: {ld_keep(base + [hd]):.2f}", flush=True)

print(f"\n  + both #17 and #18: {ld_keep(base + order[16:18]):.2f}")

top18 = order[:18]
full18 = ld_keep(top18)
print(f"\nTop 18 kept: {full18:.2f}. Remove ONE head from the top 18:")
for i, hd in enumerate(top18, start=1):
    print(f"  - #{i:>2} {hd}: {ld_keep([h for h in top18 if h != hd]):.2f}  (change {ld_keep([h for h in top18 if h != hd]) - full18:+.2f})",
          flush=True)

print("\nDone.")
