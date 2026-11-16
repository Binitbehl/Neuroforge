import torch
from transformer_lens import HookedTransformer
from nforge.data.ioi import make_pairs
from nforge.core import build_batches, per_example

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
N_LAYERS, N_HEADS = model.cfg.n_layers, model.cfg.n_heads
ALL_HEADS = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]

# Found on the DISCOVERY set (seed 0, 30 pairs) in the last run
POSITIVE = [(8, 10), (8, 6), (5, 5), (7, 9), (3, 0), (7, 3), (9, 9), (5, 9), (10, 0), (9, 7)]
NEGATIVE = [(10, 7), (11, 10), (11, 2)]

# Evaluate on held-out prompts that never appeared in discovery
disc_pairs = make_pairs(model, 30, seed=0)
seen = {p[0] for p in disc_pairs}
held_pairs = [p for p in make_pairs(model, 240, seed=1) if p[0] not in seen][:60]
held = build_batches(model, held_pairs)
print(f"Held-out pairs: {len(held_pairs)}", flush=True)

clean = per_example(model, held, "clean", [], "clean_cache")
corrupted = per_example(model, held, "corr", [], "corr_cache")
print(f"Full model: {clean.mean():.2f} | Corrupted input: {corrupted.mean():.2f} "
      f"({corrupted.mean() / clean.mean():.0%} of clean)\n")


def faithfulness(keep):
    """Run the CLEAN prompt, with every head outside `keep` replaced by its corrupted-run value."""
    keep = set(keep)
    ablate = [hd for hd in ALL_HEADS if hd not in keep]
    return per_example(model, held, "clean", ablate, "corr_cache")


def report(label, vals):
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(vals), (1000, len(vals)), generator=g)
    r = vals[idx].mean(1) / clean[idx].mean(1)
    print(f"{label:34s} {vals.mean() / clean.mean():>6.0%}  [{r.quantile(0.025):.0%}, {r.quantile(0.975):.0%}]",
          flush=True)


print(f"{'Heads kept (all others ablated)':34s} {'kept':>6}  95% CI")
report("Nothing (floor)", faithfulness([]))
report("Positive 10", faithfulness(POSITIVE))
report("Positive 10 + 3 negative", faithfulness(POSITIVE + NEGATIVE))

# Baseline: random head sets of the same size, drawn outside the circuit
circuit = set(POSITIVE + NEGATIVE)
pool = [hd for hd in ALL_HEADS if hd not in circuit]
for size in (10, 13):
    rand = []
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        pick = [pool[i] for i in torch.randperm(len(pool), generator=g)[:size]]
        rand.append((faithfulness(pick).mean() / clean.mean()).item())
    rt = torch.tensor(rand)
    print(f"{'Random ' + str(size) + ' heads (20 draws)':34s} {rt.mean():>6.0%}  ± {rt.std():.0%}  "
          f"(max {rt.max():.0%})", flush=True)

print("\nDone.")
