from transformer_lens import HookedTransformer
from nforge.data.ioi import make_pairs
from nforge.activations.batches import build_batches, logit_diff, only_z
from nforge.patching.ablation import per_example
from nforge.discovery.attribution import attribution_scores
from nforge.validation.circuit import validate

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
ALL_HEADS = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]

# Discovery set -> rank heads
disc_pairs = make_pairs(model, 30, seed=0)
disc = build_batches(model, disc_pairs)
c_ld = per_example(model, disc, "clean", [], "clean_cache").mean().item()
x_ld = per_example(model, disc, "corr", [], "corr_cache").mean().item()
scores = attribution_scores(model, disc, c_ld, x_ld)
order = sorted(ALL_HEADS, key=lambda hd: scores[hd].item(), reverse=True)
rank = {hd: i + 1 for i, hd in enumerate(order)}
print("Attribution rank of (0,1):", rank[(0, 1)], "| of (0,10):", rank[(0, 10)])

circuit = order[:18] + order[-3:]          # top 18 positive + 3 most negative

# Fresh TEST set: never used for any choice, and disjoint from earlier sets
seen = {p[0] for p in disc_pairs} | {p[0] for p in make_pairs(model, 240, seed=1)}
test_pairs = [p for p in make_pairs(model, 400, seed=2) if p[0] not in seen][:60]
test = build_batches(model, test_pairs)
print(f"Test pairs: {len(test_pairs)}\n", flush=True)

r = validate(model, test, circuit)
f = lambda t: f"{t[0]:+.2f}  [{t[1]:+.2f}, {t[2]:+.2f}]"
print(f"Full model logit diff {r['full']:.2f} | all ablated {r['floor']:.2f}")
print(f"Faithfulness (keep only circuit, 1 = full model): {f(r['faithfulness'])}")
print(f"Necessity (ablate only circuit, 0 = fully broken):  {f(r['necessity_kept'])}")
m, s, mx = r["random"]
print(f"Random same-size sets: {m:+.2f} ± {s:.2f} (max {mx:+.2f})\n")

print("Minimality: change in score (fraction of gap) when each head is removed")
for hd, t in sorted(r["minimality"].items(), key=lambda kv: kv[1][0]):
    flag = "" if t[1] <= 0 <= t[2] else "  <- CI excludes 0"
    print(f"  {str(hd):9s} {f(t)}{flag}")
print("\nDone.")
