import gc
import json
import torch
from nforge.models.loading import load_model
from nforge.data.ioi import make_pairs
from nforge.data.induction import make_induction_batch
from nforge.activations.batches import build_batches
from nforge.patching.ablation import per_example
from nforge.discovery.attribution import attribution_scores
from nforge.stability.compare import top_k, rank_corr


def endpoints(model, batches):
    c = per_example(model, batches, "clean", [], "clean_cache").mean().item()
    x = per_example(model, batches, "corr", [], "corr_cache").mean().item()
    return c, x


# ---- Part A: IOI on GPT-2 must reproduce the earlier attribution top-10 ----
EXPECTED = {(8, 10), (8, 6), (7, 9), (5, 9), (7, 3), (9, 9), (5, 5), (3, 0), (10, 0), (10, 6)}
model = load_model("gpt2")
ioi = build_batches(model, make_pairs(model, 30, seed=0))
c, x = endpoints(model, ioi)
got = set(top_k(attribution_scores(model, ioi, c, x), 10))
print(f"Part A, GPT-2 IOI: attribution top-10 matches the earlier run on {len(got & EXPECTED)}/10 heads\n", flush=True)
del model, ioi
gc.collect()

# ---- Part B: induction on Pythia-70M through the same code ----
exact = json.load(open("experiments/induction_scores.json"))            # exact patching, seed 0
earlier = json.load(open("experiments/induction_necessity_scan.json"))  # single-head ablation, seed 1
CHECK = [(2, 1), (3, 6), (3, 1)]

print(f"{'step':>7} {'rank corr (attr vs exact)':>26} {'top-5 overlap':>14} {'attr rank of (2,1)':>19} {'max |kept diff|':>16}")
for step in [1000, 2000, 4000, 16000, 143000]:
    model = load_model("pythia-70m", step=step)
    n_h = model.cfg.n_heads
    rank_set = [make_induction_batch(model, seed=0)]
    eval_set = [make_induction_batch(model, seed=1)]

    c, x = endpoints(model, rank_set)
    attr = attribution_scores(model, rank_set, c, x)
    order = attr.flatten().argsort(descending=True)
    rank21 = (order == 2 * n_h + 1).nonzero().item() + 1
    overlap = len(set(top_k(attr, 5)) & set(top_k(exact[str(step)], 5)))

    c1, x1 = endpoints(model, eval_set)
    diffs = []
    for l, h in CHECK:
        v = per_example(model, eval_set, "clean", [(l, h)], "corr_cache").mean().item()
        diffs.append(abs((v - x1) / (c1 - x1) - earlier[str(step)][l][h]))
    print(f"{step:>7} {rank_corr(attr, exact[str(step)]):>26.2f} {overlap:>11}/5 {rank21:>19} {max(diffs):>16.3f}", flush=True)
    del model
    gc.collect()

print("\nDone.")