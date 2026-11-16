import gc
import torch
from transformer_lens import HookedTransformer
from nforge.ioi_data import make_pairs
from nforge.core import build_batches, per_example

MODEL = "pythia-70m"
STEPS = [4000, 143000]

print(f"{'step':>7} {'clean ld':>9} {'corrupt ld':>11} {'gap':>6} {'clean acc':>10} {'corrupt acc':>12}")
for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    pairs = make_pairs(model, 60, seed=0)
    batches = build_batches(model, pairs)
    clean = per_example(model, batches, "clean", [], "clean_cache")
    corr = per_example(model, batches, "corr", [], "corr_cache")
    print(f"{step:>7} {clean.mean():>9.2f} {corr.mean():>11.2f} {clean.mean() - corr.mean():>6.2f} "
          f"{(clean > 0).float().mean():>10.0%} {(corr > 0).float().mean():>12.0%}", flush=True)
    del model, batches
    gc.collect()