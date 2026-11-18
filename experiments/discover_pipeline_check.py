import gc
import json
from nforge.models.loading import load_model
from nforge.data.induction import make_induction_batch
from nforge.discovery.pipeline import discover

earlier = json.load(open("experiments/induction_necessity_scan.json"))   # saved single-head ablations, seed 1
STEPS = [1000, 4000, 143000]

for step in STEPS:
    model = load_model("pythia-70m", step=step)
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    r = discover(model, [make_induction_batch(model, seed=0)], [make_induction_batch(model, seed=1)])
    old = earlier[str(step)]
    diff = max(abs(r["kept"][l][h].item() - old[l][h]) for l in range(n_l) for h in range(n_h))
    print(f"\n=== step {step} ===")
    print("  attribution top-5:        ", sorted(r["shortlist"]))
    print("  necessary (kept < 0.9):   ", sorted(r["necessary"]))
    print("  necessary but missed:     ", sorted(r["missed_by_attribution"]))
    print(f"  max |kept diff| vs saved scan: {diff:.3f}", flush=True)
    del model
    gc.collect()

print("\nDone.")