import gc
import torch
from nforge.models.loading import load_model
from nforge.data.induction import make_induction_batch
from nforge.discovery.pipeline import discover
from nforge.patching.ablation import per_example

SIGMAS = [0.02, 0.05, 0.1]          # noise as a fraction of each tensor's std (guesses)
KEY = [(2, 1), (2, 7), (3, 1), (3, 3), (3, 5), (3, 6)]


def repeat_accuracy(model, seq_len=50):
    b = make_induction_batch(model, seq_len=seq_len, seed=1)
    with torch.no_grad():
        pred = model(b["clean"])[:, :-1].argmax(-1)
    return (pred == b["clean"][:, 1:])[:, seq_len + 1:2 * seq_len].float().mean().item()


for sigma in SIGMAS:
    model = load_model("pythia-70m", step=143000)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(sigma * p.std() * torch.randn(p.shape, generator=g).to(p.device))
    rank_b = [make_induction_batch(model, seed=0)]
    eval_b = [make_induction_batch(model, seed=1)]
    c = per_example(model, eval_b, "clean", [], "clean_cache").mean().item()
    x = per_example(model, eval_b, "corr", [], "corr_cache").mean().item()
    r = discover(model, rank_b, eval_b)
    print(f"\n=== noise {sigma} x tensor std ===")
    print(f"  repeat accuracy {repeat_accuracy(model):.0%} | gap (clean - corrupted) {c - x:.2f}")
    print("  necessary (kept < 0.9):", sorted(r["necessary"]))
    print("  kept:", {hd: round(r["kept"][hd].item(), 2) for hd in KEY}, flush=True)
    del model
    gc.collect()

print("\nDone.")