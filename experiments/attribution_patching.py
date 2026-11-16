import time
import torch
from transformer_lens import HookedTransformer
from nforge.data.ioi import make_pairs
from nforge.core import build_batches, per_example, logit_diff, only_z

model = HookedTransformer.from_pretrained("gpt2")
model.eval()
N_LAYERS, N_HEADS = model.cfg.n_layers, model.cfg.n_heads

# Same discovery set as the exact-patching run (seed 0, 30 pairs)
pairs = make_pairs(model, 30, seed=0)
batches = build_batches(model, pairs)
clean_ld = per_example(model, batches, "clean", [], "clean_cache").mean().item()
corr_ld = per_example(model, batches, "corr", [], "corr_cache").mean().item()
print(f"Clean {clean_ld:.2f} | Corrupted {corr_ld:.2f}", flush=True)

# ---- Attribution patching: gradients on the CORRUPTED run ----
t0 = time.time()
attr = torch.zeros(N_LAYERS, N_HEADS)
n = 0
for b in batches:
    grads = {}

    def save_grad(g, hook):
        grads[hook.name] = g.detach()

    with model.hooks(bwd_hooks=[(only_z, save_grad)]):
        logits = model(b["corr"])
        logit_diff(logits, b).sum().backward()   # sum so each example's gradient is independent

    for l in range(N_LAYERS):
        name = f"blocks.{l}.attn.hook_z"
        delta = b["clean_cache"][name] - b["corr_cache"][name]     # [batch, pos, head, d_head]
        attr[l] += (delta * grads[name]).sum(dim=(1, 3)).sum(0).cpu()
    n += b["clean"].shape[0]

scores = attr / n / (clean_ld - corr_ld)     # same normalisation as exact patching
print(f"Attribution patching took {time.time() - t0:.1f}s\n", flush=True)

ranked = sorted(((l, h) for l in range(N_LAYERS) for h in range(N_HEADS)),
                key=lambda hd: scores[hd].item(), reverse=True)
print("Top 10 positive heads (attribution):")
for hd in ranked[:10]:
    print(f"  {hd}: {scores[hd].item():+.3f}")
print("Most negative:", [(hd, round(scores[hd].item(), 3)) for hd in ranked[::-1][:3]])

# ---- Compare with the exact patching result from the earlier run ----
EXACT_TOP10 = {(8, 10), (8, 6), (5, 5), (7, 9), (3, 0), (7, 3), (9, 9), (5, 9), (10, 0), (9, 7)}
EXACT_NEG = {(10, 7), (11, 10), (11, 2)}
neg_ranked = ranked[::-1][:3]
print(f"\nOverlap of top 10 with exact patching: {len(EXACT_TOP10 & set(ranked[:10]))}/10")
print(f"Overlap of top 20 with exact top 10:   {len(EXACT_TOP10 & set(ranked[:20]))}/10")
print(f"Overlap of negative heads:             {len(EXACT_NEG & set(neg_ranked))}/3")
