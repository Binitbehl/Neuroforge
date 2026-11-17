import gc
import os
import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("HF_TOKEN", "placeholder")

MODEL = "pythia-70m"
STEPS = [1000, 2000, 4000, 16000, 143000]
SEQ_LEN, BATCH, SEED = 50, 32, 1
TARGET = slice(SEQ_LEN + 1, 2 * SEQ_LEN)
FIRST = slice(1, SEQ_LEN + 1)                    # positions of the first copy
SECOND = slice(SEQ_LEN + 1, 2 * SEQ_LEN + 1)     # positions of the repeat
HEADS = [(2, 1), (2, 7)]


def make_batch(model, seed):
    g = torch.Generator().manual_seed(seed)
    bos = torch.full((BATCH, 1), model.tokenizer.bos_token_id)
    r = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    other = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    dev = model.cfg.device
    return torch.cat([bos, r, r], 1).to(dev), torch.cat([bos, other, r], 1).to(dev)


def metric(logits, clean):
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1).gather(-1, clean[:, 1:, None]).squeeze(-1)
    return lp[:, TARGET].mean(1)


def top(scores, k):
    n_h = scores.shape[1]
    return [(i.item() // n_h, i.item() % n_h) for i in scores.flatten().argsort(descending=True)[:k]]


for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    clean, corrupt = make_batch(model, SEED)
    with torch.no_grad():
        cl, cache = model.run_with_cache(
            clean, names_filter=lambda n: n.endswith("attn.hook_pattern") or n.endswith("attn.hook_z"))
        xl, corr_cache = model.run_with_cache(corrupt, names_filter=lambda n: n.endswith("attn.hook_z"))
    c, x = metric(cl, clean).mean().item(), metric(xl, clean).mean().item()

    # 1. Attention from each position to the token right before it
    prev = torch.zeros(n_l, n_h)
    for l in range(n_l):
        pat = cache[f"blocks.{l}.attn.hook_pattern"]                     # [batch, head, q, k]
        diag = torch.diagonal(pat, offset=-1, dim1=-2, dim2=-1)          # attention q -> q-1
        prev[l] = diag[..., 1:].mean(dim=(0, 2)).cpu()

    def kept(head, positions):
        l, h = head
        name = f"blocks.{l}.attn.hook_z"

        def hook(z, hook):
            z[:, positions, h, :] = corr_cache[name][:, positions, h, :]
            return z

        with torch.no_grad():
            v = metric(model.run_with_hooks(clean, fwd_hooks=[(name, hook)]), clean).mean().item()
        return (v - x) / (c - x)

    print(f"\n=== step {step} ===")
    print("  top-5 heads by previous-token attention:", [(hd, round(prev[hd].item(), 2)) for hd in top(prev, 5)])
    print(f"  {'head':>7} {'prev-tok attn':>14} {'ablate 1st half':>16} {'ablate 2nd half':>16} {'ablate both':>12}")
    for hd in HEADS:
        print(f"  {str(hd):>7} {prev[hd].item():>14.2f} {kept(hd, FIRST):>16.2f} "
              f"{kept(hd, SECOND):>16.2f} {kept(hd, slice(0, 2 * SEQ_LEN + 1)):>12.2f}", flush=True)
    del model
    gc.collect()

print("\nDone.")