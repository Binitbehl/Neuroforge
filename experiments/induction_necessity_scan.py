import gc
import json
import os
import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("HF_TOKEN", "placeholder")

MODEL = "pythia-70m"
STEPS = [1000, 2000, 4000, 16000, 143000]
SEQ_LEN, BATCH, SEED = 50, 32, 1            # seed 1 was never used for ranking
TARGET = slice(SEQ_LEN + 1, 2 * SEQ_LEN)

saved = json.load(open("experiments/induction_scores.json"))   # seed-0 patching scores


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


results = {}
for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    clean, corrupt = make_batch(model, SEED)
    only_z = lambda n: n.endswith("attn.hook_z")
    with torch.no_grad():
        cl, _ = model.run_with_cache(clean, names_filter=only_z)
        xl, corr_cache = model.run_with_cache(corrupt, names_filter=only_z)
    c, x = metric(cl, clean).mean().item(), metric(xl, clean).mean().item()

    kept = torch.zeros(n_l, n_h)
    for l in range(n_l):
        name = f"blocks.{l}.attn.hook_z"
        for h in range(n_h):
            def hook(z, hook, h=h, name=name):
                z[:, :, h, :] = corr_cache[name][:, :, h, :]
                return z
            with torch.no_grad():
                v = metric(model.run_with_hooks(clean, fwd_hooks=[(name, hook)]), clean).mean().item()
            kept[l, h] = (v - x) / (c - x)
    results[step] = kept.tolist()

    patch = torch.tensor(saved[str(step)])
    patch_rank = {(i.item() // n_h, i.item() % n_h): r + 1
                  for r, i in enumerate(patch.flatten().argsort(descending=True))}
    order = kept.flatten().argsort()            # lowest 'kept' = most necessary
    print(f"\n=== step {step} === single-head ablation, 10 most necessary heads")
    print(f"{'head':>8} {'kept':>6} {'patching rank (seed 0)':>24}")
    for i in order[:10]:
        hd = (i.item() // n_h, i.item() % n_h)
        print(f"{str(hd):>8} {kept[hd].item():>6.2f} {patch_rank[hd]:>24}", flush=True)
    del model
    gc.collect()

json.dump({str(k): v for k, v in results.items()}, open("experiments/induction_necessity_scan.json", "w"))
print("\nDone.")