import gc
import json
import torch
from transformer_lens import HookedTransformer

MODEL = "pythia-70m"
STEPS = [1000, 2000, 4000, 16000, 143000]
SEQ_LEN, BATCH, TOP = 50, 32, 5
TARGET_SLICE = slice(SEQ_LEN + 1, 2 * SEQ_LEN)   # positions that predict the repeated tokens


def make_batch(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    bos = torch.full((BATCH, 1), model.tokenizer.bos_token_id)
    r = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    r_other = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    dev = model.cfg.device
    clean = torch.cat([bos, r, r], 1).to(dev)           # second half repeats the first
    corrupt = torch.cat([bos, r_other, r], 1).to(dev)   # first half replaced: nothing to copy from
    return clean, corrupt


def metric(logits, clean):
    """Mean log-prob of the correct repeated token, per example."""
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    lp = lp.gather(-1, clean[:, 1:, None]).squeeze(-1)
    return lp[:, TARGET_SLICE].mean(1)


def patch_scores(model, clean, corrupt):
    n_l, n_h = model.cfg.n_layers, model.cfg.n_heads
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(clean, names_filter=lambda n: n.endswith("attn.hook_z"))
        c = metric(clean_logits, clean).mean().item()
        x = metric(model(corrupt), clean).mean().item()
        scores = torch.zeros(n_l, n_h)
        for l in range(n_l):
            name = f"blocks.{l}.attn.hook_z"
            for h in range(n_h):
                def hook(z, hook, h=h, name=name):
                    z[:, :, h, :] = cache[name][:, :, h, :]
                    return z
                p = metric(model.run_with_hooks(corrupt, fwd_hooks=[(name, hook)]), clean).mean().item()
                scores[l, h] = (p - x) / (c - x)
    return scores, c, x


def top_set(scores, k):
    n_h = scores.shape[1]
    idx = scores.flatten().argsort(descending=True)[:k]
    return {(i.item() // n_h, i.item() % n_h) for i in idx}


def rank_corr(a, b):
    ra = a.flatten().argsort().argsort().float()
    rb = b.flatten().argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


all_scores = {}
for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    clean, corrupt = make_batch(model)
    scores, c, x = patch_scores(model, clean, corrupt)
    all_scores[step] = scores
    print(f"\nstep {step}: clean log-prob {c:.2f} | corrupted {x:.2f} | gap {c - x:.2f}", flush=True)
    ranked = sorted(((l, h) for l in range(scores.shape[0]) for h in range(scores.shape[1])),
                    key=lambda hd: scores[hd].item(), reverse=True)
    for hd in ranked[:TOP]:
        print(f"   {hd}: {scores[hd].item():+.3f}", flush=True)
    del model
    gc.collect()

json.dump({str(k): v.tolist() for k, v in all_scores.items()},
          open("experiments/induction_scores.json", "w"))

final = STEPS[-1]
print(f"\nComparison with step {final} (top-{TOP} overlap, rank correlation of all 48 head scores):")
print(f"{'step':>7} {'Jaccard':>8} {'rank corr':>10}")
for step in STEPS[:-1]:
    a, b = top_set(all_scores[step], TOP), top_set(all_scores[final], TOP)
    print(f"{step:>7} {len(a & b) / len(a | b):>8.2f} {rank_corr(all_scores[step], all_scores[final]):>10.2f}")
print("\nDone.")
