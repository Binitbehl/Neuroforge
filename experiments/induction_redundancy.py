import gc
import json
import os
import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("HF_TOKEN", "placeholder")

MODEL = "pythia-70m"
STEPS = [1000, 2000, 4000, 16000, 143000]
SEQ_LEN, BATCH = 50, 32
TARGET = slice(SEQ_LEN + 1, 2 * SEQ_LEN)
KS = [1, 2, 3, 5, 8]
EVAL_SEEDS = [1, 2]        # never used for ranking (the saved scores came from seed 0)
N_RANDOM = 5

saved = json.load(open("experiments/induction_scores.json"))


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


def run_overwrite(model, tokens, clean, heads, cache):
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    hooks = []
    for l, hs in by_layer.items():
        name = f"blocks.{l}.attn.hook_z"

        def hook(z, hook, hs=hs, name=name):
            z[:, :, hs, :] = cache[name][:, :, hs, :]
            return z

        hooks.append((name, hook))
    with torch.no_grad():
        return metric(model.run_with_hooks(tokens, fwd_hooks=hooks), clean)


def prepare(model, seed):
    clean, corrupt = make_batch(model, seed)
    only_z = lambda n: n.endswith("attn.hook_z")
    with torch.no_grad():
        cl, clean_cache = model.run_with_cache(clean, names_filter=only_z)
        xl, corr_cache = model.run_with_cache(corrupt, names_filter=only_z)
    return dict(clean=clean, corr_cache=corr_cache,
                c=metric(cl, clean).mean().item(), x=metric(xl, clean).mean().item())


def ranked(scores):
    n_h = scores.shape[1]
    return [(i.item() // n_h, i.item() % n_h) for i in scores.flatten().argsort(descending=True)]


for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    order = ranked(torch.tensor(saved[str(step)]))
    pool = order[max(KS):]                      # random sets never touch the top-8 ranked heads
    evals = [prepare(model, s) for s in EVAL_SEEDS]

    def kept(heads):
        return [(run_overwrite(model, d["clean"], d["clean"], heads, d["corr_cache"]).mean().item() - d["x"])
                / (d["c"] - d["x"]) for d in evals]

    print(f"\n=== step {step} ===  ranking: {order[:5]}")
    print(f"{'k':>3} {'top-k kept':>11} {'[min, max over seeds]':>24} {'random-k kept':>18}", flush=True)
    for k in KS:
        circ = kept(order[:k])
        rand = []
        for s in range(N_RANDOM):
            g = torch.Generator().manual_seed(200 + s)
            pick = [pool[i] for i in torch.randperm(len(pool), generator=g)[:k]]
            rand.append(sum(kept(pick)) / len(evals))
        rt = torch.tensor(rand)
        print(f"{k:>3} {sum(circ) / len(circ):>11.2f} {'[' + format(min(circ), '.2f') + ', ' + format(max(circ), '.2f') + ']':>24} "
              f"{rt.mean():>11.2f} ± {rt.std():.2f}", flush=True)
    del model
    gc.collect()

print("\nDone.")