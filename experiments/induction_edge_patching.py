import gc
import os
import torch
from transformer_lens import HookedTransformer

os.environ.setdefault("HF_TOKEN", "placeholder")

MODEL = "pythia-70m"
STEPS = [1000, 2000, 4000, 16000, 143000]
SEQ_LEN, BATCH, SEED = 50, 32, 1
TARGET = slice(SEQ_LEN + 1, 2 * SEQ_LEN)
SOURCES = [(2, 1), (2, 7), (2, 2)]        # (2,2) is a control
READER_LAYER = 3


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


for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    model.set_use_split_qkv_input(True)
    clean, corrupt = make_batch(model, SEED)
    only_z = lambda n: n.endswith("attn.hook_z")
    with torch.no_grad():
        cl, clean_cache = model.run_with_cache(clean, names_filter=only_z)
        xl, corr_cache = model.run_with_cache(corrupt, names_filter=only_z)
    c, x = metric(cl, clean).mean().item(), metric(xl, clean).mean().item()

    def score(hooks):
        with torch.no_grad():
            v = metric(model.run_with_hooks(clean, fwd_hooks=hooks), clean).mean().item()
        return (v - x) / (c - x)

    def contribution_change(src):
        l, h = src
        name = f"blocks.{l}.attn.hook_z"
        dz = corr_cache[name][:, :, h, :] - clean_cache[name][:, :, h, :]
        return dz @ model.W_O[l, h]                       # change in the residual stream, [batch, pos, d_model]

    def edge(src, kind, scale=1.0):
        delta = contribution_change(src) * scale

        def hook(act, hook):
            return act + delta[:, :, None, :]             # same edit for every head of the reading layer
        return [(f"blocks.{READER_LAYER}.hook_{kind}_input", hook)]

    def full_ablation(src):
        l, h = src
        name = f"blocks.{l}.attn.hook_z"

        def hook(z, hook):
            z[:, :, h, :] = corr_cache[name][:, :, h, :]
            return z
        return [(name, hook)]

    noop = score(edge(SOURCES[0], "k", scale=0.0))
    print(f"\n=== step {step} === no-op check (must be 1.00): {noop:.2f}")
    print(f"  {'source':>7} {'full ablation':>14} {'keys@L3':>9} {'queries@L3':>11} {'values@L3':>10}")
    for src in SOURCES:
        print(f"  {str(src):>7} {score(full_ablation(src)):>14.2f} {score(edge(src, 'k')):>9.2f} "
              f"{score(edge(src, 'q')):>11.2f} {score(edge(src, 'v')):>10.2f}", flush=True)
    del model
    gc.collect()

print("\nDone.")