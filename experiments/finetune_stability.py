import gc
import glob
import json
import os
import random
import sys
import time

import torch

from nforge.models.loading import load_model
from nforge.data.induction import make_induction_batch
from nforge.discovery.pipeline import discover
from nforge.patching.ablation import per_example
from nforge.stability.compare import jaccard, rank_corr

DATA = sys.argv[1] if len(sys.argv) > 1 else "code"     # "code" or "wiki"
STEPS, BATCH, SEQ, LR, MAX_MINUTES = 300, 8, 128, 1e-5, 25
KEY_HEADS = [(2, 1), (2, 7), (3, 0), (3, 1), (3, 3), (3, 5), (3, 6)]
torch.manual_seed(0)


def load_text(kind):
    if kind == "code":
        import transformers
        root = os.path.dirname(transformers.__file__)
        files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
        random.Random(0).shuffle(files)
        parts, n = [], 0
        for f in files:
            s = open(f, encoding="utf-8", errors="ignore").read()
            parts.append(s)
            n += len(s)
            if n > 1_500_000:
                break
        return "\n".join(parts)
    from datasets import load_dataset                       # needs a download; not tested here
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return "\n".join(t for t in ds["text"] if t.strip())[:1_500_000]


def repeat_accuracy(model, seq_len=50):
    b = make_induction_batch(model, seq_len=seq_len, seed=1)
    with torch.no_grad():
        pred = model(b["clean"])[:, :-1].argmax(-1)
    return (pred == b["clean"][:, 1:])[:, seq_len + 1:2 * seq_len].float().mean().item()


def heldout_loss(model, held):
    model.eval()
    with torch.no_grad():
        return sum(model(held[i:i + 8], return_type="loss").item() for i in range(0, len(held), 8)) / (len(held) // 8)


def analyse(model, held):
    rank_b = [make_induction_batch(model, seed=0)]
    eval_b = [make_induction_batch(model, seed=1)]
    c = per_example(model, eval_b, "clean", [], "clean_cache").mean().item()
    x = per_example(model, eval_b, "corr", [], "corr_cache").mean().item()
    return dict(c=c, x=x, acc=repeat_accuracy(model), loss=heldout_loss(model, held),
                r=discover(model, rank_b, eval_b))


model = load_model("pythia-70m", step=143000)
text = load_text(DATA)
ids = model.tokenizer(text, return_tensors="pt").input_ids[0]
chunks = ids[: (len(ids) // SEQ) * SEQ].view(-1, SEQ)
chunks = chunks[torch.randperm(len(chunks), generator=torch.Generator().manual_seed(0))]
held, train = chunks[:32], chunks[32:]
print(f"Data: {DATA} | {len(ids):,} tokens | {len(train)} train chunks | {len(held)} held-out chunks", flush=True)

print("\nAnalysing the base model...", flush=True)
before = analyse(model, held)
print(f"  base: held-out loss {before['loss']:.3f} | induction repeat accuracy {before['acc']:.0%}", flush=True)

print("\nFine-tuning...", flush=True)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.arange((step - 1) * BATCH, step * BATCH) % len(train)
    loss = model(train[idx], return_type="loss")
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step <= 3 or step % 25 == 0:
        print(f"  step {step:>3} loss {loss.item():.3f} ({(time.time() - t0) / step:.1f}s/step)", flush=True)
    if time.time() - t0 > MAX_MINUTES * 60:
        print(f"  stopped at step {step} (time limit)", flush=True)
        break
model.eval()
for p in model.parameters():
    p.requires_grad_(True)

print("\nAnalysing the fine-tuned model...", flush=True)
after = analyse(model, held)

b, a = before["r"], after["r"]
print("\n================ BASE vs FINE-TUNED ================")
print(f"held-out loss on {DATA}:      {before['loss']:.3f} -> {after['loss']:.3f}")
print(f"induction repeat accuracy:   {before['acc']:.0%} -> {after['acc']:.0%}")
print(f"induction score (clean / corrupted): {before['c']:.2f} / {before['x']:.2f} -> {after['c']:.2f} / {after['x']:.2f}")
print(f"\nnecessary heads (kept < 0.9):\n  base:       {sorted(b['necessary'])}\n  fine-tuned: {sorted(a['necessary'])}")
print(f"  Jaccard: {jaccard(b['necessary'], a['necessary']):.2f}")
print(f"necessary but missed by attribution:\n  base: {sorted(b['missed_by_attribution'])}\n  fine-tuned: {sorted(a['missed_by_attribution'])}")
print(f"attribution top-5:\n  base: {sorted(b['shortlist'])}\n  fine-tuned: {sorted(a['shortlist'])}")
print(f"rank correlation of all 48 'kept' values: {rank_corr(b['kept'], a['kept']):.2f}")
print(f"\n{'head':>7} {'kept (base)':>12} {'kept (fine-tuned)':>18}")
for l, h in KEY_HEADS:
    print(f"{str((l, h)):>7} {b['kept'][l, h].item():>12.2f} {a['kept'][l, h].item():>18.2f}")

json.dump(dict(base_kept=b["kept"].tolist(), tuned_kept=a["kept"].tolist(),
               loss=[before["loss"], after["loss"]], acc=[before["acc"], after["acc"]]),
          open(f"experiments/finetune_{DATA}_results.json", "w"))
print("\nDone.")