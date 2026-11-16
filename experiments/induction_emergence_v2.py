import gc
import torch
from transformer_lens import HookedTransformer

MODEL = "pythia-70m"
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 16000, 143000]
SEQ_LEN, BATCH = 50, 64

print(f"{'step':>7} {'loss 1st':>9} {'loss 2nd':>9} {'acc 1st':>8} {'acc 2nd':>8}")
for step in STEPS:
    try:
        model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    except Exception as e:
        print(f"{step:>7} could not load: {str(e)[:120]}", flush=True)
        continue
    model.eval()
    bos = model.tokenizer.bos_token_id
    g = torch.Generator().manual_seed(0)
    rand = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    tokens = torch.cat([torch.full((BATCH, 1), bos), rand, rand], dim=1).to(model.cfg.device)

    with torch.no_grad():
        logits = model(tokens)[:, :-1].float()
    target = tokens[:, 1:]
    loss = -torch.log_softmax(logits, -1).gather(-1, target[..., None]).squeeze(-1)
    correct = (logits.argmax(-1) == target).float()

    s1, s2 = slice(1, SEQ_LEN), slice(SEQ_LEN + 1, 2 * SEQ_LEN)   # skip positions that can't be predicted
    print(f"{step:>7} {loss[:, s1].mean():>9.2f} {loss[:, s2].mean():>9.2f} "
          f"{correct[:, s1].mean():>8.0%} {correct[:, s2].mean():>8.0%}", flush=True)
    del model
    gc.collect()