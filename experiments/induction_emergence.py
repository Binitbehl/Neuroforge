import gc
import torch
from transformer_lens import HookedTransformer

MODEL = "pythia-70m"
STEPS = [1000, 2000, 4000, 8000, 16000, 32000, 143000]
SEQ_LEN, BATCH = 50, 64

print(f"{'step':>7} {'loss 1st copy':>14} {'loss 2nd copy':>14} {'induction score':>16}")
for step in STEPS:
    model = HookedTransformer.from_pretrained(MODEL, checkpoint_value=step)
    model.eval()
    bos = model.tokenizer.bos_token_id
    g = torch.Generator().manual_seed(0)
    rand = torch.randint(1000, 30000, (BATCH, SEQ_LEN), generator=g)
    tokens = torch.cat([torch.full((BATCH, 1), bos), rand, rand], dim=1).to(model.cfg.device)

    with torch.no_grad():
        logits = model(tokens)
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    loss = -logp.gather(-1, tokens[:, 1:, None]).squeeze(-1)    # loss[i] = predicting token i+1

    first = loss[:, 1:SEQ_LEN].mean().item()                    # predicting the first copy
    second = loss[:, SEQ_LEN + 1:2 * SEQ_LEN].mean().item()     # predicting the repeat
    print(f"{step:>7} {first:>14.2f} {second:>14.2f} {first - second:>16.2f}", flush=True)
    del model
    gc.collect()
