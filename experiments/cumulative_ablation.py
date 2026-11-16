import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

clean = "When John and Mary went to the store, John gave a drink to"
corrupted = "When Alice and Bob went to the store, Carl gave a drink to"
clean_tokens = model.to_tokens(clean)
corrupted_tokens = model.to_tokens(corrupted)
mary = model.to_single_token(" Mary")
john = model.to_single_token(" John")

def logit_diff(logits):
    return (logits[0, -1, mary] - logits[0, -1, john]).item()

with torch.no_grad():
    clean_ld = logit_diff(model(clean_tokens))
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)

def ablate(heads):
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    fwd_hooks = []
    for l, hs in by_layer.items():
        name = f"blocks.{l}.attn.hook_z"
        def hook(z, hook, hs=hs, name=name):
            for h in hs:
                z[:, :, h, :] = corrupted_cache[name][:, :, h, :]
            return z
        fwd_hooks.append((name, hook))
    with torch.no_grad():
        return logit_diff(model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks))

# Candidate positive heads, most important first (from your heatmap)
candidates = [(9, 9), (9, 6), (10, 0), (10, 10), (10, 1), (9, 8), (10, 2), (11, 2), (9, 7), (10, 6)]
candidate_set = set(candidates) | {(10, 7), (11, 10)}
pool = [(l, h) for l in range(12) for h in range(12) if (l, h) not in candidate_set]

print(f"Clean: {clean_ld:.2f}\n")
print(f"{'k':>2} {'circuit heads':>14} {'random (mean ± std, 30 draws)':>34}")
for k in range(1, len(candidates) + 1):
    circuit = ablate(candidates[:k]) / clean_ld
    rand = []
    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(pool), generator=g)[:k]
        rand.append(ablate([pool[i] for i in idx]) / clean_ld)
    r = torch.tensor(rand)
    print(f"{k:>2} {circuit:>13.0%} {r.mean():>20.0%} ± {r.std():.0%}")