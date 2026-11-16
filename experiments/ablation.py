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
    clean_logits = model(clean_tokens)
    _, corrupted_cache = model.run_with_cache(corrupted_tokens)
clean_ld = logit_diff(clean_logits)

def ablate(heads, mode):
    """Run the CLEAN prompt with the given (layer, head) pairs knocked out."""
    by_layer = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)

    fwd_hooks = []
    for l, hs in by_layer.items():
        name = f"blocks.{l}.attn.hook_z"
        def hook(z, hook, hs=hs, name=name):
            for h in hs:
                if mode == "zero":
                    z[:, :, h, :] = 0
                else:  # "resample": use the activation from the corrupted run
                    z[:, :, h, :] = corrupted_cache[name][:, :, h, :]
            return z
        fwd_hooks.append((name, hook))

    with torch.no_grad():
        return logit_diff(model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks))

name_movers = [(9, 9), (9, 6), (10, 0)]
negative_movers = [(10, 7), (11, 10)]

# Control: random early/mid heads. If these hurt as much, our result means nothing.
torch.manual_seed(0)
random_heads = [(int(torch.randint(0, 9, (1,))), int(torch.randint(0, 12, (1,)))) for _ in range(3)]

print(f"Clean logit diff (no ablation): {clean_ld:.2f}\n")
for label, heads in [("Name movers", name_movers),
                     ("Negative movers", negative_movers),
                     ("Random control", random_heads)]:
    for mode in ["zero", "resample"]:
        ld = ablate(heads, mode)
        print(f"{label:16s} {str(heads):32s} {mode:9s} -> {ld:5.2f}  ({ld / clean_ld:.0%} of clean kept)")
