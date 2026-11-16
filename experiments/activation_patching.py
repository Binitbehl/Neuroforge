import torch
from transformer_lens import HookedTransformer, utils

model = HookedTransformer.from_pretrained("gpt2")
model.eval()

clean = "When John and Mary went to the store, John gave a drink to"
corrupted = "When Alice and Bob went to the store, Carl gave a drink to"

clean_tokens = model.to_tokens(clean)
corrupted_tokens = model.to_tokens(corrupted)
assert clean_tokens.shape == corrupted_tokens.shape, "Prompts must have the same token length"

mary = model.to_single_token(" Mary")
john = model.to_single_token(" John")

def logit_diff(logits):
    return (logits[0, -1, mary] - logits[0, -1, john]).item()

# Run both prompts, caching the clean run's internals
with torch.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

clean_ld = logit_diff(clean_logits)
corrupted_ld = logit_diff(corrupted_logits)
print(f"Clean: {clean_ld:.2f} | Corrupted: {corrupted_ld:.2f}")

def make_patch_hook(layer, head):
    name = utils.get_act_name("z", layer)
    def hook(z, hook):
        z[:, :, head, :] = clean_cache[name][:, :, head, :]
        return z
    return name, hook

# Patch every attention head, one at a time
results = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        name, hook = make_patch_hook(layer, head)
        with torch.no_grad():
            patched_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=[(name, hook)])
        # 0 = no effect, 1 = fully restored the clean behavior
        results[layer, head] = (logit_diff(patched_logits) - corrupted_ld) / (clean_ld - corrupted_ld)

print("\nTop 10 heads by recovery:")
flat = results.flatten()
for idx in flat.abs().argsort(descending=True)[:10]:
    l, h = divmod(idx.item(), model.cfg.n_heads)
    print(f"  Layer {l}, Head {h}: {results[l, h].item():+.3f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.imshow(results.numpy(), cmap="RdBu", vmin=-0.6, vmax=0.6)
plt.colorbar(label="Fraction of clean behavior restored")
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Activation patching: GPT-2 small, IOI")
plt.xticks(range(12))
plt.yticks(range(12))
plt.savefig("experiments/head_patching_heatmap.png", dpi=150)
plt.show()