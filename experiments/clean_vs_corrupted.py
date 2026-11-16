from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")

clean = "When John and Mary went to the store, John gave a drink to"
corrupted = "When Alice and Bob went to the store, Carl gave a drink to"

mary = model.to_single_token(" Mary")
john = model.to_single_token(" John")

def logit_diff(prompt):
    logits = model(model.to_tokens(prompt))
    return (logits[0, -1, mary] - logits[0, -1, john]).item()

print("Clean logit diff (Mary - John):    ", logit_diff(clean))
print("Corrupted logit diff (Mary - John):", logit_diff(corrupted))