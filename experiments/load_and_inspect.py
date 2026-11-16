from transformer_lens import HookedTransformer

# Load GPT-2 small (124M parameters, 12 layers, 12 heads per layer)
model = HookedTransformer.from_pretrained("gpt2")

prompt = "When John and Mary went to the store, John gave a drink to"
tokens = model.to_tokens(prompt)

# Run the model AND save every internal activation
logits, cache = model.run_with_cache(tokens)

# 1. What does the model predict next?
next_token = logits[0, -1].argmax()
print("Prediction:", model.to_string(next_token))

# 2. Look inside
print("Tokens:", model.to_str_tokens(prompt))
print("Residual stream shape:", cache["resid_post", 5].shape)
print("Attention pattern shape:", cache["pattern", 5].shape)