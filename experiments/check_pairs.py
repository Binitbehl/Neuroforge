from transformer_lens import HookedTransformer
from nforge.data.ioi import make_pairs

model = HookedTransformer.from_pretrained("gpt2")
pairs = make_pairs(model, n=5)
for clean, corrupted, c, w in pairs:
    print(clean)
    print(corrupted)
    print(model.to_string(c), "vs", model.to_string(w), "\n")
