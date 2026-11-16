import os
from transformer_lens import HookedTransformer


def load_model(name="gpt2", step=None):
    """Load a model, optionally at a training checkpoint (Pythia)."""
    # Workaround: with no HF_TOKEN set, downloads failed with 'Illegal header value b"Bearer "'.
    # The cause is unconfirmed; a non-empty placeholder avoided it.
    os.environ.setdefault("HF_TOKEN", "placeholder")
    kwargs = {} if step is None else {"checkpoint_value": step}
    model = HookedTransformer.from_pretrained(name, **kwargs)
    model.eval()
    return model