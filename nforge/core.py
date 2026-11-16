# Compatibility shim: re-exports from the new locations.
from nforge.activations.batches import build_batches, logit_diff, only_z   # noqa: F401
from nforge.patching.ablation import run_with_overwrite, per_example       # noqa: F401
from nforge.discovery.attribution import attribution_scores                # noqa: F401