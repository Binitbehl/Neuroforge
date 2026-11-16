import torch
from collections import defaultdict

only_z = lambda n: n.endswith("attn.hook_z")


def build_batches(model, pairs):
    """Group (clean, corrupted, correct_id, wrong_id) pairs by token length into batches."""
    dev = model.cfg.device
    groups = defaultdict(list)
    for clean, corr, c, w in pairs:
        ct, xt = model.to_tokens(clean), model.to_tokens(corr)
        groups[ct.shape[1]].append((ct[0], xt[0], c, w))
    batches = []
    for items in groups.values():
        b = dict(
            clean=torch.stack([i[0] for i in items]),
            corr=torch.stack([i[1] for i in items]),
            correct=torch.tensor([i[2] for i in items], device=dev),
            wrong=torch.tensor([i[3] for i in items], device=dev),
        )
        with torch.no_grad():
            _, b["clean_cache"] = model.run_with_cache(b["clean"], names_filter=only_z)
            _, b["corr_cache"] = model.run_with_cache(b["corr"], names_filter=only_z)
        batches.append(b)
    return batches


def logit_diff(logits, b):
    idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[idx, -1, b["correct"]] - logits[idx, -1, b["wrong"]]