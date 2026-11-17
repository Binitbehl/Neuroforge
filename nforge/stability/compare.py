import torch


def top_k(scores, k, largest=True):
    """Return the (layer, head) pairs with the k highest (or lowest) scores."""
    s = torch.as_tensor(scores)
    n_h = s.shape[1]
    idx = s.flatten().argsort(descending=largest)[:k]
    return [(i.item() // n_h, i.item() % n_h) for i in idx]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def rank_corr(a, b):
    """Spearman-style rank correlation between two score grids (ties broken arbitrarily)."""
    a = torch.as_tensor(a).flatten()
    b = torch.as_tensor(b).flatten()
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def necessary_heads(kept, threshold=0.9):
    """Heads whose single ablation leaves less than `threshold` of the behavior."""
    s = torch.as_tensor(kept)
    return {(l, h) for l in range(s.shape[0]) for h in range(s.shape[1]) if s[l, h] < threshold}