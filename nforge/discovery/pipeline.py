from nforge.patching.ablation import per_example
from nforge.discovery.attribution import attribution_scores
from nforge.discovery.necessity import necessity_scan
from nforge.stability.compare import top_k, necessary_heads


def discover(model, rank_batches, eval_batches, k_attr=5, threshold=0.9):
    """Attribution shortlist on rank_batches, then a single-head necessity scan on separate eval_batches."""
    c = per_example(model, rank_batches, "clean", [], "clean_cache").mean().item()
    x = per_example(model, rank_batches, "corr", [], "corr_cache").mean().item()
    attr = attribution_scores(model, rank_batches, c, x)
    kept = necessity_scan(model, eval_batches)
    shortlist = set(top_k(attr, k_attr))
    necessary = necessary_heads(kept, threshold)
    return dict(attr=attr, kept=kept, shortlist=shortlist, necessary=necessary,
                missed_by_attribution=necessary - shortlist)