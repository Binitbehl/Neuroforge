import torch
from nforge.patching.ablation import per_example


def validate(model, test, circuit, n_random=20, seed=0, n_boot=1000):
    """Faithfulness, necessity, minimality and random baselines for a head circuit."""
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    circuit = list(circuit)
    cset = set(circuit)

    full = per_example(model, test, "clean", [], "clean_cache")
    floor = per_example(model, test, "clean", all_heads, "corr_cache")
    gap = (full.mean() - floor.mean()).item()
    n = len(full)
    idx = torch.randint(0, n, (n_boot, n), generator=torch.Generator().manual_seed(seed))

    def keep_only(keep):
        keep = set(keep)
        return per_example(model, test, "clean", [h for h in all_heads if h not in keep], "corr_cache")

    def norm_ci(vals):
        """0 = everything ablated, 1 = full model. Bootstrap over prompts."""
        est = (vals.mean() - floor.mean()) / (full.mean() - floor.mean())
        boot = (vals[idx].mean(1) - floor[idx].mean(1)) / (full[idx].mean(1) - floor[idx].mean(1))
        return est.item(), boot.quantile(0.025).item(), boot.quantile(0.975).item()

    out = {"full": full.mean().item(), "floor": floor.mean().item()}

    # Faithfulness: keep only the circuit
    faith = keep_only(circuit)
    out["faithfulness"] = norm_ci(faith)

    # Necessity: ablate only the circuit, keep everything else
    out["necessity_kept"] = norm_ci(per_example(model, test, "clean", circuit, "corr_cache"))

    # Minimality: remove one head at a time from the faithful circuit
    out["minimality"] = {}
    for hd in circuit:
        d = keep_only(cset - {hd}) - faith
        boot = d[idx].mean(1) / gap
        out["minimality"][hd] = (d.mean().item() / gap, boot.quantile(0.025).item(), boot.quantile(0.975).item())

    # Random baselines: same-size sets from non-circuit heads
    pool = [h for h in all_heads if h not in cset]
    rand = []
    for s in range(n_random):
        g = torch.Generator().manual_seed(1000 + s)
        pick = [pool[i] for i in torch.randperm(len(pool), generator=g)[:len(circuit)]]
        rand.append(norm_ci(keep_only(pick))[0])
    rt = torch.tensor(rand)
    out["random"] = (rt.mean().item(), rt.std().item(), rt.max().item())
    return out
