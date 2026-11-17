import json
from nforge.stability.compare import top_k, jaccard, rank_corr, necessary_heads

patch = json.load(open("experiments/induction_scores.json"))          # seed-0 patching scores
kept = json.load(open("experiments/induction_necessity_scan.json"))   # seed-1 single-head ablation
steps = sorted(patch, key=int)
final = steps[-1]
K = 5

print("Heads whose removal alone leaves < 90% of the behavior:")
for s in steps:
    print(f"  step {s:>6}: {sorted(necessary_heads(kept[s]))}")

print(f"\nSimilarity to step {final}")
print(f"{'step':>7} {'patch top-5 Jaccard':>20} {'patch rank corr':>16} "
      f"{'necessary-set Jaccard':>22} {'necessity rank corr':>20}")
for s in steps[:-1]:
    print(f"{s:>7} {jaccard(top_k(patch[s], K), top_k(patch[final], K)):>20.2f} "
          f"{rank_corr(patch[s], patch[final]):>16.2f} "
          f"{jaccard(necessary_heads(kept[s]), necessary_heads(kept[final])):>22.2f} "
          f"{rank_corr(kept[s], kept[final]):>20.2f}")

print("\nAgreement between the two measures within each step "
      "(rank corr of patching score vs 1 - kept):")
for s in steps:
    inv = [[1 - v for v in row] for row in kept[s]]
    print(f"  step {s:>6}: {rank_corr(patch[s], inv):.2f}")