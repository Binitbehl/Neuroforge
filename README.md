# NEUROFORGE

Reverse-engineering neural intelligence: discover circuits in Transformers
and test them causally with activation patching and ablation.

## Status
Work in progress. Currently reproducing the IOI circuit in GPT-2 small.

## Results so far
- Activation patching over all 144 attention heads (see `experiments/head_patching_heatmap.png`)
- Single-head ablation: weak effects, consistent with redundant backup heads
- Cumulative ablation: circuit-head removal vs. random baselines

## Setup
    python -m venv neuroforge-env
    neuroforge-env\Scripts\activate
    pip install -r requirements.txt

## Run
    python experiments/activation_patching.py