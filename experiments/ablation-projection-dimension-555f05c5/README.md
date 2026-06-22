# S-ZKP Projection Dimension Ablation

## Summary

This experiment ablates the projection dimension `d ∈ {32, 64, 128, 256, 512}` of the
S-ZKP (Secret-shared Zero-Knowledge Proof) federated learning framework. For each d,
the experiment runs 20 FL rounds on MNIST with a defense-aware null-space poisoning
attack and measures the security-complexity trade-off: poison detection rate, ZKP
proving time, and proof size.

## Hardware Requirements

- GPU: NVIDIA L4 (24 GB VRAM)
- CUDA: 12.8+
- PyTorch: 2.12.1

## Setup

```bash
pip install -r requirements.txt
```

## Training (Ablation)

```bash
cd /workspace/output/code
python ablate_projection_dim.py 2>&1 | tee /workspace/output/train.log
```

This runs 5 × 20 = 100 FL rounds in total, one run per d value.

## Post-processing / Results Assembly

```bash
cd /workspace/output/code
python make_ablation_results.py \
  --wandb-url <WANDB_RUN_URL> \
  --github-sha <COMMIT_SHA>
```

## Evaluation

```bash
cd /workspace/output/code
python eval.py --dataset mnist --weights weights/best.pt
```

Note: weights/best.pt may not be saved (no training checkpoints in ablation mode);
the ablation script focuses on per-round metrics.

## Dataset

- MNIST: Downloaded from torchvision (cached at `/workspace/datasets/`)

## Random Seed

Seed: 42 (passed to both PyTorch and NumPy for reproducibility)

## Experiment Design

- **Framework**: S-ZKP secure aggregation with random projection dimension d
- **Attack**: Defense-aware null-space attack (uses previous round's null space of R)
- **Detection**: Projected Pearson correlation on d-dimensional projected gradients
- **ZKP model**: Schnorr-type proof proving y = R @ g for projection matrix R
- **Proving time**: Actual R @ g computation + d × 0.45 ms Schnorr overhead
- **Proof size**: (2d + 1) × 33 bytes (EC points) + (d + 1) × 32 bytes (scalars)
- **Theoretical detection**: Commitment model — P(detect | d) = 1 − F(d,d).CDF(0.9025)

## Results

Per-d results are saved to `/workspace/output/ablation_d_logs/d{d}_results.json`.
Summary: `/workspace/output/ablation_d_logs/summary.json`.
Figure: `/workspace/output/figures/security_complexity_tradeoff.png`.
