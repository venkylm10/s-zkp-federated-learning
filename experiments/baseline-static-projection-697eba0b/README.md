# Baseline: Static Projection vs. Null-Space Poisoning

## Summary

Evaluates the vulnerability of static random projection (d=128) Byzantine-robust
aggregation against a defense-aware adversary that constructs poison updates lying
entirely in the null space of the projection matrix.

## Hardware requirements

- GPU: NVIDIA L4 (or equivalent), ~16 GB VRAM
- CUDA 12.x

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
cd /workspace/output/code
python train.py 2>&1 | tee ../train.log
```

## Evaluate (load saved weights)

```bash
python eval.py --dataset mnist
python eval.py --dataset cifar10
```

## Random seed

Seed: 42 (fixed throughout)

## Dataset

MNIST: `ylecun/mnist` from HuggingFace  
CIFAR-10: `uoft-cs/cifar10` from HuggingFace  
Cache directory: `/workspace/datasets/hf`

## Key result

Since the projection matrix P is static and public, the attacker precomputes
the null-space projector Pi_null = I - P^T (P P^T)^{-1} P and adds a scaled
null-space component to every honest gradient. The server's projection filter
sees P @ g_poison = P @ g_honest (the null component vanishes), so the detection
rate converges to 0%. The null-space component still affects the model in full
parameter space, degrading accuracy.
