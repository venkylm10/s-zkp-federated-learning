# SA-FL Baseline: No ZKP Projection Checks

## Summary

This experiment implements the SA-FL (Secure Aggregation Federated Learning) dual-server
secure aggregation scheme from Chang et al., IEEE TCCN 2026, WITHOUT the secret-shared
ZKP projection checks. The system is subjected to dynamic null-space poisoning attacks
to quantify the vulnerability of SA-FL without client-side compliance verification.

## Hardware Requirements

- GPU: NVIDIA L4 (24 GB VRAM)
- CUDA: 12.8+
- PyTorch: 2.11.0+cu128

## Setup

```bash
pip install -r requirements.txt
```

## Training

### MNIST (primary experiment)

```bash
cd /workspace/output/code
python train.py \
  --dataset mnist \
  --rounds 50 \
  --clients 10 \
  --malicious 2 \
  --local-epochs 2 \
  --batch-size 64 \
  --lr 0.01 \
  --correlation-threshold 0.4 \
  --poison-scale 3.0 \
  --warmup-rounds 5 \
  --seed 42
```

### CIFAR-10 (secondary experiment)

```bash
cd /workspace/output/code
python train.py \
  --dataset cifar10 \
  --rounds 40 \
  --clients 10 \
  --malicious 2 \
  --local-epochs 2 \
  --batch-size 64 \
  --lr 0.01 \
  --correlation-threshold 0.4 \
  --poison-scale 3.0 \
  --warmup-rounds 5 \
  --seed 42
```

## Evaluation

```bash
cd /workspace/output/code
python eval.py --dataset mnist --weights weights/best.pt
```

## Dataset

- MNIST: Downloaded from torchvision (cached at `/workspace/datasets/`)
- CIFAR-10: Downloaded from torchvision (cached at `/workspace/datasets/`)

## Random Seed

Seed: 42 (passed to both PyTorch and NumPy for reproducibility)

## Experiment Design

- **Defense**: SA-FL Pearson correlation coefficient threshold filtering (dual-server)
- **Attack**: Dynamic null-space poisoning (activates after warmup_rounds)
- **Missing**: ZKP projection verification (this is what makes null-space attacks viable)
- **Expected**: ~0% detection rate due to absence of projection checks

## Results

The null-space attack exploits the fact that without ZKP-based proof-of-correct-computation,
clients can submit arbitrary gradient updates. The attacker crafts updates whose poison
component is orthogonal to the span of honest gradients, preserving Pearson correlations
and evading the SA-FL defense.
