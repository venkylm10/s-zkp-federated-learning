# S-ZKP Federated Learning Framework

Implements Stochastic Zero-Knowledge Proof (S-ZKP) defense against defense-aware
null-space poisoning attacks in federated learning.

## Ablation Study: Colluding Client Fraction

`train_collusion_ablation.py` tests robustness under 4 Byzantine fractions (10%–40%)
and 3 coordinated attack types (null-space, sign-flip, label-flip) on MNIST.

### Collusion Ablation Train
```bash
cd /workspace/output/code
python train_collusion_ablation.py
```
Outputs: `/workspace/output/ablation_collusion_logs/*.json`,
`/workspace/output/figures/accuracy_vs_byzantine_fraction.png`, `results.json`.

## Ablation Study: Seed Rotation Frequency

`train_ablation.py` ablates four key rotation frequencies (1, 5, 10, static) on MNIST
with an adaptive adversary that reconstructs R_t from accumulated observations.

### Ablation Train
```bash
cd /workspace/output/code
python train_ablation.py
```
Outputs: `/workspace/output/ablation_rotation_logs/*.json`,
`/workspace/output/figures/detection_rate_vs_rotation_freq.png`, `results.json`.

## Hardware
- GPU: NVIDIA L4 (CUDA 12.8)
- VRAM: 24 GB

## Install
```bash
pip install -r requirements.txt
```

## Train
```bash
cd /workspace/output/code
python train.py
```
Runs S-ZKP + baselines on MNIST and CIFAR-10. Saves weights, figures, and results.

## Evaluate saved model
```bash
python eval.py
```

## Random seed: 42
## Datasets: ylecun/mnist, uoft-cs/cifar10 (HuggingFace)

## Key parameters
- projection_dim: 128
- n_clients: 10 (8 honest, 2 Byzantine)
- defense: S-ZKP with additive secret sharing + dynamic rotation
- attack: null-space poisoning (using stale R_{t-1})
