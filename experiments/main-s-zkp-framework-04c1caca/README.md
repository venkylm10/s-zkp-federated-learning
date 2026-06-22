# S-ZKP Federated Learning Framework

Implements Stochastic Zero-Knowledge Proof (S-ZKP) defense against defense-aware
null-space poisoning attacks in federated learning.

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
