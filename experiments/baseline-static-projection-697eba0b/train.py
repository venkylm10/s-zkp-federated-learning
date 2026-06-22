"""
Federated Learning with Static Projection Defense vs. Null-Space Poisoning Attack

Baseline experiment: static random projection d=128 is trivially bypassed by
an adversary who projects their poison update into the null space of P.
"""

import os
import sys
import json
import time
import math
import copy
import argparse
import logging
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import wandb

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJ_DIM = 128          # static projection dimension d
N_CLIENTS = 10          # total clients
N_BYZANTINE = 2         # malicious clients
ROUNDS = 60             # communication rounds
LOCAL_EPOCHS = 2        # local training epochs per round
BATCH_SIZE = 64
LR = 0.01
SEED = 42

DATASETS = ["mnist", "cifar10"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "/workspace/output"
CODE_DIR = f"{OUTPUT_DIR}/code"
LOGS_DIR = f"{OUTPUT_DIR}/baseline_static_logs"
FIG_DIR = f"{OUTPUT_DIR}/figures"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{OUTPUT_DIR}/train.log", mode="a"),
        logging.FileHandler(f"{LOGS_DIR}/baseline_static.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class MNISTModel(nn.Module):
    """Simple MLP for MNIST."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CIFAR10Model(nn.Module):
    """Small CNN for CIFAR-10."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 32x32 → 16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),           # 16x16 → 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),   # → 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(dataset_name: str) -> nn.Module:
    if dataset_name == "mnist":
        return MNISTModel().to(DEVICE)
    return CIFAR10Model().to(DEVICE)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flat_grad(model: nn.Module, loss: torch.Tensor) -> torch.Tensor:
    """Compute and return flattened gradient vector."""
    loss.backward()
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1).clone())
    return torch.cat(grads)


def set_flat_params(model: nn.Module, flat: torch.Tensor):
    """Write a flat parameter vector back into model."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].view_as(p))
        offset += numel


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


# ──────────────────────────────────────────────────────────────────────────────
# Static Projection
# ──────────────────────────────────────────────────────────────────────────────

def make_projection_matrix(d: int, D: int, seed: int = 0) -> torch.Tensor:
    """Create static random projection matrix P ∈ R^{d × D} (normalized rows)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    P = torch.randn(d, D, generator=rng)   # create on CPU
    P = P / P.norm(dim=1, keepdim=True)    # normalize
    return P.to(DEVICE)                     # move to GPU


def project(P: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project vector v ∈ R^D onto R^d using P."""
    return P @ v   # (d,)


# ──────────────────────────────────────────────────────────────────────────────
# Null-Space Attack
# ──────────────────────────────────────────────────────────────────────────────

class NullSpaceAttacker:
    """
    Defense-aware attacker that adds a perturbation lying entirely in null(P).

    Since P @ null_component = 0, the poisoned gradient looks identical to
    the honest gradient under the projection filter.

    The null-space projector is:
        Pi_null(v) = v - P^T (P P^T)^{-1} P v
    """

    def __init__(self, P: torch.Tensor, poison_scale: float = 5.0):
        self.P = P                    # (d, D)
        self.poison_scale = poison_scale
        # Precompute (P P^T)^{-1} — small d×d matrix
        PPT = P @ P.T                 # (d, d)
        self.PPT_inv = torch.linalg.inv(PPT)  # (d, d)

    def null_project(self, v: torch.Tensor) -> torch.Tensor:
        """Project v onto null(P): remove its component in range(P^T)."""
        # v_range = P^T (P P^T)^{-1} P v
        Pv = self.P @ v                           # (d,)
        coeff = self.PPT_inv @ Pv                 # (d,)
        v_range = self.P.T @ coeff                # (D,)
        return v - v_range                        # component in null(P)

    def poison(self, honest_grad: torch.Tensor) -> torch.Tensor:
        """
        Craft poisoned gradient:
            g_poison = honest_grad + scale * null_space_direction

        The null_space_direction is chosen to maximize weight disruption
        while being invisible to the projection filter.
        """
        # Use the honest gradient itself projected to null space as the
        # attack direction (inverts it to maximally disrupt learning)
        null_direction = self.null_project(-2.0 * honest_grad)
        null_norm = null_direction.norm()
        if null_norm > 1e-8:
            null_direction = null_direction / null_norm * honest_grad.norm()
        return honest_grad + self.poison_scale * null_direction

    def verify_null_space(self, poisoned_grad: torch.Tensor, honest_grad: torch.Tensor) -> dict:
        """Verify the attack is in null space (sanity check)."""
        p_poison = self.P @ poisoned_grad
        p_honest = self.P @ honest_grad
        diff = (p_poison - p_honest).norm().item()
        return {"proj_diff": diff, "is_in_null_space": diff < 1e-3}


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str):
    """Load dataset as tensors. Returns (train_X, train_y, test_X, test_y)."""
    from datasets import load_dataset as hf_load
    hf_cache = "/workspace/datasets/hf"
    os.makedirs(hf_cache, exist_ok=True)

    if name == "mnist":
        hf_name = "ylecun/mnist"
        log.info(f"Loading {hf_name} from HF ...")
        ds = hf_load(hf_name, cache_dir=hf_cache)
        def proc(split):
            imgs = torch.tensor(
                np.array([np.array(x["image"]) for x in split]), dtype=torch.float32
            ).unsqueeze(1) / 255.0   # (N,1,28,28)
            labels = torch.tensor([x["label"] for x in split], dtype=torch.long)
            return imgs, labels
        train_X, train_y = proc(ds["train"])
        test_X, test_y = proc(ds["test"])

    elif name == "cifar10":
        hf_name = "uoft-cs/cifar10"
        log.info(f"Loading {hf_name} from HF ...")
        ds = hf_load(hf_name, cache_dir=hf_cache)
        def proc(split):
            imgs = torch.tensor(
                np.array([np.array(x["img"]) for x in split]), dtype=torch.float32
            ).permute(0, 3, 1, 2) / 255.0   # (N,3,32,32)
            # Normalize
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            imgs = (imgs - mean) / std
            labels = torch.tensor([x["label"] for x in split], dtype=torch.long)
            return imgs, labels
        train_X, train_y = proc(ds["train"])
        test_X, test_y = proc(ds["test"])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    log.info(f"  Train: {train_X.shape}, Test: {test_X.shape}")
    return train_X, train_y, test_X, test_y


def split_iid(train_X, train_y, n_clients, seed=42):
    """Split training data IID across clients."""
    rng = np.random.default_rng(seed)
    n = len(train_X)
    indices = rng.permutation(n)
    chunks = np.array_split(indices, n_clients)
    return [
        TensorDataset(train_X[torch.from_numpy(idx)], train_y[torch.from_numpy(idx)])
        for idx in chunks
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Client Local Training
# ──────────────────────────────────────────────────────────────────────────────

def client_update(global_params: torch.Tensor, dataset, local_epochs: int, lr: float, D: int):
    """
    Train a local model for local_epochs, return gradient (Δ = new_params - old_params).
    """
    model = get_model_by_D(D)
    set_flat_params(model, global_params.clone())
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(local_epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    new_params = get_flat_params(model)
    delta = new_params - global_params  # pseudo-gradient (parameter update)
    return delta


# Global registry for model constructors (set at runtime)
_dataset_name: str = None

def get_model_by_D(D: int) -> nn.Module:
    return get_model(_dataset_name)


# ──────────────────────────────────────────────────────────────────────────────
# Server Projection-Based Filter
# ──────────────────────────────────────────────────────────────────────────────

def projection_filter(deltas: list, P: torch.Tensor, threshold_sigma: float = 2.5):
    """
    Projection-based anomaly filter.

    Project each gradient update into R^d, compute projected norms,
    flag as poisoned if norm deviates >threshold_sigma from the median.

    Returns (accepted_indices, rejected_indices).
    """
    proj_norms = torch.stack([project(P, d).norm() for d in deltas])  # (N,)
    median = proj_norms.median()
    mad = (proj_norms - median).abs().median()  # median absolute deviation
    # Robust z-score
    robust_z = (proj_norms - median) / (mad + 1e-8)
    accepted = [i for i, z in enumerate(robust_z) if z.abs() <= threshold_sigma]
    rejected = [i for i, z in enumerate(robust_z) if z.abs() > threshold_sigma]
    return accepted, rejected


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, test_X: torch.Tensor, test_y: torch.Tensor) -> dict:
    model.eval()
    loader = DataLoader(TensorDataset(test_X, test_y), batch_size=256)
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * len(y)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += len(y)
    return {"accuracy": 100.0 * correct / total, "loss": total_loss / total}


# ──────────────────────────────────────────────────────────────────────────────
# Main Federated Learning Loop
# ──────────────────────────────────────────────────────────────────────────────

def run_fl(dataset_name: str, rounds: int, wandb_run, run_results: dict):
    """Run federated learning experiment for one dataset."""
    global _dataset_name
    _dataset_name = dataset_name

    log.info(f"\n{'='*60}")
    log.info(f"DATASET: {dataset_name.upper()}")
    log.info(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    train_X, train_y, test_X, test_y = load_dataset(dataset_name)
    client_datasets = split_iid(train_X, train_y, N_CLIENTS, seed=SEED)

    # Initialize global model
    global_model = get_model(dataset_name)
    D = count_params(global_model)
    log.info(f"Model params D = {D}")

    # Static projection matrix
    P = make_projection_matrix(PROJ_DIM, D, seed=SEED)
    log.info(f"Projection matrix P: {P.shape}")

    # Attacker
    attacker = NullSpaceAttacker(P, poison_scale=5.0)

    # Byzantine client indices (last N_BYZANTINE clients are malicious)
    byzantine_ids = list(range(N_CLIENTS - N_BYZANTINE, N_CLIENTS))
    honest_ids = list(range(N_CLIENTS - N_BYZANTINE))

    log.info(f"Byzantine clients: {byzantine_ids}")

    # Training history
    history = {
        "round": [],
        "test_accuracy": [],
        "test_loss": [],
        "detection_rate": [],
        "n_detected": [],
        "proj_norm_honest": [],
        "proj_norm_poison": [],
    }

    best_acc = 0.0

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        global_params = get_flat_params(global_model)

        # ── Client local updates ──
        deltas = []
        for cid in range(N_CLIENTS):
            delta = client_update(
                global_params, client_datasets[cid], LOCAL_EPOCHS, LR, D
            )
            if cid in byzantine_ids:
                # Null-space poisoning
                delta = attacker.poison(delta)
            deltas.append(delta)

        # Sanity-check attack on first round
        if rnd == 1:
            honest_delta = deltas[0]
            poison_delta = deltas[byzantine_ids[0]]
            verify = attacker.verify_null_space(poison_delta, honest_delta)
            log.info(f"[Round 1] Null-space verify: proj_diff={verify['proj_diff']:.2e}, "
                     f"in_null_space={verify['is_in_null_space']}")

        # ── Server: projection-based filter ──
        accepted_ids, rejected_ids = projection_filter(deltas, P, threshold_sigma=2.5)

        # Detection rate: fraction of Byzantine clients detected (rejected)
        n_detected = sum(1 for i in rejected_ids if i in byzantine_ids)
        detection_rate = 100.0 * n_detected / N_BYZANTINE

        # Projected norms for logging
        pn_honest = np.mean([project(P, deltas[i]).norm().item() for i in honest_ids])
        pn_poison = np.mean([project(P, deltas[i]).norm().item() for i in byzantine_ids])

        # ── FedAvg aggregation (using accepted deltas, or all if none accepted) ──
        use_ids = accepted_ids if accepted_ids else list(range(N_CLIENTS))
        accepted_deltas = [deltas[i] for i in use_ids]
        avg_delta = torch.stack(accepted_deltas).mean(0)

        # Update global model
        new_params = global_params + avg_delta
        set_flat_params(global_model, new_params)

        # ── Evaluate ──
        metrics = evaluate(global_model, test_X, test_y)
        acc = metrics["accuracy"]
        loss = metrics["loss"]
        elapsed = time.time() - t0

        if acc > best_acc:
            best_acc = acc
            torch.save(global_model.state_dict(), f"{CODE_DIR}/weights/best_{dataset_name}.pt")

        torch.save(global_model.state_dict(), f"{CODE_DIR}/weights/last_{dataset_name}.pt")

        # Log
        history["round"].append(rnd)
        history["test_accuracy"].append(acc)
        history["test_loss"].append(loss)
        history["detection_rate"].append(detection_rate)
        history["n_detected"].append(n_detected)
        history["proj_norm_honest"].append(pn_honest)
        history["proj_norm_poison"].append(pn_poison)

        log.info(
            f"[{dataset_name}] Round {rnd:3d}/{rounds} | "
            f"acc={acc:.2f}% | loss={loss:.4f} | "
            f"detected={n_detected}/{N_BYZANTINE} ({detection_rate:.0f}%) | "
            f"pn_honest={pn_honest:.3f} pn_poison={pn_poison:.3f} | "
            f"t={elapsed:.1f}s"
        )

        # W&B logging
        if wandb_run is not None:
            wandb.log({
                f"{dataset_name}/test_accuracy": acc,
                f"{dataset_name}/test_loss": loss,
                f"{dataset_name}/poison_detection_rate": detection_rate,
                f"{dataset_name}/proj_norm_honest": pn_honest,
                f"{dataset_name}/proj_norm_poison": pn_poison,
                "round": rnd,
            }, step=rnd)

        # Progress file
        with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
            json.dump({
                "phase": "training",
                "current": rnd,
                "total": rounds * len(DATASETS),
            }, f)

    # Save per-dataset history
    with open(f"{LOGS_DIR}/{dataset_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_acc = history["test_accuracy"][-1]
    avg_detection = np.mean(history["detection_rate"])
    log.info(f"\n[{dataset_name}] FINAL: acc={final_acc:.2f}%, "
             f"avg_detection={avg_detection:.2f}%, best_acc={best_acc:.2f}%")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_acc,
        "avg_detection_rate": avg_detection,
        "history": history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figure generation
# ──────────────────────────────────────────────────────────────────────────────

def plot_accuracy_under_attack(mnist_hist: dict, cifar_hist: dict):
    """Plot test accuracy over rounds for both datasets under attack."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, hist, name, color in zip(
        axes,
        [mnist_hist, cifar_hist],
        ["MNIST", "CIFAR-10"],
        ["steelblue", "darkorange"],
    ):
        rounds = hist["round"]
        acc = hist["test_accuracy"]
        det = hist["detection_rate"]

        ax.plot(rounds, acc, color=color, linewidth=2, label="Test Accuracy")
        ax.fill_between(rounds, acc, alpha=0.2, color=color)
        ax2 = ax.twinx()
        ax2.plot(rounds, det, color="crimson", linewidth=1.5, linestyle="--",
                 alpha=0.7, label="Detection Rate")
        ax2.set_ylim(-5, 105)
        ax2.set_ylabel("Detection Rate (%)", color="crimson")
        ax2.tick_params(axis="y", colors="crimson")

        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy (%)", color=color)
        ax.tick_params(axis="y", colors=color)
        ax.set_title(f"{name}: Null-Space Poisoning vs Static Projection")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)

    plt.tight_layout()
    out_path = f"{FIG_DIR}/accuracy_under_attack.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved figure: {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Init W&B
    wandb_run = None
    wandb_run_url = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb_run = wandb.init(
                project="prof-f9e1ad4b-plan-14e30fb6",
                name="baseline_static_projection",
                config={
                    "proj_dim": PROJ_DIM,
                    "n_clients": N_CLIENTS,
                    "n_byzantine": N_BYZANTINE,
                    "rounds": ROUNDS,
                    "local_epochs": LOCAL_EPOCHS,
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                    "attack": "null_space",
                    "defense": "static_projection",
                },
            )
            wandb_run_url = wandb_run.url
            log.info(f"WANDB_RUN_URL: {wandb_run_url}")
        except Exception as e:
            log.warning(f"W&B init failed: {e}")

    all_results = {}

    for ds in DATASETS:
        result = run_fl(ds, ROUNDS, wandb_run, all_results)
        all_results[ds] = result

    # ── Aggregate metrics ──
    # Average test accuracy across both datasets
    test_acc_mnist = all_results["mnist"]["final_accuracy"]
    test_acc_cifar = all_results["cifar10"]["final_accuracy"]
    avg_test_acc = (test_acc_mnist + test_acc_cifar) / 2.0

    # Average detection rate across both datasets
    det_mnist = all_results["mnist"]["avg_detection_rate"]
    det_cifar = all_results["cifar10"]["avg_detection_rate"]
    avg_detection = (det_mnist + det_cifar) / 2.0

    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY")
    log.info(f"  MNIST: acc={test_acc_mnist:.2f}%, detection={det_mnist:.2f}%")
    log.info(f"  CIFAR-10: acc={test_acc_cifar:.2f}%, detection={det_cifar:.2f}%")
    log.info(f"  Average acc={avg_test_acc:.2f}%, detection={avg_detection:.2f}%")
    log.info(f"{'='*60}")

    # ── Figures ──
    with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
        json.dump({"phase": "generating_figures", "current": 0, "total": 1}, f)

    plot_accuracy_under_attack(
        all_results["mnist"]["history"],
        all_results["cifar10"]["history"],
    )

    # ── W&B final logging ──
    if wandb_run is not None:
        wandb.log({
            "test_accuracy": avg_test_acc,
            "poison_detection_rate": avg_detection,
            "mnist_test_accuracy": test_acc_mnist,
            "cifar10_test_accuracy": test_acc_cifar,
        })
        wandb.finish()

    # ── Build inline_data for figures ──
    mnist_hist = all_results["mnist"]["history"]
    cifar_hist = all_results["cifar10"]["history"]
    inline_data_acc = {
        "mnist_accuracy": mnist_hist["test_accuracy"],
        "cifar10_accuracy": cifar_hist["test_accuracy"],
        "rounds": mnist_hist["round"],
    }

    # ── Write results.json ──
    with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
        json.dump({"phase": "writing_results", "current": 0, "total": 1}, f)

    results = {
        # Schema required fields
        "test_accuracy": avg_test_acc,
        "poison_detection_rate": avg_detection,
        "wandb_run_url": wandb_run_url,
        # Enriched manifest
        "manifest_version": 1,
        "config": {
            "proj_dim": PROJ_DIM,
            "n_clients": N_CLIENTS,
            "n_byzantine": N_BYZANTINE,
            "rounds": ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "attack": "null_space",
            "defense": "static_projection",
        },
        "results": [
            {
                "name": "test_accuracy",
                "value": avg_test_acc,
                "unit": "%",
                "provenance": "measured",
                "method": "fedavg_static_proj_null_attack",
                "formula": "mean(mnist_test_acc, cifar10_test_acc) at final round",
            },
            {
                "name": "mnist_test_accuracy",
                "value": test_acc_mnist,
                "unit": "%",
                "provenance": "measured",
                "method": "fedavg_static_proj_null_attack",
            },
            {
                "name": "cifar10_test_accuracy",
                "value": test_acc_cifar,
                "unit": "%",
                "provenance": "measured",
                "method": "fedavg_static_proj_null_attack",
            },
            {
                "name": "poison_detection_rate",
                "value": avg_detection,
                "unit": "%",
                "provenance": "measured",
                "method": "static_projection_filter",
                "formula": "mean(mnist_det_rate, cifar10_det_rate) averaged over rounds",
            },
            {
                "name": "mnist_detection_rate",
                "value": det_mnist,
                "unit": "%",
                "provenance": "measured",
                "method": "static_projection_filter",
            },
            {
                "name": "cifar10_detection_rate",
                "value": det_cifar,
                "unit": "%",
                "provenance": "measured",
                "method": "static_projection_filter",
            },
        ],
        "baselines": [
            {
                "name": "no_defense_fedavg",
                "provenance": "claimed_unverified",
                "description": "Standard FedAvg without any Byzantine defense",
                "headline": False,
            }
        ],
        "figures": [
            {
                "name": "accuracy_under_attack",
                "renders": ["test_accuracy", "mnist_test_accuracy", "cifar10_test_accuracy"],
                "inline_data": inline_data_acc,
            }
        ],
        "metrics": {
            "test_accuracy": avg_test_acc,
            "poison_detection_rate": avg_detection,
            "mnist_test_accuracy": test_acc_mnist,
            "cifar10_test_accuracy": test_acc_cifar,
            "mnist_detection_rate": det_mnist,
            "cifar10_detection_rate": det_cifar,
        },
        "validation_status": "pending",
        "github_commit_sha": None,
    }

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Wrote results.json: test_acc={avg_test_acc:.2f}%, detection={avg_detection:.2f}%")

    return results


if __name__ == "__main__":
    main()
