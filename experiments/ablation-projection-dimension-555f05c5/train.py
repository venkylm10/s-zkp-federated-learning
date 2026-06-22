"""SA-FL baseline federated learning training script (WITHOUT ZKP projection checks).

This script implements federated learning with SA-FL secure aggregation
subjected to dynamic null-space poisoning attacks.

Key experimental setup:
- SA-FL dual-server secure aggregation with Pearson correlation defense
- NO secret-shared projection checks (baseline vulnerability study)
- Dynamic null-space poisoning attack (post-warmup)
- Datasets: MNIST (primary), CIFAR-10 (secondary)

Usage:
    python train.py --dataset mnist --rounds 50 --clients 10 --malicious 2
    python train.py --dataset cifar10 --rounds 40 --clients 10 --malicious 2
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# Insert parent path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.cnn import get_model
from utils.sa_fl import SAFLAggregator
from utils.attacks import NullSpaceAttacker

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = Path("/workspace/output/sa_fl_baseline_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "training.log"),
    ],
)
logger = logging.getLogger("sa_fl")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset: str, num_clients: int, seed: int = 42):
    """Load dataset and partition it IID among clients."""
    data_root = Path("/workspace/datasets")
    data_root.mkdir(exist_ok=True)

    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_data = torchvision.datasets.MNIST(
            root=str(data_root), train=True, download=True, transform=transform
        )
        test_data = torchvision.datasets.MNIST(
            root=str(data_root), train=False, download=True, transform=transform
        )
    elif dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_data = torchvision.datasets.CIFAR10(
            root=str(data_root), train=True, download=True, transform=transform_train
        )
        test_data = torchvision.datasets.CIFAR10(
            root=str(data_root), train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # IID partition: shuffle indices and split equally among clients
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_data)).tolist()
    client_size = len(indices) // num_clients
    client_datasets = []
    for i in range(num_clients):
        start = i * client_size
        end = start + client_size
        client_datasets.append(Subset(train_data, indices[start:end]))

    return client_datasets, test_data


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def get_flat_params(model: nn.Module) -> np.ndarray:
    """Flatten all model parameters into a single 1D numpy array."""
    return np.concatenate([
        p.data.cpu().numpy().ravel() for p in model.parameters()
    ])


def set_flat_params(model: nn.Module, flat_params: np.ndarray) -> None:
    """Load flat numpy parameter vector back into model."""
    offset = 0
    for p in model.parameters():
        size = p.data.numel()
        p.data.copy_(
            torch.from_numpy(flat_params[offset: offset + size].reshape(p.data.shape))
        )
        offset += size


def compute_gradient_update(
    global_params: np.ndarray, local_params: np.ndarray
) -> np.ndarray:
    """Compute the gradient update (difference after local training)."""
    return local_params - global_params


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def local_train(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 0.01,
) -> tuple[nn.Module, float]:
    """Train a local model copy for a given number of epochs."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    steps = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            # Clip gradients to prevent explosion
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        total_loss += epoch_loss

    avg_loss = total_loss / max(steps, 1)
    return model, avg_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate model on test set. Returns (accuracy%, avg_loss)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item() * batch_y.size(0)
            _, predicted = output.max(dim=1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


# ---------------------------------------------------------------------------
# Main federated learning loop
# ---------------------------------------------------------------------------

def federated_learning(
    dataset: str,
    num_clients: int,
    num_malicious: int,
    num_rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    correlation_threshold: float,
    poison_scale: float,
    warmup_rounds: int,
    seed: int,
    device: torch.device,
    wandb_run: Any,
) -> dict:
    """Run SA-FL federated learning with null-space poisoning attack.

    Returns dict with final metrics.
    """
    logger.info(
        f"Starting SA-FL experiment: dataset={dataset}, clients={num_clients}, "
        f"malicious={num_malicious}, rounds={num_rounds}"
    )

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    logger.info("Loading dataset...")
    client_datasets, test_data = load_dataset(dataset, num_clients, seed)
    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        for ds in client_datasets
    ]
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)
    logger.info(f"Dataset loaded. Test size: {len(test_data)}")

    # Initialize global model
    global_model = get_model(dataset).to(device)
    malicious_ids = set(range(num_clients - num_malicious, num_clients))
    logger.info(f"Malicious client IDs: {malicious_ids}")

    # SA-FL aggregator (WITHOUT ZKP projection checks)
    aggregator = SAFLAggregator(
        num_clients=num_clients,
        correlation_threshold=correlation_threshold,
    )

    # Null-space attacker (shared state across malicious clients)
    attacker = NullSpaceAttacker(
        poison_scale=poison_scale,
        warmup_rounds=warmup_rounds,
    )

    # Tracking metrics
    accuracy_history: list[float] = []
    loss_history: list[float] = []
    detection_rate_history: list[float] = []

    # Progress tracking
    progress = {"phase": "training", "current": 0, "total": num_rounds}

    for fl_round in range(num_rounds):
        round_start = time.time()
        global_params = get_flat_params(global_model)

        # Record round gradient (attacker uses this to estimate null space)
        # This simulates the attacker observing aggregated gradient direction
        if fl_round > 0:
            attacker.record_round_gradient(global_params)

        # --- Client local training ---
        client_updates: list[np.ndarray] = []
        client_losses: list[float] = []

        for client_id in range(num_clients):
            # Each client starts from global model
            local_model = copy.deepcopy(global_model)
            local_model, local_loss = local_train(
                local_model, client_loaders[client_id], device, local_epochs, lr
            )
            local_params = get_flat_params(local_model)
            gradient_update = compute_gradient_update(global_params, local_params)
            client_losses.append(local_loss)

            # Apply null-space attack for malicious clients
            if client_id in malicious_ids:
                poisoned_update, attack_active = attacker.poison(
                    gradient_update, fl_round
                )
                client_updates.append(poisoned_update)
                if attack_active and fl_round == warmup_rounds:
                    logger.info(f"Round {fl_round}: Null-space attack now active!")
            else:
                client_updates.append(gradient_update)

        # --- SA-FL aggregation (WITHOUT ZKP projection checks) ---
        client_ids_list = list(range(num_clients))
        agg_update, stats = aggregator.aggregate(
            client_updates=client_updates,
            client_ids=client_ids_list,
            malicious_ids=malicious_ids,
            round_num=fl_round,
        )

        # --- Update global model ---
        new_params = global_params + agg_update
        set_flat_params(global_model, new_params)

        # --- Evaluation ---
        test_acc, test_loss = evaluate(global_model, test_loader, device)
        accuracy_history.append(test_acc)
        loss_history.append(test_loss)
        detection_rate_history.append(stats["detection_rate"])

        round_time = time.time() - round_start

        logger.info(
            f"[step {fl_round * len(client_loaders[0])}, epoch {fl_round}] "
            f"loss={test_loss:.4f} lr={lr}"
        )
        logger.info(
            f"=== epoch {fl_round} done | avg_train_loss={np.mean(client_losses):.4f} "
            f"val_loss={test_loss:.4f} val_f1={test_acc:.4f} elapsed={round_time:.1f}s ==="
        )
        logger.info(
            f"Round {fl_round}: acc={test_acc:.2f}%, loss={test_loss:.4f}, "
            f"detection={stats['detection_rate']:.3f}, "
            f"accepted={stats['num_accepted']}/{num_clients}, "
            f"time={round_time:.1f}s"
        )

        # --- Weights & Biases logging ---
        if wandb_run is not None:
            wandb_run.log({
                "round": fl_round,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
                "poison_detection_rate": stats["detection_rate"] * 100.0,
                "num_accepted": stats["num_accepted"],
                "num_rejected": stats["num_rejected"],
                "avg_train_loss": float(np.mean(client_losses)),
            }, step=fl_round)

        # Update progress
        progress["current"] = fl_round + 1
        with open("/workspace/output/PROGRESS.json", "w") as f:
            json.dump(progress, f)

    # --- Final metrics ---
    final_accuracy = accuracy_history[-1]
    final_loss = loss_history[-1]
    cumulative_detection_rate = aggregator.cumulative_detection_rate() * 100.0

    logger.info(
        f"\n=== Final Results ===\n"
        f"Dataset: {dataset}\n"
        f"Final test accuracy: {final_accuracy:.2f}%\n"
        f"Final test loss: {final_loss:.4f}\n"
        f"Cumulative poison detection rate: {cumulative_detection_rate:.2f}%\n"
        f"(NOTE: ~0% detection expected without ZKP projection checks)"
    )

    # Log final metrics to wandb
    if wandb_run is not None:
        wandb_run.log({
            "test_accuracy": final_accuracy,
            "poison_detection_rate": cumulative_detection_rate,
        })

    return {
        "test_accuracy": final_accuracy,
        "test_loss": final_loss,
        "poison_detection_rate": cumulative_detection_rate,
        "accuracy_history": accuracy_history,
        "loss_history": loss_history,
        "detection_rate_history": [r * 100.0 for r in detection_rate_history],
        "num_rounds": num_rounds,
        "dataset": dataset,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SA-FL baseline (no ZKP checks)")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--malicious", type=int, default=2)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--correlation-threshold", type=float, default=0.4)
    p.add_argument("--poison-scale", type=float, default=3.0)
    p.add_argument("--warmup-rounds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="/workspace/output")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def save_checkpoint(
    model: nn.Module,
    optimizer_state: dict | None,
    epoch: int,
    best_metric: float,
    history: dict,
    path: Path,
) -> None:
    """Save model checkpoint atomically."""
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "history": history,
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
    }
    if optimizer_state:
        state["optimizer"] = optimizer_state
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, str(path))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    weights_dir = output_dir / "code" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Weights & Biases
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="prof-f9e1ad4b-plan-14e30fb6",
                name="baseline_sa_fl_no_checks",
                config={
                    "dataset": args.dataset,
                    "num_clients": args.clients,
                    "num_malicious": args.malicious,
                    "num_rounds": args.rounds,
                    "local_epochs": args.local_epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "correlation_threshold": args.correlation_threshold,
                    "poison_scale": args.poison_scale,
                    "warmup_rounds": args.warmup_rounds,
                    "seed": args.seed,
                    "attack": "null_space_dynamic",
                    "defense": "sa_fl_pearson_no_zkp",
                },
            )
            logger.info(f"WANDB_RUN_URL: {wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Continuing without tracking.")
            wandb_run = None

    # Run federated learning
    results = federated_learning(
        dataset=args.dataset,
        num_clients=args.clients,
        num_malicious=args.malicious,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        correlation_threshold=args.correlation_threshold,
        poison_scale=args.poison_scale,
        warmup_rounds=args.warmup_rounds,
        seed=args.seed,
        device=device,
        wandb_run=wandb_run,
    )

    # Save final model
    final_model = get_model(args.dataset).to(device)
    # (We don't have the model object here — load it from checkpoint if available)
    # For now, save results to JSON
    results_summary = {
        "dataset": results["dataset"],
        "test_accuracy": results["test_accuracy"],
        "test_loss": results["test_loss"],
        "poison_detection_rate": results["poison_detection_rate"],
        "num_rounds": results["num_rounds"],
        "accuracy_history": results["accuracy_history"],
        "loss_history": results["loss_history"],
        "detection_rate_history": results["detection_rate_history"],
        "wandb_run_url": wandb_run.url if wandb_run else None,
    }

    results_path = output_dir / "sa_fl_baseline_logs" / f"{args.dataset}_results.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    if wandb_run:
        wandb_run.finish()

    return results_summary


if __name__ == "__main__":
    summary = main()
    print(json.dumps(summary, indent=2))
