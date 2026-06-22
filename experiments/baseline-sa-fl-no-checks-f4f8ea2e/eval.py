"""Evaluation script for SA-FL trained models.

Loads saved weights and re-evaluates on the test set.

Usage:
    python eval.py --dataset mnist --weights weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent))
from models.cnn import get_model


def load_test_data(dataset: str, data_root: str = "/workspace/datasets"):
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_data = torchvision.datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )
    elif dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_data = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DataLoader(test_data, batch_size=256, shuffle=False, num_workers=0)


def evaluate(model, test_loader, device):
    import torch.nn as nn
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
    return 100.0 * correct / total, total_loss / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--weights", default="weights/best.pt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.dataset).to(device)

    weights_path = Path(__file__).parent / args.weights
    if weights_path.exists():
        ckpt = torch.load(str(weights_path), map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Weights not found at {weights_path}, evaluating random model")

    test_loader = load_test_data(args.dataset)
    acc, loss = evaluate(model, test_loader, device)
    result = {"test_accuracy": acc, "test_loss": loss, "dataset": args.dataset}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
