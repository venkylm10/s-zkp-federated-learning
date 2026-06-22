"""Evaluate saved model weights on test set."""
import os
import json
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from train import (
    MNISTModel, CIFAR10Model, get_model, evaluate,
    load_dataset, DEVICE, CODE_DIR, OUTPUT_DIR,
    set_flat_params, get_flat_params
)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    args = parser.parse_args()

    import train as T
    T._dataset_name = args.dataset

    _, _, test_X, test_y = load_dataset(args.dataset)
    model = get_model(args.dataset)
    weight_path = f"{CODE_DIR}/weights/best_{args.dataset}.pt"
    if not os.path.exists(weight_path):
        print(f"No weights at {weight_path}")
        return

    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    metrics = evaluate(model, test_X, test_y)
    print(f"[{args.dataset}] Best weights: accuracy={metrics['accuracy']:.2f}%, loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
