"""Evaluate saved S-ZKP model weights on test sets."""
import os, sys, json
import torch, numpy as np

OUTPUT_DIR = "/workspace/output"
CODE_DIR   = f"{OUTPUT_DIR}/code"
sys.path.insert(0, CODE_DIR)
from train import make_model, evaluate, load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    out = {}
    for ds in ["mnist", "cifar10"]:
        wpath = f"{CODE_DIR}/weights/best_{ds}_szkp.pt"
        if not os.path.exists(wpath):
            print(f"No weights for {ds}, skipping.")
            continue
        model = make_model(ds)
        model.load_state_dict(torch.load(wpath, map_location=DEVICE))
        _, _, teX, teY = load_dataset(ds)
        acc, loss = evaluate(model, teX, teY)
        print(f"{ds}: acc={acc:.2f}%  loss={loss:.4f}")
        out[ds] = {"accuracy": acc, "loss": loss}
    with open(f"{OUTPUT_DIR}/eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved eval_results.json")

if __name__ == "__main__":
    main()
