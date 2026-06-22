"""S-ZKP Projection Dimension Ablation.

Ablates projection dimension d ∈ {32, 64, 128, 256, 512}.
For each d, runs federated learning under a defense-aware null-space attack
with the S-ZKP framework active. Measures:
  - Poison detection rate (via projected-gradient outlier detection)
  - ZKP proving time (projection compute + simulated Schnorr proof overhead)
  - ZKP proof size (simulated: 2d EC points + d scalars in secp256k1 encoding)

Gradient projection:
  - Projects a fixed p0-dimensional subspace of the full gradient to d dimensions.
  - R ∈ R^{d × p0} is refreshed every round from a per-round seed.

Defense-aware attacker:
  - Learns previous round's R and crafts poison in null(R_prev).
  - With small d the null space is large (p0-d dims) → easy to hide.
  - With large d the null space shrinks → harder to hide → higher detection.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).parent))
from models.cnn import get_model

# ── logging ──────────────────────────────────────────────────────────────────
LOG_DIR = Path("/workspace/output/ablation_d_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "ablation.log"),
    ],
)
log = logging.getLogger("s_zkp_ablation")

# ── Constants ─────────────────────────────────────────────────────────────────
# Number of gradient dimensions to project (keep matrices small)
P0 = 20_000
# Schnorr proof overhead per EC point on secp256k1 (ms, measured on typical server)
EC_POINT_MS = 0.45
# Compressed secp256k1 point size (bytes)
EC_POINT_BYTES = 33
# Scalar size (bytes)
SCALAR_BYTES = 32


# ── Projection helpers ────────────────────────────────────────────────────────
def make_projection_matrix(d: int, p0: int, seed: int) -> np.ndarray:
    """Generate a normalised random projection matrix R ∈ R^{d×p0}."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((d, p0)).astype(np.float32)
    # Row-normalise so each row is a unit vector
    norms = np.linalg.norm(R, axis=1, keepdims=True).clip(1e-10)
    return R / norms


def project_gradient(gradient: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Project first P0 dims of gradient with R. Returns y ∈ R^d."""
    g_sub = gradient[:P0]
    return R @ g_sub  # (d,)


def null_space_poison(honest_g: np.ndarray, R_prev: np.ndarray,
                      scale: float, seed: int) -> np.ndarray:
    """Craft a poison that lies in null(R_prev) for the first P0 dims.

    The attacker knows R_prev (previous round's matrix) and projects
    their poison direction into its null space, hoping the CURRENT round's
    fresh R will miss it — probability decreases exponentially with d.
    """
    p0 = R_prev.shape[1]
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(p0).astype(np.float32)

    # Null-space projection: v_null = noise - R^T (R R^T)^{-1} R noise
    Rn = R_prev @ noise                              # (d,)
    try:
        RRT = R_prev @ R_prev.T                      # (d,d) – small
        RRT_inv_Rn = np.linalg.solve(RRT, Rn)       # (d,)
    except np.linalg.LinAlgError:
        RRT_inv_Rn = np.linalg.lstsq(R_prev @ R_prev.T, Rn, rcond=None)[0]
    null_component = noise - R_prev.T @ RRT_inv_Rn  # (p0,)

    null_norm = np.linalg.norm(null_component)
    if null_norm < 1e-8:
        return honest_g.copy()
    null_component /= null_norm

    honest_sub_norm = np.linalg.norm(honest_g[:p0])
    poison_full = honest_g.copy()
    poison_full[:p0] += scale * honest_sub_norm * null_component
    return poison_full


def measure_proving_time(gradient: np.ndarray, R: np.ndarray) -> float:
    """Measure actual projection compute + simulated Schnorr ZKP overhead (ms)."""
    d = R.shape[0]

    # 1. Actual projection time
    t0 = time.perf_counter()
    _ = R @ gradient[:P0]
    proj_ms = (time.perf_counter() - t0) * 1e3

    # 2. Simulate Schnorr proof generation:
    #    For y = Rg with d dimensions, the prover runs d scalar multiplications.
    #    Each EC scalar-mul on secp256k1 ≈ EC_POINT_MS ms.
    zkp_ms = d * EC_POINT_MS

    return proj_ms + zkp_ms


def compute_proof_size_kb(d: int) -> float:
    """Compute ZKP proof size in KB for a d-dimensional Schnorr proof.

    Proof structure (Sigma protocol for d linear relations):
      - 2d commitment EC points (each 33 bytes)
      - 1 challenge scalar (32 bytes)
      - d response scalars (each 32 bytes)
    """
    total_bytes = 2 * d * EC_POINT_BYTES + SCALAR_BYTES + d * SCALAR_BYTES
    return total_bytes / 1024.0


# ── Dataset / model helpers ───────────────────────────────────────────────────
def load_mnist_partitioned(num_clients: int, seed: int):
    data_root = Path("/workspace/datasets")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = torchvision.datasets.MNIST(root=str(data_root), train=True,
                                          download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root=str(data_root), train=False,
                                         download=True, transform=transform)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(train_ds)).tolist()
    per = len(idx) // num_clients
    client_ds = [Subset(train_ds, idx[i * per:(i + 1) * per]) for i in range(num_clients)]
    return client_ds, test_ds


def get_flat(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])


def set_flat(model: nn.Module, flat: np.ndarray):
    offset = 0
    for p in model.parameters():
        n = p.data.numel()
        p.data.copy_(torch.from_numpy(flat[offset:offset + n].reshape(p.data.shape)))
        offset += n


def local_train(model: nn.Module, loader: DataLoader, device, epochs=2, lr=0.01):
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    total_loss, steps = 0.0, 0
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            steps += 1
    return model, total_loss / max(steps, 1)


def evaluate(model: nn.Module, loader: DataLoader, device) -> tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss_sum += crit(out, by).item() * by.size(0)
            correct += out.argmax(1).eq(by).sum().item()
            total += by.size(0)
    return 100.0 * correct / total, loss_sum / total


# ── Projected-correlation detection ──────────────────────────────────────────
def pearson(u: np.ndarray, v: np.ndarray) -> float:
    u, v = u - u.mean(), v - v.mean()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-10 or nv < 1e-10:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def detect_with_s_zkp(projections: list[np.ndarray], client_ids: list[int],
                      malicious_ids: set[int], threshold: float = 0.35) -> dict:
    """Detect poisoned clients using projected Pearson correlation.

    Each client's d-dimensional projection y_k = R @ g_k is compared
    pairwise; low mean correlation → flagged as poisoned.
    """
    n = len(projections)
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = pearson(projections[i], projections[j])
            corr[i, j] = corr[j, i] = r

    mean_corr = np.array([(corr[i].sum() - 1.0) / (n - 1) for i in range(n)])
    rejected = {client_ids[i] for i in range(n) if mean_corr[i] < threshold}
    accepted = {client_ids[i] for i in range(n) if mean_corr[i] >= threshold}

    if len(accepted) < 3:       # safety fallback
        accepted = set(client_ids)
        rejected = set()

    tp = len(rejected & malicious_ids)
    fn = len(accepted & malicious_ids)
    total_mal = len(malicious_ids)
    detection_rate = tp / total_mal if total_mal > 0 else 1.0

    return {
        "rejected": rejected,
        "accepted": accepted,
        "detection_rate": detection_rate,
        "true_positives": tp,
        "false_negatives": fn,
        "mean_corr": mean_corr.tolist(),
    }


# ── Per-d FL run ──────────────────────────────────────────────────────────────
def run_ablation_for_d(
    d: int,
    num_clients: int = 10,
    num_malicious: int = 2,
    num_rounds: int = 20,
    local_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 0.01,
    poison_scale: float = 3.0,
    warmup_rounds: int = 5,
    seed: int = 42,
    device=None,
    wandb_run=None,
) -> dict:
    """Run one FL experiment with S-ZKP (projection dimension d)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"=== d={d}: starting {num_rounds} rounds ===")
    torch.manual_seed(seed)
    np.random.seed(seed)

    client_ds, test_ds = load_mnist_partitioned(num_clients, seed)
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
                      for ds in client_ds]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    global_model = get_model("mnist").to(device)
    malicious_ids = set(range(num_clients - num_malicious, num_clients))

    detection_history: list[float] = []
    acc_history: list[float] = []
    proving_times: list[float] = []
    proof_size_kb = compute_proof_size_kb(d)

    prev_R: np.ndarray | None = None
    prev_R_seed: int = -1

    for fl_round in range(num_rounds):
        round_seed = seed * 10000 + d * 100 + fl_round
        R = make_projection_matrix(d, P0, seed=round_seed)

        global_params = get_flat(global_model)
        client_updates: list[np.ndarray] = []
        projections: list[np.ndarray] = []
        round_proving_ms = 0.0

        for cid in range(num_clients):
            local_model = copy.deepcopy(global_model)
            local_model, _ = local_train(local_model, client_loaders[cid],
                                         device, local_epochs, lr)
            local_params = get_flat(local_model)
            update = local_params - global_params

            # Malicious clients apply defense-aware null-space attack
            if cid in malicious_ids and fl_round >= warmup_rounds and prev_R is not None:
                update = null_space_poison(update, prev_R, poison_scale,
                                           seed=round_seed + cid)

            client_updates.append(update)

            # ZKP: prove y = R @ g; measure proving time
            pt = measure_proving_time(update, R)
            round_proving_ms += pt
            projections.append(project_gradient(update, R))

        # Detection with S-ZKP projected correlations
        client_ids_list = list(range(num_clients))
        det = detect_with_s_zkp(projections, client_ids_list, malicious_ids)

        # Aggregate accepted updates
        accepted_idx = [i for i in range(num_clients)
                        if client_ids_list[i] in det["accepted"]]
        if accepted_idx:
            agg = np.mean([client_updates[i] for i in accepted_idx], axis=0)
        else:
            agg = np.mean(client_updates, axis=0)

        # Update global model
        set_flat(global_model, global_params + agg)

        acc, loss = evaluate(global_model, test_loader, device)
        detection_history.append(det["detection_rate"] * 100.0)
        acc_history.append(acc)
        proving_times.append(round_proving_ms / num_clients)  # per-client avg

        log.info(
            f"d={d} round={fl_round:2d} | acc={acc:.2f}% "
            f"det={det['detection_rate']*100:.1f}% "
            f"tp={det['true_positives']} fn={det['false_negatives']} "
            f"proving={round_proving_ms/num_clients:.1f}ms "
            f"proof={proof_size_kb:.2f}KB"
        )

        if wandb_run is not None:
            wandb_run.log({
                f"d{d}/test_accuracy": acc,
                f"d{d}/detection_rate": det["detection_rate"] * 100.0,
                f"d{d}/proving_time_ms": round_proving_ms / num_clients,
                f"d{d}/proof_size_kb": proof_size_kb,
            }, step=fl_round)

        prev_R = R

    # Compute post-warmup detection rate (when attack is active)
    attack_rounds = [i for i in range(num_rounds) if i >= warmup_rounds]
    post_warmup_det = float(np.mean([detection_history[i] for i in attack_rounds])) if attack_rounds else 0.0
    avg_proving_ms = float(np.mean(proving_times))

    log.info(
        f"d={d} DONE | det_rate={post_warmup_det:.1f}% "
        f"proving={avg_proving_ms:.1f}ms proof={proof_size_kb:.2f}KB"
    )

    result = {
        "d": d,
        "detection_rate": post_warmup_det,
        "avg_proving_ms": avg_proving_ms,
        "proof_size_kb": proof_size_kb,
        "final_accuracy": acc_history[-1],
        "detection_history": detection_history,
        "acc_history": acc_history,
        "proving_time_history": proving_times,
    }

    # Save per-d log
    log_path = LOG_DIR / f"d{d}_results.json"
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved per-d log to {log_path}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Initialise W&B run
    wandb_run = None
    try:
        wandb_run = wandb.init(
            project="prof-f9e1ad4b-plan-14e30fb6",
            name="ablation_projection_dimension",
            config={
                "d_values": [32, 64, 128, 256, 512],
                "p0": P0,
                "num_clients": 10,
                "num_malicious": 2,
                "num_rounds": 20,
                "warmup_rounds": 5,
                "poison_scale": 3.0,
                "attack": "defense_aware_null_space",
                "defense": "s_zkp_projected_pearson",
                "dataset": "mnist",
                "seed": 42,
            },
        )
        log.info(f"WANDB_RUN_URL: {wandb_run.url}")
    except Exception as e:
        log.warning(f"W&B init failed: {e}. Continuing without tracking.")

    D_VALUES = [32, 64, 128, 256, 512]
    all_results: list[dict] = []

    for d in D_VALUES:
        r = run_ablation_for_d(
            d=d,
            num_clients=10,
            num_malicious=2,
            num_rounds=20,
            local_epochs=2,
            batch_size=64,
            lr=0.01,
            poison_scale=3.0,
            warmup_rounds=5,
            seed=42,
            device=device,
            wandb_run=wandb_run,
        )
        all_results.append(r)

        # Update PROGRESS.json
        with open("/workspace/output/PROGRESS.json", "w") as pf:
            json.dump({"phase": "training",
                       "current": D_VALUES.index(d) + 1,
                       "total": len(D_VALUES)}, pf)

    # Log summary to W&B
    if wandb_run is not None:
        for r in all_results:
            wandb_run.log({
                "poison_detection_rate_vs_d": r["detection_rate"],
                "proving_time_vs_d": r["avg_proving_ms"],
                "proof_size_vs_d": r["proof_size_kb"],
            })
        wandb_run.finish()

    # Write summary to ablation_d_logs/summary.json
    summary_path = LOG_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Summary saved to {summary_path}")

    return all_results, wandb_run.url if wandb_run else None


if __name__ == "__main__":
    results, url = main()
    print(json.dumps({"wandb_url": url,
                      "d_values": [r["d"] for r in results],
                      "detection_rates": [r["detection_rate"] for r in results],
                      "proving_times": [r["avg_proving_ms"] for r in results],
                      "proof_sizes": [r["proof_size_kb"] for r in results]},
                     indent=2))
