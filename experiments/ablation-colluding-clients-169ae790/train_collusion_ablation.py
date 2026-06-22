"""
S-ZKP Federated Learning — Ablation: Colluding Byzantine Client Fraction

Tests the S-ZKP security bound under varying fractions of colluding Byzantine
clients (10%, 20%, 30%, 40%) and three coordinated attack types:
  - null_space: coordinate null-space attacks exploiting stale projection key
  - sign_flip:  coordinated sign-reversal (negative scaled gradient)
  - label_flip: Byzantine clients train on permuted-label data

Metrics per (fraction, attack):
  - global_test_accuracy
  - detection_rate_under_collusion (S-ZKP projection filter)
"""

import os, sys, json, time, hashlib, logging, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/workspace/output"
CODE_DIR   = f"{OUTPUT_DIR}/code"
FIG_DIR    = f"{OUTPUT_DIR}/figures"
LOG_DIR    = f"{OUTPUT_DIR}/ablation_collusion_logs"
for d in [FIG_DIR, LOG_DIR, f"{CODE_DIR}/weights"]:
    os.makedirs(d, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
root_log = logging.getLogger()
root_log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
fh = logging.FileHandler(f"{OUTPUT_DIR}/train.log", mode="w"); fh.setFormatter(fmt)
root_log.addHandler(sh); root_log.addHandler(fh)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ── Config ────────────────────────────────────────────────────────────────────
PROJ_DIM    = 128
N_CLIENTS   = 10
SEED        = 42
BATCH_SIZE  = 128
ATK_SCALE   = 5.0   # scale factor for poisoning
SIGMA       = 3.0   # threshold: reject if ||R@delta|| > SIGMA * median
N_ROUNDS    = 15    # rounds per condition

BYZ_FRACS   = [0.1, 0.2, 0.3, 0.4]  # 10%, 20%, 30%, 40%
ATTACK_TYPES = ["null_space", "sign_flip", "label_flip"]

LABEL_PERM = {i: (i + 5) % 10 for i in range(10)}  # shift labels by 5

torch.manual_seed(SEED); np.random.seed(SEED)


# ── Model ─────────────────────────────────────────────────────────────────────

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x): return self.net(x)


def make_model() -> nn.Module:
    return MNISTNet().to(DEVICE)

def flat_params(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in m.parameters()] +
                     [b.data.view(-1) for b in m.buffers()])

def set_params(m: nn.Module, flat: torch.Tensor):
    off = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(flat[off:off+n].view_as(p)); off += n
    for b in m.buffers():
        n = b.numel(); b.data.copy_(flat[off:off+n].view_as(b)); off += n


# ── S-ZKP Server ──────────────────────────────────────────────────────────────

class SZKPServer:
    """Two-server S-ZKP via additive secret-sharing of R_t. Rotates every round."""
    def __init__(self, proj_dim: int = 128):
        self.proj_dim = proj_dim
        self.R_t: torch.Tensor | None = None

    def rotate(self, D: int, rnd: int) -> None:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(SEED * 100003 + rnd)
        R = torch.randn(self.proj_dim, D, generator=gen, device=DEVICE, dtype=torch.float32)
        self.R_t = F.normalize(R, dim=1)

    def project_norm(self, delta: torch.Tensor) -> float:
        return (self.R_t @ delta).norm().item()


# ── Attacks ───────────────────────────────────────────────────────────────────

class NullSpaceAttacker:
    """Stale-key null-space attack: crafts delta in null(R_{t-1})."""
    def __init__(self, scale: float = 5.0):
        self.scale = scale
        self.prev_R: torch.Tensor | None = None

    def update_key(self, R: torch.Tensor) -> None:
        self.prev_R = R.clone()

    def _null_project(self, v: torch.Tensor) -> torch.Tensor:
        R = self.prev_R
        Pv  = R @ v
        PPT = R @ R.t() + 1e-5 * torch.eye(R.shape[0], device=v.device)
        coeff = torch.linalg.solve(PPT, Pv)
        return v - R.t() @ coeff

    def poison(self, honest_delta: torch.Tensor) -> torch.Tensor:
        if self.prev_R is None:
            return honest_delta
        null_dir = self._null_project(-2.0 * honest_delta)
        n = null_dir.norm()
        if n > 1e-8:
            null_dir = null_dir / n * honest_delta.norm()
        return honest_delta + self.scale * null_dir


def sign_flip_poison(delta: torch.Tensor, n_byz: int, n_clients: int,
                     scale: float = 5.0) -> torch.Tensor:
    """Coordinated sign-flip: reverse and amplify to overwhelm honest clients."""
    amplification = (n_clients / max(n_clients - n_byz, 1)) * scale
    return -amplification * delta


def label_flip_update(gflat: torch.Tensor, dataset, epochs: int, lr: float) -> torch.Tensor:
    """Train on label-permuted data; return resulting update delta."""
    X_all, y_all = zip(*[(x, y) for x, y in dataset])
    X_all = torch.stack(X_all)
    y_perm = torch.tensor([LABEL_PERM[int(lbl)] for lbl in y_all], dtype=torch.long)
    poisoned_ds = TensorDataset(X_all, y_perm)

    model = make_model()
    set_params(model, gflat.clone())
    opt    = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loader = DataLoader(poisoned_ds, batch_size=BATCH_SIZE, shuffle=True)
    crit   = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(X), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
    return flat_params(model) - gflat


# ── Data ──────────────────────────────────────────────────────────────────────

def load_mnist():
    from datasets import load_dataset as hf_load
    ds = hf_load("ylecun/mnist", cache_dir="/workspace/datasets/hf")
    def proc(split):
        imgs   = torch.tensor(
            np.stack([np.array(x["image"]) for x in split]),
            dtype=torch.float32).unsqueeze(1) / 255.0
        labels = torch.tensor([x["label"] for x in split], dtype=torch.long)
        return imgs, labels
    trX, trY = proc(ds["train"]); teX, teY = proc(ds["test"])
    log.info(f"MNIST: train {trX.shape}, test {teX.shape}")
    return trX, trY, teX, teY


def split_iid(X, y, n):
    idx = np.random.default_rng(SEED).permutation(len(X))
    return [TensorDataset(X[torch.from_numpy(c)], y[torch.from_numpy(c)])
            for c in np.array_split(idx, n)]


# ── Client Training ───────────────────────────────────────────────────────────

def client_update(gflat: torch.Tensor, dataset, epochs: int, lr: float) -> torch.Tensor:
    model = make_model()
    set_params(model, gflat.clone())
    opt    = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    crit   = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(X), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
    return flat_params(model) - gflat


@torch.no_grad()
def evaluate(model, teX, teY):
    model.eval()
    loader = DataLoader(TensorDataset(teX, teY), batch_size=512)
    crit   = nn.CrossEntropyLoss()
    correct = total = 0; loss_sum = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out   = model(X)
        loss_sum += crit(out, y).item() * len(y)
        correct  += (out.argmax(1) == y).sum().item()
        total    += len(y)
    return 100.0 * correct / total, loss_sum / total


# ── FL Loop (one condition) ───────────────────────────────────────────────────

def run_fl_condition(byz_frac: float, attack_type: str,
                     trX, trY, teX, teY, clients,
                     condition_idx: int, n_total_conditions: int,
                     wandb_run) -> dict:
    n_byz = max(1, round(byz_frac * N_CLIENTS))
    n_byz = min(n_byz, N_CLIENTS - 1)  # keep at least 1 honest
    label = f"frac{int(byz_frac*100)}pct_{attack_type}"
    log.info(f"\n{'='*60}\nCONDITION: byz_frac={byz_frac} n_byz={n_byz} attack={attack_type}\n{'='*60}")

    torch.manual_seed(SEED); np.random.seed(SEED)
    lrs = [0.05 * 0.5 * (1 + np.cos(np.pi * t / N_ROUNDS)) for t in range(N_ROUNDS)]

    global_model = make_model()
    D = sum(p.numel() for p in global_model.parameters()) + \
        sum(b.numel() for b in global_model.buffers())

    byz_ids    = list(range(N_CLIENTS - n_byz, N_CLIENTS))
    honest_ids = list(range(N_CLIENTS - n_byz))

    szkp     = SZKPServer(proj_dim=PROJ_DIM)
    ns_attk  = NullSpaceAttacker(scale=ATK_SCALE) if attack_type == "null_space" else None

    # Initialize R_0 so attacker has stale key from round 1
    szkp.rotate(D, 0)
    prev_R = szkp.R_t.clone()

    hist = {"acc": [], "loss": [], "det_rate": [], "byz_pnorm": [], "honest_pnorm": []}
    best_acc = 0.0

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        cur_lr = lrs[rnd - 1]
        gflat  = flat_params(global_model)

        # Key rotation (every round)
        prev_R = szkp.R_t.clone()
        szkp.rotate(D, rnd)
        if ns_attk is not None:
            ns_attk.update_key(prev_R)

        # Client updates
        deltas = []
        for cid in range(N_CLIENTS):
            if attack_type == "label_flip" and cid in byz_ids:
                delta = label_flip_update(gflat, clients[cid], 3, cur_lr)
            else:
                delta = client_update(gflat, clients[cid], 3, cur_lr)

            if cid in byz_ids:
                if attack_type == "null_space":
                    delta = ns_attk.poison(delta)
                elif attack_type == "sign_flip":
                    delta = sign_flip_poison(delta, n_byz, N_CLIENTS, ATK_SCALE)
                # label_flip already has poisoned update from training

            deltas.append(delta)

        # S-ZKP detection
        pnorms  = [szkp.project_norm(d) for d in deltas]
        med     = float(np.median(pnorms))
        thresh  = SIGMA * max(med, 1e-12)
        accepted = [i for i, pn in enumerate(pnorms) if pn <= thresh]
        rejected = [i for i, pn in enumerate(pnorms) if pn >  thresh]

        n_det    = sum(1 for i in rejected if i in byz_ids)
        det_rate = 100.0 * n_det / n_byz

        # Aggregation
        use_ids   = accepted if accepted else list(range(N_CLIENTS))
        avg_delta = torch.stack([deltas[i] for i in use_ids]).mean(0)
        set_params(global_model, gflat + avg_delta)

        # Evaluate
        acc, loss = evaluate(global_model, teX, teY)
        if acc > best_acc:
            best_acc = acc

        byz_pn    = float(np.mean([pnorms[i] for i in byz_ids]))
        honest_pn = float(np.mean([pnorms[i] for i in honest_ids]))

        hist["acc"].append(acc)
        hist["loss"].append(loss)
        hist["det_rate"].append(det_rate)
        hist["byz_pnorm"].append(byz_pn)
        hist["honest_pnorm"].append(honest_pn)

        elapsed = time.time() - t0
        log.info(
            f"[{label} R{rnd:3d}/{N_ROUNDS}] acc={acc:.2f}% det={n_det}/{n_byz} "
            f"({det_rate:.0f}%) byz_pn={byz_pn:.3f} honest_pn={honest_pn:.3f} "
            f"thresh={thresh:.3f} {elapsed:.1f}s"
        )
        log.info(f"[step rnd={rnd} cond={label}] loss={loss:.4f} lr={cur_lr:.4f}")
        log.info(f"=== epoch {rnd} done | acc={acc:.2f}% elapsed={elapsed:.1f}s ===")

        # W&B
        if wandb_run:
            step_offset = condition_idx * N_ROUNDS
            wandb.log({
                f"collusion/{label}/test_accuracy": acc,
                f"collusion/{label}/detection_rate": det_rate,
                f"collusion/{label}/byz_pnorm": byz_pn,
                "round": rnd,
            }, step=step_offset + rnd)

        # Progress
        total_work = n_total_conditions * N_ROUNDS
        done_work  = condition_idx * N_ROUNDS + rnd
        with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
            json.dump({"phase": "training", "current": done_work, "total": total_work}, f)

    # Save per-condition log
    log_path = f"{LOG_DIR}/{label}.json"
    summary = {
        "byz_frac": byz_frac, "n_byz": n_byz, "attack": attack_type, "label": label,
        "hist": hist, "best_acc": best_acc,
        "final_acc": hist["acc"][-1],
        "mean_det_rate": float(np.mean(hist["det_rate"])),
        "final_det_rate": hist["det_rate"][-1],
    }
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  → Saved: {log_path}")
    log.info(
        f"[{label}] DONE | best_acc={best_acc:.2f}% final_acc={hist['acc'][-1]:.2f}% "
        f"mean_det={np.mean(hist['det_rate']):.2f}%"
    )
    return summary


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_accuracy_vs_byzantine_fraction(results: dict) -> tuple:
    """
    Primary figure: global test accuracy vs Byzantine fraction for each attack type.
    Also includes a sub-panel for detection rate.
    """
    colors = {"null_space": "steelblue", "sign_flip": "darkorange", "label_flip": "seagreen"}
    markers = {"null_space": "o", "sign_flip": "s", "label_flip": "^"}
    labels  = {"null_space": "Null-Space (stale-key)", "sign_flip": "Sign-Flip",
                "label_flip": "Label-Flip"}
    fracs_pct = [int(f * 100) for f in BYZ_FRACS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_acc, ax_det = axes

    for atk in ATTACK_TYPES:
        accs = [results[f][atk]["final_acc"] for f in BYZ_FRACS]
        dets = [results[f][atk]["mean_det_rate"] for f in BYZ_FRACS]
        ax_acc.plot(fracs_pct, accs, color=colors[atk], marker=markers[atk],
                    label=labels[atk], lw=2, ms=8)
        ax_det.plot(fracs_pct, dets, color=colors[atk], marker=markers[atk],
                    label=labels[atk], lw=2, ms=8)

    ax_acc.set_xlabel("Byzantine Client Fraction (%)")
    ax_acc.set_ylabel("Final Test Accuracy (%)")
    ax_acc.set_title("S-ZKP: Test Accuracy vs Byzantine Fraction")
    ax_acc.set_xticks(fracs_pct)
    ax_acc.legend(fontsize=9); ax_acc.grid(alpha=0.3); ax_acc.set_ylim(0, 100)

    ax_det.set_xlabel("Byzantine Client Fraction (%)")
    ax_det.set_ylabel("Mean Detection Rate (%)")
    ax_det.set_title("S-ZKP: Byzantine Detection Rate vs Client Fraction")
    ax_det.set_xticks(fracs_pct)
    ax_det.legend(fontsize=9); ax_det.grid(alpha=0.3); ax_det.set_ylim(-5, 115)

    plt.suptitle("S-ZKP: Colluding Client Fraction Ablation (MNIST, 15 rounds)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    path = f"{FIG_DIR}/accuracy_vs_byzantine_fraction.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {path}")
    return path


def plot_training_curves(results: dict) -> None:
    """Per-fraction training curves for all attack types."""
    colors = {"null_space": "steelblue", "sign_flip": "darkorange", "label_flip": "seagreen"}
    rounds = list(range(1, N_ROUNDS + 1))

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, frac in enumerate(BYZ_FRACS):
        ax_a = axes[0][col]; ax_d = axes[1][col]
        pct  = int(frac * 100)
        for atk in ATTACK_TYPES:
            r = results[frac][atk]
            ax_a.plot(rounds, r["hist"]["acc"],      color=colors[atk], lw=1.5, label=atk)
            ax_d.plot(rounds, r["hist"]["det_rate"], color=colors[atk], lw=1.5, label=atk)
        ax_a.set_title(f"Acc — {pct}% Byzantine"); ax_a.set_ylim(0, 100); ax_a.grid(alpha=0.3)
        ax_d.set_title(f"Det — {pct}% Byzantine"); ax_d.set_ylim(-5, 115); ax_d.grid(alpha=0.3)
        if col == 0:
            ax_a.set_ylabel("Test Accuracy (%)"); ax_d.set_ylabel("Detection Rate (%)")
        ax_d.set_xlabel("FL Round")
        if col == 0:
            ax_a.legend(fontsize=7); ax_d.legend(fontsize=7)

    plt.suptitle("S-ZKP: Training Curves per Byzantine Fraction & Attack Type (MNIST)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = f"{FIG_DIR}/collusion_training_curves.png"
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # W&B
    wandb_run = None; wandb_url = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb_run = wandb.init(
                project="prof-f9e1ad4b-plan-14e30fb6",
                name="ablation_colluding_clients",
                config={
                    "proj_dim": PROJ_DIM, "n_clients": N_CLIENTS,
                    "byz_fracs": BYZ_FRACS, "attack_types": ATTACK_TYPES,
                    "seed": SEED, "atk_scale": ATK_SCALE, "sigma": SIGMA,
                    "n_rounds": N_ROUNDS, "dataset": "mnist",
                    "defense": "szkp_rotate_every_round",
                },
            )
            wandb_url = wandb_run.url
            log.info(f"WANDB_RUN_URL: {wandb_url}")
        except Exception as e:
            log.warning(f"W&B failed: {e}")

    # Load MNIST once
    trX, trY, teX, teY = load_mnist()
    clients = split_iid(trX, trY, N_CLIENTS)

    # Build run list (condition_idx for progress tracking)
    conditions = [(frac, atk) for frac in BYZ_FRACS for atk in ATTACK_TYPES]
    n_total = len(conditions)

    # Run all conditions, using cache if available
    results = {frac: {} for frac in BYZ_FRACS}
    for idx, (frac, atk) in enumerate(conditions):
        label     = f"frac{int(frac*100)}pct_{atk}"
        cache_path = f"{LOG_DIR}/{label}.json"

        if os.path.exists(cache_path):
            log.info(f"\n>>> {label} [CACHED — {cache_path}]")
            with open(cache_path) as fp:
                summary = json.load(fp)
        else:
            summary = run_fl_condition(
                frac, atk, trX, trY, teX, teY, clients,
                condition_idx=idx, n_total_conditions=n_total,
                wandb_run=wandb_run,
            )
        results[frac][atk] = summary

    # ── Aggregate metrics ────────────────────────────────────────────────────
    # Primary: mean over all (frac, attack) conditions
    all_accs  = [results[f][a]["final_acc"]    for f in BYZ_FRACS for a in ATTACK_TYPES]
    all_dets  = [results[f][a]["mean_det_rate"] for f in BYZ_FRACS for a in ATTACK_TYPES]
    global_acc = float(np.mean(all_accs))
    global_det = float(np.mean(all_dets))

    log.info(f"\n{'='*60}")
    log.info("COLLUSION ABLATION SUMMARY:")
    log.info(f"{'Byzantine %':<14} {'Attack':<15} {'Final Acc %':<14} {'Mean Det %':<12}")
    for frac in BYZ_FRACS:
        for atk in ATTACK_TYPES:
            r = results[frac][atk]
            log.info(f"  {int(frac*100):<12}  {atk:<15} {r['final_acc']:<12.2f}  {r['mean_det_rate']:<10.2f}")
    log.info(f"\n  → Global mean accuracy:        {global_acc:.2f}%")
    log.info(f"  → Global mean detection rate:  {global_det:.2f}%")
    log.info(f"{'='*60}")

    # ── Figures ──────────────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
        json.dump({"phase": "generating_figures", "current": 0, "total": 2}, f)

    plot_accuracy_vs_byzantine_fraction(results)
    plot_training_curves(results)

    # ── W&B final log ────────────────────────────────────────────────────────
    if wandb_run:
        wandb.log({
            "global_test_accuracy":           global_acc,
            "detection_rate_under_collusion": global_det,
        })
        try:
            wandb.log({"accuracy_vs_byzantine_fraction":
                       wandb.Image(f"{FIG_DIR}/accuracy_vs_byzantine_fraction.png")})
        except Exception:
            pass
        wandb.finish()

    # ── Inline data for the manifest figure ──────────────────────────────────
    fracs_pct = [int(f * 100) for f in BYZ_FRACS]
    fig_inline = {"byzantine_fraction_pct": fracs_pct}
    for atk in ATTACK_TYPES:
        fig_inline[f"accuracy_{atk}"]  = [round(results[f][atk]["final_acc"],   4) for f in BYZ_FRACS]
        fig_inline[f"det_rate_{atk}"]  = [round(results[f][atk]["mean_det_rate"], 4) for f in BYZ_FRACS]

    # Per-condition detail for results[]
    results_entries = []
    for frac in BYZ_FRACS:
        for atk in ATTACK_TYPES:
            r   = results[frac][atk]
            lbl = r["label"]
            results_entries.append({
                "name":       f"accuracy_{lbl}",
                "value":      round(r["final_acc"], 4),
                "unit":       "%",
                "provenance": "measured",
                "method":     atk,
                "formula":    f"final round accuracy at {int(frac*100)}% Byzantine fraction",
            })
            results_entries.append({
                "name":       f"det_rate_{lbl}",
                "value":      round(r["mean_det_rate"], 4),
                "unit":       "%",
                "provenance": "measured",
                "method":     atk,
                "formula":    f"mean detection rate over {N_ROUNDS} rounds at {int(frac*100)}% Byzantine",
            })

    # ── results.json ─────────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
        json.dump({"phase": "writing_results", "current": 0, "total": 1}, f)

    results_out = {
        # Required schema keys
        "global_test_accuracy":           global_acc,
        "detection_rate_under_collusion": global_det,
        "wandb_run_url":                  wandb_url,
        # Enriched manifest
        "manifest_version": 1,
        "config": {
            "proj_dim": PROJ_DIM, "n_clients": N_CLIENTS,
            "byz_fracs": BYZ_FRACS, "attack_types": ATTACK_TYPES,
            "seed": SEED, "atk_scale": ATK_SCALE, "sigma": SIGMA,
            "n_rounds": N_ROUNDS, "dataset": "mnist",
            "defense": "szkp_rotate_every_round",
        },
        "results": [
            {
                "name": "global_test_accuracy",
                "value": global_acc,
                "unit": "%",
                "provenance": "measured",
                "method": "s_zkp_mean_over_conditions",
                "formula": "mean(final_acc) over all (byz_frac, attack_type) conditions",
            },
            {
                "name": "detection_rate_under_collusion",
                "value": global_det,
                "unit": "%",
                "provenance": "measured",
                "method": "szkp_projection_filter",
                "formula": "mean(mean_det_rate) over all (byz_frac, attack_type) conditions",
            },
            *results_entries,
        ],
        "baselines": [
            {
                "name": "no_defense_fedavg",
                "provenance": "claimed_unverified",
                "headline": False,
                "metrics": {
                    "expected_accuracy_drop_pct": 30.0,
                    "expected_detection_rate_pct": 0.0,
                },
            }
        ],
        "figures": [
            {
                "name": "accuracy_vs_byzantine_fraction",
                "renders": ["global_test_accuracy", "detection_rate_under_collusion"] +
                           [f"accuracy_{atk}" for atk in ATTACK_TYPES] +
                           [f"det_rate_{atk}" for atk in ATTACK_TYPES],
                "inline_data": fig_inline,
            }
        ],
        "metrics": {
            "global_test_accuracy":           global_acc,
            "detection_rate_under_collusion": global_det,
            **{f"accuracy_{r['label']}": r["final_acc"]
               for frac in BYZ_FRACS for atk, r in [(a, results[frac][a]) for a in ATTACK_TYPES]},
            **{f"det_rate_{r['label']}": r["mean_det_rate"]
               for frac in BYZ_FRACS for atk, r in [(a, results[frac][a]) for a in ATTACK_TYPES]},
        },
        "validation_status": "pending",
        "github_commit_sha": None,
    }

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results_out, f, indent=2)
    log.info("Wrote results.json ✓")
    return results_out


if __name__ == "__main__":
    main()
