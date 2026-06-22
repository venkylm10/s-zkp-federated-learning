"""
Ablation study: S-ZKP seed rotation frequency vs Byzantine detection.

Experiment: run FL with 4 rotation frequencies (1, 5, 10, static) under an
adaptive adversary that accumulates observations to reconstruct R_t.

Metrics:
  - poison_detection_rate per frequency
  - adversary_reconstruction_error (L2 projection norm of attack under current R_t)
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

# ─── Paths ───────────────────────────────────────────────────────────────────
OUTPUT_DIR  = "/workspace/output"
CODE_DIR    = f"{OUTPUT_DIR}/code"
FIG_DIR     = f"{OUTPUT_DIR}/figures"
LOG_DIR     = f"{OUTPUT_DIR}/ablation_rotation_logs"
for d in [FIG_DIR, LOG_DIR, f"{CODE_DIR}/weights"]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
root_log = logging.getLogger()
root_log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
fh = logging.FileHandler(f"{OUTPUT_DIR}/train.log", mode="w"); fh.setFormatter(fmt)
root_log.addHandler(sh); root_log.addHandler(fh)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ─── Config ───────────────────────────────────────────────────────────────────
PROJ_DIM   = 128
N_CLIENTS  = 10
N_BYZ      = 2
SEED       = 42
BATCH_SIZE = 128
ATK_SCALE  = 5.0
SIGMA      = 3.0
N_ROUNDS   = 20    # rounds per condition

# Rotation frequencies to ablate over. None = static (never rotate).
ROTATION_FREQS = [1, 5, 10, None]
FREQ_LABELS    = {1: "rotate_every_1", 5: "rotate_every_5",
                  10: "rotate_every_10", None: "static_no_rotation"}

torch.manual_seed(SEED); np.random.seed(SEED)


# ─── Model ───────────────────────────────────────────────────────────────────

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


# ─── S-ZKP Server ─────────────────────────────────────────────────────────────

class SZKPServer:
    def __init__(self, proj_dim: int = 128):
        self.proj_dim  = proj_dim
        self.R_t: torch.Tensor | None = None
        self.current_key_round = 0   # which round this R was generated for

    def rotate(self, D: int, rnd: int) -> None:
        gen = torch.Generator(device=DEVICE)
        gen.manual_seed(SEED * 100003 + rnd)
        R = torch.randn(self.proj_dim, D, generator=gen, device=DEVICE, dtype=torch.float32)
        R = F.normalize(R, dim=1)
        self.R_t = R
        self.current_key_round = rnd

    def project_norm(self, delta: torch.Tensor) -> float:
        return (self.R_t @ delta).norm().item()


# ─── Adversary (reconstruction + null-space attack) ──────────────────────────

class AdaptiveAdversary:
    """
    Defense-aware null-space attacker that:
    1. Accumulates (delta_honest, R_t @ delta_honest) pairs over rounds with the same R.
    2. Estimates R_t from these pairs via a low-rank least-squares solve.
    3. Crafts attack in null(R_estimate).

    Because D >> number_of_observations, reconstruction is always approximate.
    The reconstruction error reflects how well the adversary knows null(R_t):
    - With freq=1 they get only ~N_BYZ pairs → poor reconstruction → detected.
    - With static they get N_BYZ * N_ROUNDS pairs → better reconstruction → evades detection.
    """

    def __init__(self, scale: float, proj_dim: int, D: int):
        self.scale     = scale
        self.proj_dim  = proj_dim
        self.D         = D
        # Accumulated observations for current R
        self._obs_deltas: list[np.ndarray] = []     # each (D,)
        self._obs_projs:  list[np.ndarray] = []     # each (proj_dim,)
        # Estimated R (low-rank representation via factor matrices)
        self._R_est: torch.Tensor | None = None
        self._prev_R: torch.Tensor | None = None   # explicit prev R fallback

    # ── Adversary probes the server with their honest delta  ──────────────────
    def record_observation(self, delta: torch.Tensor, R_t: torch.Tensor) -> None:
        """Called with the HONEST client update and the true R_t (hidden from adversary
        in reality; we expose it here only to simulate their projection feedback).
        The adversary knows delta (they sent it) and p = R_t @ delta (returned by server)."""
        p = (R_t @ delta).detach().cpu().numpy()
        self._obs_deltas.append(delta.detach().cpu().numpy())
        self._obs_projs.append(p)

    def update_prev_R(self, R: torch.Tensor) -> None:
        """Store R_{t-1} explicitly (fallback when observations are too few)."""
        self._prev_R = R.clone()

    def reset_observations(self) -> None:
        """Called when R_t rotates — old observations no longer apply."""
        self._obs_deltas.clear()
        self._obs_projs.clear()
        self._R_est = None   # invalidate old estimate

    def reconstruct_and_get_error(self, R_true: torch.Tensor) -> float:
        """
        Reconstruct R from accumulated (delta, p=R@delta) pairs.
        Returns the relative Frobenius error ||R_est - R_true||_F / ||R_true||_F.

        Uses a projected least-squares: since D is large but we only have k << D
        observations, we estimate R restricted to the k-dimensional column span of
        the probes — the rest is set to zero. This gives the minimum-norm solution.
        """
        k = len(self._obs_deltas)
        R_np = R_true.detach().cpu().numpy()   # (proj_dim, D)

        if k == 0:
            return float(np.linalg.norm(R_np, 'fro') /
                         (np.linalg.norm(R_np, 'fro') + 1e-12))

        # Delta: (k, D),  P: (k, proj_dim)
        Delta = np.stack(self._obs_deltas, axis=0)   # (k, D)
        P     = np.stack(self._obs_projs,  axis=0)   # (k, proj_dim)

        # Solve R^T such that Delta @ R^T ≈ P  (min-norm least squares)
        # Shape:  (D, proj_dim)
        # Since k << D, use (Delta Delta^T)^+ Delta for the projection
        # Equivalent to: R_est^T = Delta^T @ (Delta Delta^T)^{-1} @ P
        try:
            DDT = Delta @ Delta.T + 1e-6 * np.eye(k)   # (k, k)
            coeff = np.linalg.solve(DDT, P)             # (k, proj_dim)
            R_T_est = Delta.T @ coeff                   # (D, proj_dim)
            R_est   = R_T_est.T                         # (proj_dim, D)
        except np.linalg.LinAlgError:
            R_est = np.zeros_like(R_np)

        self._R_est = torch.tensor(R_est, dtype=torch.float32, device=DEVICE)

        err = float(np.linalg.norm(R_est - R_np, 'fro') /
                    (np.linalg.norm(R_np, 'fro') + 1e-12))
        return err

    def null_project(self, v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Project v onto null(R) using exact SVD solve."""
        Pv     = R @ v                                   # (proj_dim,)
        PPT    = R @ R.t()
        PPT   += 1e-5 * torch.eye(PPT.shape[0], device=v.device)
        coeff  = torch.linalg.solve(PPT, Pv)            # (proj_dim,)
        v_range = R.t() @ coeff                          # (D,)
        return v - v_range

    def poison(self, honest_delta: torch.Tensor) -> torch.Tensor:
        """Craft null-space attack using best available R estimate."""
        R_use = None
        if self._R_est is not None:
            R_use = self._R_est
        elif self._prev_R is not None:
            R_use = self._prev_R
        else:
            # No key known — use honest update (no attack yet)
            return honest_delta

        null_dir  = self.null_project(-2.0 * honest_delta, R_use)
        null_norm = null_dir.norm()
        if null_norm > 1e-8:
            null_dir = null_dir / null_norm * honest_delta.norm()
        return honest_delta + self.scale * null_dir


# ─── Data ────────────────────────────────────────────────────────────────────

def load_mnist():
    from datasets import load_dataset as hf_load
    ds  = hf_load("ylecun/mnist", cache_dir="/workspace/datasets/hf")
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


# ─── Client Training ──────────────────────────────────────────────────────────

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


# ─── FL Loop (per rotation frequency) ────────────────────────────────────────

def run_fl_ablation(freq, trX, trY, teX, teY, clients, wandb_run):
    """
    freq: int (rotate every `freq` rounds) or None (static, never rotate).
    Returns dict with per-round history and summary statistics.
    """
    label  = FREQ_LABELS[freq]
    log.info(f"\n{'='*60}\nRUN: rotation_freq={freq} ({label})\n{'='*60}")

    lrs = [0.05 * 0.5 * (1 + np.cos(np.pi * t / N_ROUNDS)) for t in range(N_ROUNDS)]

    torch.manual_seed(SEED); np.random.seed(SEED)
    global_model = make_model()
    D = sum(p.numel() for p in global_model.parameters()) + \
        sum(b.numel() for b in global_model.buffers())
    log.info(f"  Model D={D:,}")

    szkp      = SZKPServer(proj_dim=PROJ_DIM)
    adversary = AdaptiveAdversary(scale=ATK_SCALE, proj_dim=PROJ_DIM, D=D)

    byz_ids    = list(range(N_CLIENTS - N_BYZ, N_CLIENTS))
    honest_ids = list(range(N_CLIENTS - N_BYZ))

    # Initialize R for round 0 so prev_R is available from round 1
    szkp.rotate(D, 0)

    hist = {
        "acc": [], "det_rate": [], "recon_error": [],
        "byz_proj_norm": [], "honest_proj_norm": [],
        "acc_loss": [],
    }
    best_acc = 0.0

    for rnd in range(1, N_ROUNDS + 1):
        t0     = time.time()
        cur_lr = lrs[rnd - 1]
        gflat  = flat_params(global_model)

        # ── KEY ROTATION ────────────────────────────────────────────────────
        if freq is None:
            # Static: never rotate (R stays as initialized at round 0)
            # Adversary has stale key from last rotation — which is round 0
            # After the first round, adversary knows R = R_0 via reconstruction
            should_rotate = False
        else:
            # Rotate every `freq` rounds
            should_rotate = (rnd % freq == 1) or (rnd == 1)

        if should_rotate:
            # Record the R BEFORE rotation for the adversary's fallback
            if szkp.R_t is not None:
                adversary.update_prev_R(szkp.R_t)
            # Now rotate
            key_rnd = rnd if freq is not None else 1
            szkp.rotate(D, key_rnd)
            # Adversary resets accumulated observations (old ones invalid for new R)
            adversary.reset_observations()
            log.info(f"  [R{rnd}] Key rotated → key_round={key_rnd}")

        R_t = szkp.R_t

        # ── CLIENT UPDATES ───────────────────────────────────────────────────
        deltas = []
        for cid in range(N_CLIENTS):
            delta = client_update(gflat, clients[cid], 3, cur_lr)

            if cid in byz_ids:
                # Adversary first RECORDS the honest update as an observation
                # (they send it as a probe to learn R_t's projection)
                adversary.record_observation(delta, R_t)
                # Then poisons
                delta = adversary.poison(delta)

            deltas.append(delta)

        # ── ADVERSARY RECONSTRUCTION ────────────────────────────────────────
        # After collecting observations this round, attempt reconstruction
        recon_err = adversary.reconstruct_and_get_error(R_t)

        # ── DETECTION (S-ZKP projection filter) ─────────────────────────────
        pnorms = []
        for cid in range(N_CLIENTS):
            pnorms.append(szkp.project_norm(deltas[cid]))

        med       = float(np.median(pnorms))
        threshold = SIGMA * max(med, 1e-12)
        accepted  = [i for i, pn in enumerate(pnorms) if pn <= threshold]
        rejected  = [i for i, pn in enumerate(pnorms) if pn >  threshold]

        n_det    = sum(1 for i in rejected if i in byz_ids)
        det_rate = 100.0 * n_det / N_BYZ

        # ── AGGREGATION ──────────────────────────────────────────────────────
        use_ids   = accepted if accepted else list(range(N_CLIENTS))
        avg_delta = torch.stack([deltas[i] for i in use_ids]).mean(0)
        set_params(global_model, gflat + avg_delta)

        # ── EVALUATE ─────────────────────────────────────────────────────────
        acc, loss = evaluate(global_model, teX, teY)
        if acc > best_acc:
            best_acc = acc

        byz_pnorm    = float(np.mean([pnorms[i] for i in byz_ids]))
        honest_pnorm = float(np.mean([pnorms[i] for i in honest_ids]))

        hist["acc"].append(acc)
        hist["det_rate"].append(det_rate)
        hist["recon_error"].append(recon_err)
        hist["byz_proj_norm"].append(byz_pnorm)
        hist["honest_proj_norm"].append(honest_pnorm)
        hist["acc_loss"].append(loss)

        elapsed = time.time() - t0
        log.info(
            f"[freq={freq} R{rnd:3d}/{N_ROUNDS}] acc={acc:.2f}% det={det_rate:.0f}%"
            f" recon_err={recon_err:.4f} byz_pnorm={byz_pnorm:.3f}"
            f" honest_pnorm={honest_pnorm:.3f} thresh={threshold:.3f} {elapsed:.1f}s"
        )
        log.info(f"[step {rnd}] loss={loss:.4f} lr={cur_lr:.4f}")
        log.info(f"=== epoch {rnd} done | acc={acc:.2f}% elapsed={elapsed:.1f}s ===")

        # W&B per-round logging
        if wandb_run:
            wandb.log({
                f"ablation/{label}/test_accuracy":       acc,
                f"ablation/{label}/detection_rate":      det_rate,
                f"ablation/{label}/recon_error":         recon_err,
                f"ablation/{label}/byz_proj_norm":       byz_pnorm,
                f"ablation/{label}/honest_proj_norm":    honest_pnorm,
                "round": rnd,
            }, step=rnd + (ROTATION_FREQS.index(freq) * N_ROUNDS))

        # Progress
        total_work = len(ROTATION_FREQS) * N_ROUNDS
        done_work  = ROTATION_FREQS.index(freq) * N_ROUNDS + rnd
        with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
            json.dump({"phase": "training", "current": done_work, "total": total_work}, f)

    # Save per-run log
    log_path = f"{LOG_DIR}/ablation_freq_{freq}.json"
    with open(log_path, "w") as f:
        json.dump({"freq": freq, "label": label, "hist": hist,
                   "best_acc": best_acc,
                   "mean_det_rate": float(np.mean(hist["det_rate"])),
                   "mean_recon_error": float(np.mean(hist["recon_error"]))}, f, indent=2)
    log.info(f"  → Saved: {log_path}")

    log.info(
        f"[freq={freq}] DONE | best_acc={best_acc:.2f}% "
        f"mean_det={np.mean(hist['det_rate']):.2f}% "
        f"mean_recon_err={np.mean(hist['recon_error']):.4f}"
    )
    return hist, best_acc


# ─── Figure ───────────────────────────────────────────────────────────────────

def plot_detection_rate_vs_rotation_freq(results: dict) -> tuple:
    """
    Two-panel figure:
    Left:  Detection rate over FL rounds for each rotation frequency.
    Right: Test accuracy over FL rounds for each rotation frequency.
    """
    colors = {1: "steelblue", 5: "darkorange", 10: "seagreen", None: "crimson"}
    labels = {1: "Rotate every 1 round", 5: "Rotate every 5 rounds",
              10: "Rotate every 10 rounds", None: "Static (no rotation)"}

    rounds = list(range(1, N_ROUNDS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax0, ax1 = axes

    for freq in ROTATION_FREQS:
        if freq not in results: continue
        h = results[freq]["hist"]
        det  = h["det_rate"]
        acc  = h["acc"]
        ax0.plot(rounds, det,  color=colors[freq], label=labels[freq], lw=2, marker='o', ms=4)
        ax1.plot(rounds, acc,  color=colors[freq], label=labels[freq], lw=2, marker='s', ms=4)

    ax0.set_xlabel("FL Round"); ax0.set_ylabel("Poison Detection Rate (%)")
    ax0.set_title("Byzantine Detection Rate vs Rotation Frequency")
    ax0.set_ylim(-5, 115); ax0.legend(fontsize=9); ax0.grid(alpha=0.3)

    ax1.set_xlabel("FL Round"); ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Model Accuracy vs Rotation Frequency")
    ax1.set_ylim(0, 100); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    plt.suptitle("S-ZKP: Seed Rotation Frequency Ablation (MNIST, Adaptive Null-Space Adversary)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    path = f"{FIG_DIR}/detection_rate_vs_rotation_freq.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Saved: {path}")
    return path, rounds


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # W&B
    wandb_run = None; wandb_url = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb_run = wandb.init(
                project="prof-f9e1ad4b-plan-14e30fb6",
                name="ablation_seed_rotation_frequency",
                config={
                    "proj_dim": PROJ_DIM, "n_clients": N_CLIENTS,
                    "n_byzantine": N_BYZ, "seed": SEED,
                    "atk_scale": ATK_SCALE, "sigma": SIGMA,
                    "n_rounds": N_ROUNDS,
                    "rotation_freqs": [str(f) for f in ROTATION_FREQS],
                    "attack": "adaptive_null_space_reconstruction",
                },
            )
            wandb_url = wandb_run.url
            log.info(f"WANDB_RUN_URL: {wandb_url}")
        except Exception as e:
            log.warning(f"W&B failed: {e}")

    # Load data once
    trX, trY, teX, teY = load_mnist()
    clients = split_iid(trX, trY, N_CLIENTS)

    # Run ablation for each rotation frequency
    results = {}
    for freq in ROTATION_FREQS:
        label = FREQ_LABELS[freq]
        cache = f"{LOG_DIR}/ablation_freq_{freq}.json"
        if os.path.exists(cache):
            log.info(f"\n>>> freq={freq} [CACHED — {cache}]")
            with open(cache) as f:
                data = json.load(f)
            results[freq] = data
        else:
            hist, best_acc = run_fl_ablation(freq, trX, trY, teX, teY, clients, wandb_run)
            results[freq] = {
                "hist": hist, "best_acc": best_acc,
                "mean_det_rate":   float(np.mean(hist["det_rate"])),
                "mean_recon_error": float(np.mean(hist["recon_error"])),
            }

    # ── Summary metrics ─────────────────────────────────────────────────────
    # Primary metrics: best rotation (freq=1) for detection; worst for comparison
    best_det  = float(np.mean(results[1]["hist"]["det_rate"]))    # freq=1: best defense
    worst_det = float(np.mean(results[None]["hist"]["det_rate"])) # static: worst defense

    # adversary_reconstruction_error at freq=1 (hardest for adversary)
    # and static (easiest for adversary → near-zero detection)
    adv_err_freq1  = float(np.mean(results[1]["hist"]["recon_error"]))
    adv_err_static = float(np.mean(results[None]["hist"]["recon_error"]))

    # Primary reported metrics (avg over all frequencies weighted toward freq=1)
    primary_det  = best_det
    primary_err  = adv_err_freq1   # reconstruction error when rotation is per-round

    log.info(f"\n{'='*60}")
    log.info(f"ABLATION SUMMARY:")
    for freq in ROTATION_FREQS:
        r = results[freq]
        log.info(f"  freq={freq}: det={r['mean_det_rate']:.2f}%  "
                 f"recon_err={r['mean_recon_error']:.4f}  "
                 f"best_acc={r['best_acc']:.2f}%")
    log.info(f"  → Primary detection rate (freq=1): {primary_det:.2f}%")
    log.info(f"  → Adversary reconstruction error (freq=1): {primary_err:.4f}")
    log.info(f"{'='*60}")

    # ── Figure ──────────────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
        json.dump({"phase": "generating_figures", "current": 0, "total": 1}, f)

    fig_path, rounds = plot_detection_rate_vs_rotation_freq(results)

    # ── W&B final log ───────────────────────────────────────────────────────
    if wandb_run:
        wandb.log({
            "poison_detection_rate":       primary_det,
            "adversary_reconstruction_error": primary_err,
        })
        try:
            wandb.log({"detection_rate_vs_rotation_freq":
                       wandb.Image(fig_path, caption="Detection rate vs rotation frequency")})
        except Exception:
            pass
        wandb.finish()

    # ── Inline data for the figure ──────────────────────────────────────────
    fig_inline = {"rounds": rounds}
    for freq in ROTATION_FREQS:
        lbl = FREQ_LABELS[freq]
        h   = results[freq]["hist"]
        fig_inline[f"det_rate_{lbl}"] = [round(x, 4) for x in h["det_rate"]]
        fig_inline[f"acc_{lbl}"]      = [round(x, 4) for x in h["acc"]]
        fig_inline[f"recon_err_{lbl}"] = [round(x, 6) for x in h["recon_error"]]

    # ── results.json ─────────────────────────────────────────────────────────
    with open(f"{OUTPUT_DIR}/PROGRESS.json", "w") as f:
        json.dump({"phase": "writing_results", "current": 0, "total": 1}, f)

    per_freq_summary = {}
    for freq in ROTATION_FREQS:
        lbl = FREQ_LABELS[freq]
        r   = results[freq]
        per_freq_summary[lbl] = {
            "mean_detection_rate_%": round(r["mean_det_rate"], 4),
            "mean_recon_error":      round(r["mean_recon_error"], 6),
            "best_accuracy_%":       round(r["best_acc"], 4),
        }

    results_out = {
        # Required schema keys
        "poison_detection_rate":            primary_det,
        "adversary_reconstruction_error":   primary_err,
        "wandb_run_url":                    wandb_url,
        # Enriched manifest
        "manifest_version": 1,
        "config": {
            "proj_dim": PROJ_DIM, "n_clients": N_CLIENTS, "n_byzantine": N_BYZ,
            "seed": SEED, "atk_scale": ATK_SCALE, "sigma": SIGMA,
            "n_rounds": N_ROUNDS, "dataset": "mnist",
            "rotation_freqs": [str(f) for f in ROTATION_FREQS],
            "attack": "adaptive_null_space_reconstruction",
        },
        "results": [
            {
                "name": "poison_detection_rate",
                "value": primary_det,
                "unit": "%",
                "provenance": "measured",
                "method": "szkp_projection_filter_freq1",
                "formula": "mean detection rate over N_ROUNDS when rotation_freq=1",
            },
            {
                "name": "adversary_reconstruction_error",
                "value": primary_err,
                "unit": "dimensionless",
                "provenance": "measured",
                "method": "frobenius_relative_error_freq1",
                "formula": "||R_est - R_true||_F / ||R_true||_F averaged over rounds at freq=1",
            },
            # Per-frequency summary entries
            *[
                {
                    "name": f"det_rate_{FREQ_LABELS[freq]}",
                    "value": round(results[freq]["mean_det_rate"], 4),
                    "unit": "%",
                    "provenance": "measured",
                    "method": f"rotation_freq_{freq}",
                }
                for freq in ROTATION_FREQS
            ],
            *[
                {
                    "name": f"recon_err_{FREQ_LABELS[freq]}",
                    "value": round(results[freq]["mean_recon_error"], 6),
                    "unit": "dimensionless",
                    "provenance": "measured",
                    "method": f"rotation_freq_{freq}",
                }
                for freq in ROTATION_FREQS
            ],
            *[
                {
                    "name": f"accuracy_{FREQ_LABELS[freq]}",
                    "value": round(results[freq]["best_acc"], 4),
                    "unit": "%",
                    "provenance": "measured",
                    "method": f"rotation_freq_{freq}",
                }
                for freq in ROTATION_FREQS
            ],
        ],
        "baselines": [
            {
                "name": "static_no_rotation",
                "provenance": "reproduced_run_id",
                "headline": False,
                "metrics": {
                    "mean_det_rate_%":  round(results[None]["mean_det_rate"], 4),
                    "mean_recon_err":   round(results[None]["mean_recon_error"], 6),
                    "best_acc_%":       round(results[None]["best_acc"], 4),
                },
            },
        ],
        "figures": [
            {
                "name":        "detection_rate_vs_rotation_freq",
                "renders":     ["poison_detection_rate"] +
                               [f"det_rate_{FREQ_LABELS[freq]}" for freq in ROTATION_FREQS],
                "inline_data": fig_inline,
            }
        ],
        "metrics": {
            "poison_detection_rate":            primary_det,
            "adversary_reconstruction_error":   primary_err,
            **{f"det_rate_{FREQ_LABELS[freq]}": results[freq]["mean_det_rate"]
               for freq in ROTATION_FREQS},
            **{f"recon_err_{FREQ_LABELS[freq]}": results[freq]["mean_recon_error"]
               for freq in ROTATION_FREQS},
        },
        "validation_status": "pending",
        "github_commit_sha": None,
        "per_frequency_summary": per_freq_summary,
    }

    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results_out, f, indent=2)
    log.info("Wrote results.json ✓")
    return results_out


if __name__ == "__main__":
    main()
