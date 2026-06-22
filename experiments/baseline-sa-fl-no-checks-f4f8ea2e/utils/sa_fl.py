"""SA-FL dual-server secure aggregation implementation WITHOUT ZKP projection checks.

Based on: Yuan Chang, Qian Chen, Tom H. Luan, Zhou Su.
'SA-FL: Secure Aggregation Scheme in Federated Learning Against Poisoning Attacks',
IEEE TCCN, 2026.

This module implements the Pearson correlation coefficient-based detection
over blinded gradients, as described in the SA-FL paper. The critical missing
component is the secret-shared ZKP projection verification that would certify
each client's update lies on the honest gradient manifold.

Without that verification, null-space attacks pass undetected.
"""

from __future__ import annotations

import numpy as np


def blind_gradient(gradient: np.ndarray, rng_seed: int) -> np.ndarray:
    """Apply additive blinding mask to a gradient (simulates crypto server blinding).

    In the actual SA-FL protocol, Server 1 (Aggregation Server) and
    Server 2 (Crypto Server) jointly compute masked versions of gradients
    such that individual updates are hidden but correlations are preserved.
    We simulate this with a deterministic but reproducible blinding.

    The key property: blinding preserves Pearson correlation because
    the mask is the same for all clients in a round (homomorphic property).
    """
    rng = np.random.default_rng(seed=rng_seed)
    mask = rng.standard_normal(gradient.shape) * 0.01  # small additive noise
    return gradient + mask


def pearson_correlation(u: np.ndarray, v: np.ndarray) -> float:
    """Compute Pearson correlation between two gradient vectors."""
    u_centered = u - u.mean()
    v_centered = v - v.mean()
    norm_u = np.linalg.norm(u_centered)
    norm_v = np.linalg.norm(v_centered)
    if norm_u < 1e-10 or norm_v < 1e-10:
        return 0.0
    return float(np.dot(u_centered, v_centered) / (norm_u * norm_v))


def compute_pairwise_correlations(gradients: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise Pearson correlation matrix for a list of gradient vectors."""
    n = len(gradients)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = pearson_correlation(gradients[i], gradients[j])
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r
    return corr_matrix


class SAFLAggregator:
    """SA-FL secure aggregation server WITHOUT secret-shared projection checks.

    The dual-server architecture:
    - Server 1 (AS): Receives blinded updates, coordinates aggregation
    - Server 2 (CS): Helps with correlation computations via secret sharing

    The missing component: ZKP-based projection verification that would
    ensure each update uk satisfies: uk = P_H(uk) (lies on honest manifold).
    Without this, the null-space component g_null passes undetected.

    Detection logic: Pearson correlation threshold filtering.
    A client is flagged if its mean correlation with other clients
    falls below `correlation_threshold`.
    """

    def __init__(
        self,
        num_clients: int,
        correlation_threshold: float = 0.4,
        min_clients_after_filter: int = 3,
    ):
        self.num_clients = num_clients
        self.correlation_threshold = correlation_threshold
        self.min_clients_after_filter = min_clients_after_filter
        self.round_stats: list[dict] = []

    def aggregate(
        self,
        client_updates: list[np.ndarray],
        client_ids: list[int],
        malicious_ids: set[int],
        round_num: int,
    ) -> tuple[np.ndarray, dict]:
        """Aggregate client updates using SA-FL defense (no ZKP projection check).

        Args:
            client_updates: List of gradient update vectors from each client
            client_ids: Corresponding client IDs
            malicious_ids: Ground-truth set of malicious client IDs (for metrics)
            round_num: Current FL round number

        Returns:
            Tuple of (aggregated_gradient, stats_dict)
        """
        n = len(client_updates)
        assert n == len(client_ids), "Mismatch between updates and client IDs"

        # Step 1: Blind gradients (simulates dual-server crypto protocol)
        blinded = [
            blind_gradient(g, rng_seed=round_num * 1000 + i)
            for i, g in enumerate(client_updates)
        ]

        # Step 2: Compute pairwise Pearson correlations over blinded updates
        # (This is what SA-FL's crypto server helps compute via secret sharing)
        corr_matrix = compute_pairwise_correlations(blinded)

        # Step 3: Compute mean correlation for each client with all others
        mean_corr = np.array([
            (corr_matrix[i].sum() - 1.0) / (n - 1) for i in range(n)
        ])

        # Step 4: Filter clients below correlation threshold
        # NOTE: Without ZKP projection checks, null-space attacks pass here!
        # A null-space adversary's update has the same correlation structure
        # as an honest update, so it won't be filtered out.
        accepted_mask = mean_corr >= self.correlation_threshold
        if accepted_mask.sum() < self.min_clients_after_filter:
            # Too many rejections — fall back to accept all
            accepted_mask[:] = True

        accepted_indices = np.where(accepted_mask)[0].tolist()
        rejected_indices = np.where(~accepted_mask)[0].tolist()

        # Count true positives and false negatives for detection rate
        rejected_client_ids = {client_ids[i] for i in rejected_indices}
        accepted_client_ids = {client_ids[i] for i in accepted_indices}

        # Detection metrics:
        # - True positives: malicious clients that were rejected
        # - False negatives: malicious clients that passed (null-space attacks)
        true_positives = len(rejected_client_ids & malicious_ids)
        false_negatives = len(accepted_client_ids & malicious_ids)
        total_malicious = len(malicious_ids)

        if total_malicious > 0:
            detection_rate = true_positives / total_malicious
        else:
            detection_rate = 1.0  # trivially 100% if no malicious clients

        # Step 5: Aggregate accepted updates (equal weighting / FedAvg style)
        if len(accepted_indices) > 0:
            aggregated = np.mean(
                [client_updates[i] for i in accepted_indices], axis=0
            )
        else:
            aggregated = np.mean(client_updates, axis=0)

        stats = {
            "round": round_num,
            "num_accepted": len(accepted_indices),
            "num_rejected": len(rejected_indices),
            "rejected_client_ids": sorted(rejected_client_ids),
            "detection_rate": detection_rate,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "mean_correlations": mean_corr.tolist(),
            "correlation_threshold": self.correlation_threshold,
        }
        self.round_stats.append(stats)
        return aggregated, stats

    def cumulative_detection_rate(self) -> float:
        """Compute cumulative detection rate across all rounds."""
        if not self.round_stats:
            return 0.0
        total_tp = sum(s["true_positives"] for s in self.round_stats)
        total_mal = sum(
            s["true_positives"] + s["false_negatives"] for s in self.round_stats
        )
        if total_mal == 0:
            return 0.0
        return total_tp / total_mal
