"""Null-space poisoning attack implementation for SA-FL experiments.

The null-space attack exploits the absence of ZKP projection checks.
Malicious clients craft gradient updates that:
1. Lie close to the null space of the span of honest client gradients
2. Pass Pearson correlation checks (since the null component is orthogonal)
3. Carry a poison component that corrupts the global model

Without ZKP-based projection verification, the server cannot distinguish
these adversarial updates from honest ones.
"""

import numpy as np
import torch


def null_space_projection(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project a vector onto the null space of the given basis vectors.

    Args:
        vector: The vector to project (1D numpy array)
        basis: Matrix of basis vectors, shape (k, d) where k < d

    Returns:
        The component of vector that lies in the null space of basis.
        i.e., the part of vector NOT in the span of basis.
    """
    if basis.shape[0] == 0:
        return vector.copy()

    # Orthonormalize basis using QR decomposition
    if basis.shape[0] >= basis.shape[1]:
        # More basis vectors than dimensions — take only first d
        basis = basis[: basis.shape[1]]

    Q, _ = np.linalg.qr(basis.T)
    # Project onto span(basis)
    proj_span = Q @ (Q.T @ vector)
    # Null space component = original - span projection
    return vector - proj_span


def craft_null_space_update(
    honest_gradient: np.ndarray,
    previous_gradients: list[np.ndarray],
    poison_direction: np.ndarray,
    poison_scale: float = 3.0,
    warmup: bool = False,
) -> np.ndarray:
    """Craft a poisoned update that lies in the null space of honest gradients.

    The attack:
    1. Computes the null-space component of the poison w.r.t. recent honest gradients
    2. Adds this to the honest gradient so Pearson correlation is preserved
    3. Without ZKP checks, this passes the SA-FL defense

    Args:
        honest_gradient: The honest gradient the client would normally send
        previous_gradients: List of recent gradient updates from observed rounds
        poison_direction: Direction of the desired poison effect
        poison_scale: Magnitude of the poison component
        warmup: If True, return honest gradient (attack hasn't gathered enough info)
    """
    if warmup or len(previous_gradients) < 3:
        return honest_gradient.copy()

    # Stack recent gradients as basis
    basis = np.stack(previous_gradients[-10:], axis=0)

    # Project poison direction onto null space of recent gradient span
    null_component = null_space_projection(poison_direction, basis)

    # Normalize null component
    null_norm = np.linalg.norm(null_component)
    if null_norm < 1e-8:
        # Poison fully lies in span — use a random orthogonal direction instead
        rng = np.random.default_rng(seed=42)
        noise = rng.standard_normal(honest_gradient.shape)
        null_component = null_space_projection(noise, basis)
        null_norm = np.linalg.norm(null_component)
        if null_norm < 1e-8:
            return honest_gradient.copy()

    null_component = null_component / null_norm

    # Scale the null component to have the same magnitude as the honest gradient
    honest_norm = np.linalg.norm(honest_gradient)
    null_component = null_component * (honest_norm * poison_scale)

    return honest_gradient + null_component


class NullSpaceAttacker:
    """Manages the null-space attack state across FL rounds.

    In a real attack, the adversary would observe other clients' gradients.
    In this simulation, we use the aggregated gradients as a proxy for the
    honest gradient span (since no ZKP checks verify authenticity).
    """

    def __init__(
        self,
        poison_scale: float = 3.0,
        warmup_rounds: int = 5,
        target_class: int = 0,
    ):
        self.poison_scale = poison_scale
        self.warmup_rounds = warmup_rounds
        self.target_class = target_class
        self.gradient_history: list[np.ndarray] = []

    def record_round_gradient(self, gradient: np.ndarray) -> None:
        """Record an observed gradient for null space estimation."""
        self.gradient_history.append(gradient.copy())
        # Keep only last 15 rounds for efficiency
        if len(self.gradient_history) > 15:
            self.gradient_history = self.gradient_history[-15:]

    def poison(
        self,
        honest_gradient: np.ndarray,
        round_num: int,
        poison_direction: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Return poisoned gradient and whether attack was active this round."""
        if round_num < self.warmup_rounds:
            return honest_gradient.copy(), False

        if poison_direction is None:
            # Default poison: amplify gradient to push model toward uniform predictions
            rng = np.random.default_rng(seed=round_num)
            poison_direction = rng.standard_normal(honest_gradient.shape)

        poisoned = craft_null_space_update(
            honest_gradient=honest_gradient,
            previous_gradients=self.gradient_history,
            poison_direction=poison_direction,
            poison_scale=self.poison_scale,
            warmup=(round_num < self.warmup_rounds),
        )
        return poisoned, True
