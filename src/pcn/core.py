from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class Transition:
    """A single environment step inside a trajectory."""
    state: np.ndarray              # shape (obs_dim,)
    action: np.ndarray             # shape (action_dim,)
    reward: np.ndarray             # shape (reward_dim,)  -- vector reward
    return_to_go: np.ndarray = field(  # shape (reward_dim,)
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    horizon_to_go: int = 0         # remaining-steps command for PCN


@dataclass(slots=True)
class Trajectory:
   
    transitions: list[Transition]
    episode_return: np.ndarray
    length: int

    @property
    def reward_dim(self) -> int:
        return int(self.episode_return.shape[0])


# --------------------------------------------------------------------- #
# Experience buffer
# --------------------------------------------------------------------- #
class ExperienceBuffer:
    
    def __init__(self, capacity: int, reward_dim: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if reward_dim < 1:
            raise ValueError(f"reward_dim must be >= 1, got {reward_dim}")
        self.capacity = capacity
        self.reward_dim = reward_dim
        self._traj: deque[Trajectory] = deque(maxlen=capacity)

    # ------------------------------------------------------------------ #
    # Insertion
    # ------------------------------------------------------------------ #
    def add_trajectory(
        self,
        states: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
    ) -> Trajectory:
       
        T = len(rewards)
        if not (len(states) == len(actions) == T):
            raise ValueError(
                f"states/actions/rewards length mismatch: "
                f"{len(states)}/{len(actions)}/{T}"
            )
        if T == 0:
            raise ValueError("Cannot add an empty trajectory")

        # Backwards pass to compute returns-to-go (undiscounted as per PCN).
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        if rewards_arr.shape[1] != self.reward_dim:
            raise ValueError(
                f"reward dim mismatch: got {rewards_arr.shape[1]}, "
                f"expected {self.reward_dim}"
            )
        returns_to_go = np.empty_like(rewards_arr)
        running = np.zeros(self.reward_dim, dtype=np.float32)
        for t in range(T - 1, -1, -1):
            running = running + rewards_arr[t]
            returns_to_go[t] = running

        transitions: list[Transition] = []
        for t in range(T):
            transitions.append(
                Transition(
                    state=np.asarray(states[t], dtype=np.float32),
                    action=np.asarray(actions[t], dtype=np.float32),
                    reward=rewards_arr[t],
                    return_to_go=returns_to_go[t].copy(),
                    horizon_to_go=T - t,
                )
            )
        traj = Trajectory(
            transitions=transitions,
            episode_return=returns_to_go[0].copy(),  # = sum of rewards
            length=T,
        )
        self._traj.append(traj)
        return traj

    # ------------------------------------------------------------------ #
    # Inspection / sampling helpers
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._traj)

    def __iter__(self) -> Iterable[Trajectory]:
        return iter(self._traj)

    @property
    def episode_returns(self) -> np.ndarray:
        """Matrix of stored episode returns, shape ``(N, reward_dim)``."""
        if not self._traj:
            return np.zeros((0, self.reward_dim), dtype=np.float32)
        return np.stack([t.episode_return for t in self._traj], axis=0)

    def pareto_episode_indices(self) -> np.ndarray:
        """Indices (into the buffer) of Pareto-optimal trajectories."""
        return pareto_front_indices(self.episode_returns)

    def pareto_front(self) -> np.ndarray:
        """Return matrix of Pareto-optimal *episode returns*."""
        idx = self.pareto_episode_indices()
        return self.episode_returns[idx]

    def sample_transitions(
        self,
        batch_size: int,
        pareto_weight: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        
        if not self._traj:
            raise RuntimeError("Buffer is empty")
        rng = np.random.default_rng() if rng is None else rng
        traj_list = list(self._traj)

        def _flat_sample(indices: np.ndarray, n: int) -> list:
            lens = np.array([traj_list[i].length for i in indices], dtype=np.int64)
            c = np.concatenate(([0], np.cumsum(lens)))
            flat = rng.integers(0, int(c[-1]), size=n)
            local_ti = np.searchsorted(c, flat, side="right") - 1
            si = flat - c[local_ti]
            return [traj_list[indices[lt]].transitions[s]
                    for lt, s in zip(local_ti, si)]

        all_idx = np.arange(len(traj_list))

        if pareto_weight > 0.0:
            pareto_idx = self.pareto_episode_indices()
            if len(pareto_idx) > 0:
                n_pareto = max(1, round(batch_size * pareto_weight))
                n_rest = batch_size - n_pareto
                transitions = _flat_sample(pareto_idx, n_pareto)
                if n_rest > 0:
                    transitions += _flat_sample(all_idx, n_rest)
            else:
                transitions = _flat_sample(all_idx, batch_size)
        else:
            # Original: uniform flat sampling proportional to episode length.
            lengths = np.array([t.length for t in traj_list], dtype=np.int64)
            cum = np.concatenate(([0], np.cumsum(lengths)))
            flat = rng.integers(0, int(cum[-1]), size=batch_size)
            traj_idx = np.searchsorted(cum, flat, side="right") - 1
            step_idx = flat - cum[traj_idx]
            transitions = [traj_list[ti].transitions[si]
                           for ti, si in zip(traj_idx, step_idx)]

        return {
            "state": np.stack([t.state for t in transitions], axis=0).astype(np.float32),
            "action": np.stack([t.action for t in transitions], axis=0).astype(np.float32),
            "return_to_go": np.stack([t.return_to_go for t in transitions], axis=0).astype(np.float32),
            "horizon_to_go": np.asarray([t.horizon_to_go for t in transitions], dtype=np.float32),
        }


# --------------------------------------------------------------------- #
# Pareto helpers
# --------------------------------------------------------------------- #
def is_non_dominated(returns: np.ndarray) -> np.ndarray:
    
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2-D, got shape {returns.shape}")
    n = returns.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Vectorised dominance test:
    # dominated[i] is True iff exists j with returns[j] >= returns[i] (all dims)
    # and returns[j] > returns[i] (any dim).
    # Use broadcasting: a[i, j, k] = returns[j, k] >= returns[i, k].
    ge = returns[None, :, :] >= returns[:, None, :]   # (N, N, d)
    gt = returns[None, :, :] > returns[:, None, :]    # (N, N, d)
    dominates = ge.all(axis=-1) & gt.any(axis=-1)     # (i, j): j dominates i
    # Don't let a point dominate itself (numerically equal rows).
    np.fill_diagonal(dominates, False)
    dominated = dominates.any(axis=1)
    return ~dominated


def pareto_front_indices(returns: np.ndarray) -> np.ndarray:
    """Indices of Pareto-optimal rows of ``returns`` (maximisation)."""
    mask = is_non_dominated(returns)
    return np.flatnonzero(mask)


def crowding_distance(front: np.ndarray) -> np.ndarray:
    """Crowding distance of points on a Pareto front (NSGA-II style).

    Boundary points get ``+inf`` so they are always preferred when used
    to bias sampling towards diverse regions of the front.
    """
    front = np.asarray(front, dtype=np.float64)
    n, d = front.shape
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n <= 2:
        return np.full(n, np.inf, dtype=np.float64)

    dist = np.zeros(n, dtype=np.float64)
    for k in range(d):
        order = np.argsort(front[:, k])
        sorted_vals = front[order, k]
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = sorted_vals[-1] - sorted_vals[0]
        if span < 1e-12:
            continue
        dist[order[1:-1]] += (sorted_vals[2:] - sorted_vals[:-2]) / span
    return dist


# --------------------------------------------------------------------- #
# Target-return sampling
# --------------------------------------------------------------------- #
def sample_target_return(
    front: np.ndarray,
    noise_scale: float = 0.6,
    use_crowding: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
   
    front = np.asarray(front, dtype=np.float64)
    if front.ndim != 2 or front.shape[0] == 0:
        raise ValueError(
            f"front must be a non-empty 2-D array, got shape {front.shape}"
        )
    rng = np.random.default_rng() if rng is None else rng

    # ---- 1. choose a reference point on the front -------------------- #
    if use_crowding and front.shape[0] > 2:
        cd = crowding_distance(front)
        # Replace +inf by max-finite so probabilities are well defined,
        # while still strongly favouring boundary points.
        finite = cd[np.isfinite(cd)]
        big = (finite.max() * 2.0) if finite.size > 0 else 1.0
        weights = np.where(np.isfinite(cd), cd, big)
        weights = weights + 1e-8
        probs = weights / weights.sum()
        idx = int(rng.choice(front.shape[0], p=probs))
    else:
        idx = int(rng.integers(0, front.shape[0]))
    reference = front[idx].copy()

    # ---- 2. push the command outward --------------------------------- #
    if noise_scale > 0.0:
        spread = front.max(axis=0) - front.min(axis=0)
        # When the front is degenerate (single point or very narrow), fall back
        # to a fraction of the mean return magnitude so the push is always
        # meaningful regardless of absolute scale.
        min_spread = np.abs(front).mean(axis=0) * 0.1 + 1.0
        effective_spread = np.maximum(spread, min_spread)
        noise = np.abs(rng.normal(0.0, 1.0, size=front.shape[1])) * (noise_scale * effective_spread)
    else:
        noise = np.zeros(front.shape[1], dtype=np.float64)

    target = reference + noise
    # Safety clip: target must dominate the reference (>= on every dim).
    target = np.maximum(target, reference)
    return target.astype(np.float32)
