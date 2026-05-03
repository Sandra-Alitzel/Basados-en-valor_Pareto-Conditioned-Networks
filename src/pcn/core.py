"""
Core data structures and algorithms for Pareto Conditioned Networks (PCN).

This module implements three building blocks of PCN
(Reymond et al., AAMAS 2022) adapted to continuous control:

1. **Trajectory Experience Buffer** (:class:`ExperienceBuffer`).
   Stores *full episodes*. Per-step multi-objective returns-to-go
   ``G_t = sum_{k>=t} r_k`` (no discount, as in the original paper) are
   computed once on insertion via a backwards pass, because PCN trains the
   policy via supervised behavioural cloning conditioned on those
   returns-to-go.

2. **Non-dominated sorting** (:func:`pareto_front_indices`,
   :func:`is_non_dominated`). Given the matrix of *episode* returns
   ``R in R^{N x d}`` (sums of step rewards over each stored trajectory)
   it extracts the indices of the Pareto-optimal trajectories under the
   "more is better" convention for every objective.

3. **Target-return sampling** (:func:`sample_target_return`). During
   training PCN conditions the actor on *desired* returns. To force the
   policy to "beat its own records" we sample a return vector from the
   empirical Pareto front of the buffer and *push it outwards* by a noise
   term proportional to the per-objective spread of the front.
"""
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
    """A full episode, with per-step return-to-go pre-computed.

    Attributes
    ----------
    transitions : list[Transition]
        Time-ordered transitions of the episode.
    episode_return : np.ndarray
        ``sum_t r_t``, shape ``(reward_dim,)``. This is the vector used
        for Pareto comparisons across trajectories.
    length : int
        Number of transitions (== episode horizon).
    """
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
    """Bounded FIFO buffer of complete trajectories.

    Parameters
    ----------
    capacity : int
        Maximum number of trajectories retained. Older trajectories are
        evicted FIFO when ``len(self) == capacity``.
    reward_dim : int
        Dimensionality of the multi-objective reward vector. Used to
        sanity-check inserts.

    Notes
    -----
    The original PCN paper keeps only the *non-dominated* trajectories in
    the buffer plus a small set of crowding-diverse points. This
    implementation stores everything FIFO and exposes
    :meth:`pareto_episode_indices` so callers can apply that policy on
    top -- this keeps the buffer reusable for ablations.
    """

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
        """Compute returns-to-go and store the trajectory.

        ``states[t]``, ``actions[t]`` and ``rewards[t]`` describe step
        ``t``; lengths must match. ``rewards[t]`` must be a vector of
        shape ``(reward_dim,)``.
        """
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
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Uniformly sample step-level transitions across stored episodes.

        Returned dict has keys ``state``, ``action``, ``return_to_go``,
        ``horizon_to_go``, each a stacked numpy array of size
        ``batch_size``. This is the supervised-learning batch consumed by
        PCN's behavioural-cloning loss.
        """
        if not self._traj:
            raise RuntimeError("Buffer is empty")
        rng = np.random.default_rng() if rng is None else rng

        # Build a flat index (traj_idx, step_idx) once and sample.
        lengths = np.array([t.length for t in self._traj], dtype=np.int64)
        cum = np.concatenate(([0], np.cumsum(lengths)))
        total = int(cum[-1])
        flat = rng.integers(0, total, size=batch_size)

        traj_idx = np.searchsorted(cum, flat, side="right") - 1
        step_idx = flat - cum[traj_idx]

        states, actions, rtg, htg = [], [], [], []
        traj_list = list(self._traj)
        for ti, si in zip(traj_idx, step_idx):
            tr = traj_list[ti].transitions[si]
            states.append(tr.state)
            actions.append(tr.action)
            rtg.append(tr.return_to_go)
            htg.append(tr.horizon_to_go)

        return {
            "state": np.stack(states, axis=0).astype(np.float32),
            "action": np.stack(actions, axis=0).astype(np.float32),
            "return_to_go": np.stack(rtg, axis=0).astype(np.float32),
            "horizon_to_go": np.asarray(htg, dtype=np.float32),
        }


# --------------------------------------------------------------------- #
# Pareto helpers
# --------------------------------------------------------------------- #
def is_non_dominated(returns: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows of ``returns``.

    Convention: **maximisation** on every objective. A point ``a`` is
    *dominated* by ``b`` iff ``b >= a`` componentwise and ``b > a`` on at
    least one component. The output mask is ``True`` for points that are
    **not dominated** by any other point in the set.

    Implementation is O(N^2) which is fine for the buffer sizes used by
    PCN (a few hundred trajectories at most).

    Parameters
    ----------
    returns : np.ndarray
        Shape ``(N, d)``. May be empty.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(N,)``.
    """
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
    """Sample a desired return vector to condition the next episode.

    Strategy (matches the PCN paper's "augmented" sampling):

    1. Pick a point ``R*`` from the empirical Pareto ``front``. If
       ``use_crowding=True`` the probability is proportional to the
       NSGA-II crowding distance, biasing exploration towards
       under-represented regions of the front; otherwise picks uniformly.
    2. Add Gaussian noise scaled by the per-objective spread of the
       front, then **clip from below** at ``R*`` so the sampled command
       *cannot be worse* than the chosen reference point on any
       objective. This is what forces the network to keep beating its
       own records.

    Parameters
    ----------
    front : np.ndarray
        Pareto front matrix, shape ``(M, d)``. Must be non-empty.
    noise_scale : float, optional
        Std of the additive noise as a fraction of the per-objective
        spread of the front. Use ``0.0`` to sample exactly on the front.
    use_crowding : bool, optional
        Use crowding-distance-weighted sampling (default ``True``).
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Target-return vector of shape ``(d,)``.
    """
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
                noise = np.abs(rng.normal(0.0, 1.0, size=front.shape[1])) * noise_scale 
        
    else:
        noise = np.zeros(front.shape[1], dtype=np.float64)

    target = reference + noise
    # Safety clip: target must dominate the reference (>= on every dim).
    target = np.maximum(target, reference)
    return target.astype(np.float32)
