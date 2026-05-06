"""
Hypervolume indicator for multi-objective evaluation.

Convention used throughout the project: **maximisation on every
objective**. The hypervolume of a Pareto front ``F`` w.r.t. a reference
point ``r`` is the Lebesgue measure of the region

    { y in R^d  :  exists p in F  s.t.  r <= y <= p }

i.e. the volume of the dominated region above the reference point. The
reference point must be *dominated* by every point of the front; in
practice we pick a vector slightly worse than the worst observed return
on each axis.

For ``d == 2`` we use a closed-form O(n log n) sweep. For higher
dimensions we delegate to ``pymoo`` if available, otherwise fall back to
a simple inclusion-exclusion that is correct (but exponential in ``d``).
"""
from __future__ import annotations

import numpy as np


def _hv_2d(front: np.ndarray, ref: np.ndarray) -> float:
    """Closed-form 2-D hypervolume (maximisation)."""
    # Keep only points that dominate the reference.
    mask = (front[:, 0] > ref[0]) & (front[:, 1] > ref[1])
    pts = front[mask]
    if pts.size == 0:
        return 0.0
    # Sort by obj0 descending, sweep on obj1.
    order = np.argsort(-pts[:, 0])
    pts = pts[order]
    hv = 0.0
    prev_y = ref[1]
    for x, y in pts:
        if y <= prev_y:
            continue
        hv += (x - ref[0]) * (y - prev_y)
        prev_y = y
    return float(hv)


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute the hypervolume of ``front`` w.r.t. ``ref_point``.

    Parameters
    ----------
    front : np.ndarray
        Pareto front matrix, shape ``(n, d)`` (maximisation convention).
        May be empty (returns 0).
    ref_point : np.ndarray
        Reference point of shape ``(d,)``. Must be component-wise
        worse than every point of ``front``.

    Returns
    -------
    float
        Non-negative hypervolume.
    """
    front = np.asarray(front, dtype=np.float64)
    ref = np.asarray(ref_point, dtype=np.float64)
    if front.size == 0:
        return 0.0
    if front.ndim != 2:
        raise ValueError(f"front must be 2-D, got shape {front.shape}")
    if ref.shape != (front.shape[1],):
        raise ValueError(
            f"ref_point shape {ref.shape} incompatible with front shape "
            f"{front.shape}"
        )

    if front.shape[1] == 2:
        return _hv_2d(front, ref)

    # ---- d >= 3 ----------------------------------------------------- #
    try:
        from pymoo.indicators.hv import HV  # type: ignore

        # pymoo minimises by convention -> negate everything.
        hv_indicator = HV(ref_point=-ref)
        return float(hv_indicator(-front))
    except Exception:
        # Inclusion-exclusion fallback (correctness over speed).
        n = front.shape[0]
        total = 0.0
        # Iterate over non-empty subsets.
        for mask in range(1, 1 << n):
            bits = [i for i in range(n) if mask & (1 << i)]
            mn = front[bits].min(axis=0)
            vol = np.prod(np.maximum(mn - ref, 0.0))
            sign = -1.0 if (len(bits) % 2 == 0) else 1.0
            total += sign * vol
        return float(max(total, 0.0))


def hypervolume_from_returns(
    returns: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """Convenience: compute the empirical Pareto front of ``returns``
    and return its hypervolume w.r.t. ``ref_point``.
    """
    from src.pcn import pareto_front_indices  # local import to avoid cycle

    returns = np.asarray(returns, dtype=np.float64)
    if returns.size == 0:
        return 0.0
    idx = pareto_front_indices(returns)
    front = returns[idx] if idx.size else returns
    return hypervolume(front, ref_point)
