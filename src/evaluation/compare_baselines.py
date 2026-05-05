"""
Rigorous benchmark of our PCN implementation against PGMORL.

Pipeline
--------
1. Train *our* PCN with ``N_SEEDS`` independent seeds. For each seed the
   training output is redirected to ``results/benchmark_seed_<S>/`` so
   the global ``results/training_log.csv`` of regular runs is never
   overwritten.
2. Train PGMORL (``morl_baselines.multi_policy.pgmorl``) with the same
   seeds, again into per-seed isolated directories.
3. Compute the hypervolume of the final empirical Pareto front of every
   run.
4. Plot a comparative box-plot (``boxplot_hv.png``) and a scatter of the
   Pareto fronts (``pareto_fronts.png``). The scatter is robust to the
   degenerate case of a front with only one point.
5. Run a Wilcoxon rank-sum test (``scipy.stats.ranksums``) on the per-seed
   hypervolumes and write the verdict to ``stats_report.txt``.

Run as::

    python -m src.evaluation.compare_baselines --n-seeds 20

The script is **idempotent**: each seed's directory is independent, so
a crash mid-benchmark does not corrupt previously-completed runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / cluster safe
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from scipy import stats           # noqa: E402

# Make ``src``/repo root importable when invoked as ``python ...``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.hypervolume import hypervolume    # noqa: E402
from src.pcn import pareto_front_indices              # noqa: E402


# --------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------- #
DEFAULT_BENCHMARK_CFG: dict[str, Any] = {
    "env_id": "mo-halfcheetah-v5",      # used by PGMORL / mo-gymnasium
    "pcn_env_id": "HalfCheetah-v5",     # used by our wrapper
    "n_seeds":  3,
    "base_seed": 1000,
    "results_root": "results",
    # Reference point for the hypervolume indicator (MAXIMISATION convention).
    # Must be component-wise dominated by every Pareto-optimal point.
    # Both PCN and PGMORL operate on raw (unscaled) rewards, so episode
    # returns live roughly in:
    #   obj0 (reward_run)  ~ [-50, +250]
    #   obj1 (reward_ctrl) ~ [-300,  0]
    # We anchor the HV floor at (0, -500) -- strictly dominated by every
    # reasonable front while leaving slack for noisier seeds.
    "ref_point": (0.0, -500.0),
    "pcn_overrides": {                   # passed to main.train()
        "total_iterations": 200,
        "num_envs": 4,
        "noise_scale": 0.60,             # default is 0.5 but our preliminary tests showed better performance with a slightly higher value; feel free to tune this further for your specific setup and random seeds
    },
    "pgmorl_kwargs": {                   # passed to PGMORL constructor
        "total_timesteps": 200_000,
        "num_envs": 4,
    },
}


# --------------------------------------------------------------------- #
# Per-seed runners
# --------------------------------------------------------------------- #
def run_pcn_seed(seed: int, cfg: dict[str, Any]) -> np.ndarray:
    """Train our PCN for one seed and return its final Pareto front.

    The training writes to ``results/benchmark_seed_<seed>/`` (own CSV +
    own checkpoints), leaving the global ``training_log.csv`` untouched.
    """
    # Local import to avoid pulling MuJoCo at module import time.
    from main import DEFAULT_CFG, train

    seed_dir = Path(cfg["results_root"]) / f"benchmark_seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        **DEFAULT_CFG,
        **cfg.get("pcn_overrides", {}),
        "env_id": cfg.get("pcn_env_id", DEFAULT_CFG["env_id"]),
        "seed": seed,
        # *** Critical: redirect ALL outputs to an isolated subdir. ***
        "results_dir": str(seed_dir),
        "log_filename": "training_log.csv",
    }
    train(run_cfg)

    return _read_front_from_csv(seed_dir / "training_log.csv")


def run_pgmorl_seed(seed: int, cfg: dict[str, Any]) -> np.ndarray:
    """Train PGMORL for one seed via ``morl-baselines``.

    Returns the final empirical Pareto front matrix (n, d). If
    ``morl-baselines`` / ``mo-gymnasium`` are not installed the function
    raises ``ModuleNotFoundError`` with a clear remediation hint and the
    caller logs a NaN HV for this seed.
    """
    try:
        import gymnasium as gym          # noqa: F401
        import mo_gymnasium              # noqa: F401
        from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"PGMORL baseline requires 'mo-gymnasium' and 'morl-baselines'. "
            f"Install with:  pip install mo-gymnasium morl-baselines  "
            f"(original error: {e})"
        ) from e

    seed_dir = Path(cfg["results_root"]) / f"benchmark_pgmorl_seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # PGMORL builds its own *vectorised* env internally from ``env_id``.
    # Passing a non-vectorised env via ``env=`` triggers the upstream check
    # "Environments should be vectorized for PPO." We therefore omit the
    # ``env`` kwarg entirely and let PGMORL handle vectorisation.
    eval_env = mo_gymnasium.make(cfg["env_id"])

    # Extract kwargs and pop ``total_timesteps`` (it's a ``train()`` arg, not
    # a constructor arg).
    pgmorl_kwargs = cfg.get("pgmorl_kwargs", {}).copy()
    timesteps = pgmorl_kwargs.pop("total_timesteps", 200_000)

    # Origin = worst-case point used by PGMORL for hypervolume-based selection
    # of candidate weights. Must be component-wise dominated by every point;
    # we reuse the user-supplied ``ref_point`` for consistency.
    origin = np.asarray(cfg["ref_point"], dtype=np.float32)

    agent = PGMORL(
        env_id=cfg["env_id"],
        origin=origin,
        seed=seed,
        log=False,
        experiment_name=f"pgmorl_seed_{seed}",
        **pgmorl_kwargs,
    )

    agent.train(
        total_timesteps=timesteps,
        eval_env=eval_env,
        ref_point=np.asarray(cfg["ref_point"], dtype=np.float32),
    )

    # Extraer el frente de Pareto directamente de las evaluaciones numéricas
    try:
        returns = np.array(agent.archive.evaluations, dtype=np.float32)
    except AttributeError:
        # Fallback de seguridad en caso de distintas versiones de la librería
        returns = np.array(
            [e for e in getattr(agent, "population_returns", [])],
            dtype=np.float32,
        )

    if returns.size == 0:
        return np.zeros((0, len(cfg["ref_point"])), dtype=np.float32)

    if returns.ndim == 1:
        returns = returns.reshape(-1, len(cfg["ref_point"]))

    idx = pareto_front_indices(returns)
    return returns[idx]

# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _read_front_from_csv(csv_path: Path) -> np.ndarray:
    """Recover an approximate Pareto front from our PCN training log.

    The CSV stores per-iteration ``best_obj0/1`` and target commands,
    which is *not* the full set of returns. To get a proper front we
    instead read the latest checkpoint's ``extra["pareto_front"]``.
    """
    seed_dir = csv_path.parent
    ckpts = sorted(seed_dir.glob("checkpoint_iter*.pkl"))
    if ckpts:
        import pickle
        with ckpts[-1].open("rb") as f:
            payload = pickle.load(f)
        front = np.asarray(
            payload.get("extra", {}).get("pareto_front", []),
            dtype=np.float32,
        )
        if front.size:
            return front

    # Fallback: derive a front from CSV columns (best_obj0, best_obj1).
    if not csv_path.exists():
        return np.zeros((0, 2), dtype=np.float32)
    import csv as _csv
    rows: list[list[float]] = []
    with csv_path.open("r") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                rows.append([float(row["best_obj0"]), float(row["best_obj1"])])
            except (KeyError, ValueError):
                continue
    if not rows:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    idx = pareto_front_indices(arr)
    return arr[idx]


# --------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------- #
def plot_hv_boxplot(
    pcn_hvs: np.ndarray,
    pgmorl_hvs: np.ndarray,
    out_path: Path,
) -> None:
    """Save a comparative box-plot of hypervolumes."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [pcn_hvs, pgmorl_hvs]
    labels = ["PCN (ours)", "PGMORL"]
    # ``tick_labels`` is the new name (matplotlib >= 3.9); fall back to
    # the deprecated ``labels`` kwarg for older versions.
    try:
        ax.boxplot(data, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showmeans=True)
    # Overlay individual seeds for transparency.
    for i, arr in enumerate(data, start=1):
        if arr.size:
            x = np.full_like(arr, i, dtype=float) + np.random.uniform(
                -0.05, 0.05, size=arr.size
            )
            ax.scatter(x, arr, alpha=0.6, s=18)
    ax.set_ylabel("Hypervolume")
    ax.set_title("PCN vs PGMORL — Hypervolume across seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pareto_fronts(
    pcn_fronts: list[np.ndarray],
    pgmorl_fronts: list[np.ndarray],
    out_path: Path,
) -> None:
    """Scatter of all Pareto fronts. Robust to fronts with a single point."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    def _plot_group(fronts: list[np.ndarray], color: str, label: str) -> None:
        first = True
        for fr in fronts:
            if fr is None or fr.size == 0:
                continue
            fr = np.atleast_2d(np.asarray(fr, dtype=np.float32))
            if fr.shape[1] < 2:
                continue
            xs, ys = fr[:, 0], fr[:, 1]
            ax.scatter(
                xs,
                ys,
                color=color,
                alpha=0.7,
                s=40 if fr.shape[0] == 1 else 25,
                marker="*" if fr.shape[0] == 1 else "o",
                label=label if first else None,
            )
            # Connect with a line only if there are >=2 points (sorted).
            if fr.shape[0] >= 2:
                order = np.argsort(xs)
                ax.plot(xs[order], ys[order], color=color, alpha=0.3, lw=1)
            first = False

    _plot_group(pcn_fronts, "tab:blue", "PCN (ours)")
    _plot_group(pgmorl_fronts, "tab:orange", "PGMORL")

    ax.set_xlabel("Objective 0  (reward_run)")
    ax.set_ylabel("Objective 1  (reward_ctrl)")
    ax.set_title("Empirical Pareto fronts across seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------- #
# Statistical test
# --------------------------------------------------------------------- #
def wilcoxon_ranksums_report(
    pcn_hvs: np.ndarray,
    pgmorl_hvs: np.ndarray,
    out_path: Path,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Wilcoxon rank-sum (Mann-Whitney) two-sample test.

    ``scipy.stats.ranksums`` does *not* assume paired samples, which is
    appropriate here because the two methods don't share seeds in any
    meaningful way (PGMORL's seed only controls its own RNG).
    """
    pcn_clean = pcn_hvs[np.isfinite(pcn_hvs)]
    pg_clean = pgmorl_hvs[np.isfinite(pgmorl_hvs)]

    report: dict[str, Any] = {
        "pcn_n": int(pcn_clean.size),
        "pcn_mean": float(np.mean(pcn_clean)) if pcn_clean.size else float("nan"),
        "pcn_std": float(np.std(pcn_clean, ddof=1))
        if pcn_clean.size > 1 else float("nan"),
        "pgmorl_n": int(pg_clean.size),
        "pgmorl_mean": float(np.mean(pg_clean)) if pg_clean.size else float("nan"),
        "pgmorl_std": float(np.std(pg_clean, ddof=1))
        if pg_clean.size > 1 else float("nan"),
        "alpha": alpha,
    }

    if pcn_clean.size >= 1 and pg_clean.size >= 1:
        # Two-sided test then a one-sided "PCN > PGMORL" follow-up.
        two = stats.ranksums(pcn_clean, pg_clean, alternative="two-sided")
        greater = stats.ranksums(pcn_clean, pg_clean, alternative="greater")
        report["statistic_two_sided"] = float(two.statistic)
        report["p_two_sided"] = float(two.pvalue)
        report["statistic_greater"] = float(greater.statistic)
        report["p_greater"] = float(greater.pvalue)
        report["pcn_beats_pgmorl"] = bool(greater.pvalue < alpha)
    else:
        report["statistic_two_sided"] = float("nan")
        report["p_two_sided"] = float("nan")
        report["statistic_greater"] = float("nan")
        report["p_greater"] = float("nan")
        report["pcn_beats_pgmorl"] = False

    out_path.write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
def run_benchmark(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """End-to-end benchmark. Returns the stats report dict."""
    cfg = {**DEFAULT_BENCHMARK_CFG, **(cfg or {})}
    results_root = Path(cfg["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    ref_point = np.asarray(cfg["ref_point"], dtype=np.float32)

    seeds = [cfg["base_seed"] + i for i in range(int(cfg["n_seeds"]))]

    # ---- 1. PCN ------------------------------------------------------ #
    pcn_fronts: list[np.ndarray] = []
    pcn_hvs: list[float] = []
    for s in seeds:
        try:
            front = run_pcn_seed(s, cfg)
        except Exception:
            print(f"[PCN seed={s}] FAILED:\n{traceback.format_exc()}",
                  file=sys.stderr)
            front = np.zeros((0, len(ref_point)), dtype=np.float32)
        pcn_fronts.append(front)
        hv = hypervolume(front, ref_point) if front.size else float("nan")
        pcn_hvs.append(hv)
        print(f"[PCN seed={s}] HV = {hv}", flush=True)

    # ---- 2. PGMORL --------------------------------------------------- #
    pgmorl_fronts: list[np.ndarray] = []
    pgmorl_hvs: list[float] = []
    for s in seeds:
        try:
            front = run_pgmorl_seed(s, cfg)
        except Exception:
            print(f"[PGMORL seed={s}] FAILED:\n{traceback.format_exc()}",
                  file=sys.stderr)
            front = np.zeros((0, len(ref_point)), dtype=np.float32)
        pgmorl_fronts.append(front)
        hv = hypervolume(front, ref_point) if front.size else float("nan")
        pgmorl_hvs.append(hv)
        print(f"[PGMORL seed={s}] HV = {hv}", flush=True)

    pcn_hv_arr = np.asarray(pcn_hvs, dtype=np.float64)
    pg_hv_arr = np.asarray(pgmorl_hvs, dtype=np.float64)

    # ---- 3. Save raw HVs --------------------------------------------- #
    np.savez(
        results_root / "benchmark_hypervolumes.npz",
        seeds=np.asarray(seeds),
        pcn=pcn_hv_arr,
        pgmorl=pg_hv_arr,
    )

    # ---- 4. Plots ---------------------------------------------------- #
    plot_hv_boxplot(pcn_hv_arr, pg_hv_arr, results_root / "boxplot_hv.png")
    plot_pareto_fronts(
        pcn_fronts, pgmorl_fronts, results_root / "pareto_fronts.png"
    )

    # ---- 5. Wilcoxon ------------------------------------------------- #
    report = wilcoxon_ranksums_report(
        pcn_hv_arr,
        pg_hv_arr,
        results_root / "stats_report.txt",
    )
    report["ref_point"] = list(map(float, ref_point))
    (results_root / "stats_report.txt").write_text(json.dumps(report, indent=2))
    print("Stats report:", json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seeds", type=int, default=DEFAULT_BENCHMARK_CFG["n_seeds"])
    p.add_argument("--base-seed", type=int, default=DEFAULT_BENCHMARK_CFG["base_seed"])
    p.add_argument(
        "--results-root", type=str, default=DEFAULT_BENCHMARK_CFG["results_root"]
    )
    p.add_argument(
        "--ref-point",
        type=float,
        nargs="+",
        default=list(DEFAULT_BENCHMARK_CFG["ref_point"]),
        help="Hypervolume reference point, one float per objective "
             "(maximisation convention; must be dominated by the front).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(
        {
            "n_seeds": args.n_seeds,
            "base_seed": args.base_seed,
            "results_root": args.results_root,
            "ref_point": tuple(args.ref_point),
        }
    )