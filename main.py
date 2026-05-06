"""
PCN — Pareto Conditioned Networks training loop for continuous MuJoCo
environments (Tarea 5 of the MORL course).

Workflow per outer iteration:

    1. Sample a target return R* from the empirical Pareto front of the
       trajectory buffer (random initial command if the buffer is still
       cold).
    2. Roll out ``cfg["episodes_per_iter"]`` episodes in parallel
       MuJoCo workers (``AsyncVectorEnv`` w/ ``spawn``), conditioning the
       actor on (state, R*, h_remaining). Each finished episode is added
       to the experience buffer.
    3. Train the conditioned actor with **behavioural cloning** on a
       batch of transitions sampled uniformly from the buffer. The
       supervised target is the action that was actually taken when the
       trajectory reached its observed return-to-go, so the network
       learns the inverse map  (s, G) -> a  that PCN exploits at test
       time.
    4. Log scalar metrics (one CSV row appended atomically per iter) and
       checkpoint the weights every ``checkpoint_every`` iterations.

The implementation is intentionally fail-safe:

    * CSV is opened in ``append`` mode and ``flush()``-ed every row, so
      a Ctrl-C / OOM / segfault never corrupts previous data.
    * Checkpoints are written to a temporary path and atomically renamed.
    * The ``__main__`` guard is mandatory because ``AsyncVectorEnv`` with
      ``context="spawn"`` re-imports this module in every worker.
"""
from __future__ import annotations

import csv
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Make ``src`` importable when running ``python main.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.envs import make_vector_envs                     # noqa: E402
from src.models import ConditionedActor                   # noqa: E402
from src.pcn import (                                     # noqa: E402
    ExperienceBuffer,
    sample_target_return,
)


# --------------------------------------------------------------------- #
# Default configuration
# --------------------------------------------------------------------- #
DEFAULT_CFG: dict[str, Any] = {
    # Environment
    "env_id": "HalfCheetah-v5",
    "num_envs": 8,
    "max_episode_steps": 1000,        # HalfCheetah default

    # Buffer / PCN
    "buffer_capacity": 500,           # # of trajectories
    "reward_dim": 2,                  # [reward_run, reward_ctrl]
    "target_noise_scale": 0.6,        # outward push when sampling R*
    "use_crowding": True,
    "warmup_episodes": 16,             # random-target episodes before training

    # Training loop
    "total_iterations": 500,
    "episodes_per_iter": 4,           # collected per outer iteration
    "updates_per_iter": 100,           # SGD steps per outer iteration
    "pareto_weight": 0.7, 
    "batch_size": 256,
    "lr": 3e-4,

    # Network
    "hidden_sizes": (256, 256),
    "use_horizon": True,
    "state_dependent_std": True,

    # Logging / checkpointing
    "results_dir": "results",
    "log_filename": "training_log.csv",
    "checkpoint_every": 10,           # iterations
    "log_every": 1,

    # Reproducibility / hardware
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_csv_logger(path: Path, fieldnames: list[str]) -> None:
    """Create the CSV with a header row iff it does not exist yet.

    Subsequent writes use append-mode; the header is never re-emitted, so
    a resumed run continues to pile rows on the same file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_csv_row(path: Path, fieldnames: list[str], row: dict) -> None:
    """Append-mode write + flush + fsync so logs survive crashes."""
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fsync may fail on some FS (e.g. tmpfs); not critical.
            pass


def save_checkpoint_atomic(
    path: Path,
    actor: ConditionedActor,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    cfg: dict,
    extra: dict | None = None,
) -> None:
    """Pickle weights+optim state to a temp file, then ``os.replace``.

    ``os.replace`` is atomic on POSIX/Windows, so partial writes from a
    Ctrl-C never leave a corrupted checkpoint behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "actor_state_dict": actor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg": cfg,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


# --------------------------------------------------------------------- #
# Rollout (vectorised, target-conditioned)
# --------------------------------------------------------------------- #
def rollout_episodes(
    vec_env,
    actor: ConditionedActor,
    target_return: np.ndarray,
    max_episode_steps: int,
    device: torch.device,
    deterministic: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Run ``num_envs`` parallel episodes conditioned on ``target_return``.

    Returns one dict per *finished* episode with keys
    ``states / actions / rewards``, each a numpy array of shape
    ``(T, ...)``.
    """
    num_envs = vec_env.num_envs
    obs, _ = vec_env.reset()

    # Per-worker rolling buffers for the current episode.
    cur_states: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    cur_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    cur_rewards: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    cur_step = np.zeros(num_envs, dtype=np.int64)

    finished: list[dict[str, np.ndarray]] = []
    target_t = torch.as_tensor(
        target_return, dtype=torch.float32, device=device
    ).unsqueeze(0).expand(num_envs, -1)

    for _ in range(max_episode_steps):
        with torch.no_grad():
            state_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            horizon_remaining = (max_episode_steps - cur_step).astype(np.float32)
            horizon_t = torch.as_tensor(
                horizon_remaining, dtype=torch.float32, device=device
            )
            mu, std = actor(state_t, target_t, horizon_t)
            if deterministic:
                action_t = mu
            else:
                action_t = mu + std * torch.randn_like(std)

        # Clip to MuJoCo's [-1, 1] action bounds (HalfCheetah convention).
        action_np = action_t.cpu().numpy().astype(np.float32)
        action_np = np.clip(action_np, -1.0, 1.0)

        next_obs, vec_reward, terminated, truncated, _info = vec_env.step(
            action_np
        )

        for i in range(num_envs):
            cur_states[i].append(obs[i].astype(np.float32))
            cur_actions[i].append(action_np[i])
            cur_rewards[i].append(np.asarray(vec_reward[i], dtype=np.float32))
            cur_step[i] += 1

            if terminated[i] or truncated[i]:
                finished.append(
                    {
                        "states": np.stack(cur_states[i], axis=0),
                        "actions": np.stack(cur_actions[i], axis=0),
                        "rewards": np.stack(cur_rewards[i], axis=0),
                    }
                )
                cur_states[i] = []
                cur_actions[i] = []
                cur_rewards[i] = []
                cur_step[i] = 0

        obs = next_obs
        # AsyncVectorEnv auto-resets, so we just keep stepping until we
        # have collected enough finished episodes.
        if len(finished) >= num_envs:
            break

    return finished


# --------------------------------------------------------------------- #
# Behavioural-cloning update
# --------------------------------------------------------------------- #
def bc_update(
    actor: ConditionedActor,
    optimizer: torch.optim.Optimizer,
    buffer: ExperienceBuffer,
    batch_size: int,
    n_updates: int,
    device: torch.device,
    rng: np.random.Generator,
    pareto_weight: float = 0.0,
) -> dict[str, float]:
    """Train the actor by negative-log-likelihood imitation of buffer
    actions, conditioned on the actual returns-to-go they achieved.

    This is the supervised loss originally proposed for PCN
    (Reymond et al., 2022), adapted to a Gaussian policy via NLL.
    """
    actor.train()
    losses: list[float] = []
    for _ in range(n_updates):
        batch = buffer.sample_transitions(batch_size, pareto_weight=pareto_weight, rng=rng)
        s = torch.as_tensor(batch["state"], device=device)
        a = torch.as_tensor(batch["action"], device=device)
        g = torch.as_tensor(batch["return_to_go"], device=device)
        h = torch.as_tensor(batch["horizon_to_go"], device=device)

        dist = actor.distribution(s, g, h if actor.use_horizon else None)
        log_prob = dist.log_prob(a)
        loss = -log_prob.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    actor.eval()
    return {
        "bc_loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "bc_loss_last": float(losses[-1]) if losses else float("nan"),
    }


# --------------------------------------------------------------------- #
# Main training entry point
# --------------------------------------------------------------------- #
def train(cfg: dict[str, Any] | None = None) -> None:
    """Run the full PCN training loop with the given config."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    set_global_seed(cfg["seed"])
    device = torch.device(cfg["device"])
    rng = np.random.default_rng(cfg["seed"])

    # ---- Output paths ------------------------------------------------ #
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / cfg["log_filename"]
    fieldnames = [
        "iteration",
        "wall_time_s",
        "n_trajectories",
        "pareto_front_size",
        "best_obj0",
        "best_obj1",
        "mean_episode_return_obj0",
        "mean_episode_return_obj1",
        "target_obj0",
        "target_obj1",
        "bc_loss_mean",
        "bc_loss_last",
    ]
    init_csv_logger(log_path, fieldnames)

    # ---- Vector env (spawn-isolated MuJoCo workers) ------------------ #
    vec_env = make_vector_envs(
        env_id=cfg["env_id"],
        num_envs=cfg["num_envs"],
        seed=cfg["seed"],
    )
    obs_dim = int(np.prod(vec_env.single_observation_space.shape))
    action_dim = int(np.prod(vec_env.single_action_space.shape))

    # ---- Actor + optimiser ------------------------------------------- #
    actor = ConditionedActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=cfg["reward_dim"],
        hidden_sizes=tuple(cfg["hidden_sizes"]),
        use_horizon=cfg["use_horizon"],
        state_dependent_std=cfg["state_dependent_std"],
    ).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])

    # ---- Buffer ------------------------------------------------------ #
    buffer = ExperienceBuffer(
        capacity=cfg["buffer_capacity"],
        reward_dim=cfg["reward_dim"],
    )

    # ---- Training loop ----------------------------------------------- #
    t0 = time.time()
    try:
        for it in range(1, cfg["total_iterations"] + 1):
            # --- 1. Pick a target return ----------------------------- #
            if len(buffer) < cfg["warmup_episodes"]:
                # Cold start: random commands so the buffer fills with
                # a diverse mix of behaviours.
                target_return = rng.uniform(low=-2.0, high=2.0, size=cfg["reward_dim"]).astype(np.float32)

            else:
                front = buffer.pareto_front()
                target_return = sample_target_return(
                    front,
                    noise_scale=cfg["target_noise_scale"],
                    use_crowding=cfg["use_crowding"],
                    rng=rng,
                )

            # --- 2. Roll out & store --------------------------------- #
            episodes = rollout_episodes(
                vec_env=vec_env,
                actor=actor,
                target_return=target_return,
                max_episode_steps=cfg["max_episode_steps"],
                device=device,
                deterministic=False,
            )
            # Keep at most ``episodes_per_iter`` episodes per iteration.
            for ep in episodes[: cfg["episodes_per_iter"]]:
                buffer.add_trajectory(
                    states=ep["states"],
                    actions=ep["actions"],
                    rewards=ep["rewards"],
                )

            # --- 3. BC update ---------------------------------------- #
            if len(buffer) >= 1:
                update_stats = bc_update(
                    actor=actor,
                    optimizer=optimizer,
                    buffer=buffer,
                    batch_size=cfg["batch_size"],
                    n_updates=cfg["updates_per_iter"],
                    device=device,
                    rng=rng,
                    pareto_weight=cfg.get("pareto_weight", 0.7),
                )
            else:
                update_stats = {"bc_loss_mean": float("nan"),
                                "bc_loss_last": float("nan")}

            # --- 4. Log + checkpoint --------------------------------- #
            ep_returns = buffer.episode_returns
            front_idx = buffer.pareto_episode_indices()
            front = ep_returns[front_idx] if front_idx.size else ep_returns

            row = {
                "iteration": it,
                "wall_time_s": round(time.time() - t0, 3),
                "n_trajectories": len(buffer),
                "pareto_front_size": int(front_idx.size),
                "best_obj0": float(ep_returns[:, 0].max())
                if ep_returns.size else float("nan"),
                "best_obj1": float(ep_returns[:, 1].max())
                if ep_returns.size else float("nan"),
                "mean_episode_return_obj0": float(ep_returns[:, 0].mean())
                if ep_returns.size else float("nan"),
                "mean_episode_return_obj1": float(ep_returns[:, 1].mean())
                if ep_returns.size else float("nan"),
                "target_obj0": float(target_return[0]),
                "target_obj1": float(target_return[1]),
                **update_stats,
            }

            if it % cfg["log_every"] == 0:
                append_csv_row(log_path, fieldnames, row)
                print(
                    f"[iter {it:04d}] traj={row['n_trajectories']:3d} "
                    f"front={row['pareto_front_size']:3d} "
                    f"best=({row['best_obj0']:.2f}, {row['best_obj1']:.2f}) "
                    f"target=({row['target_obj0']:.2f}, "
                    f"{row['target_obj1']:.2f}) "
                    f"bc_loss={row['bc_loss_mean']:.4f}",
                    flush=True,
                )

            if it % cfg["checkpoint_every"] == 0:
                ckpt_path = results_dir / f"checkpoint_iter{it:04d}.pkl"
                save_checkpoint_atomic(
                    path=ckpt_path,
                    actor=actor,
                    optimizer=optimizer,
                    iteration=it,
                    cfg=cfg,
                    extra={"pareto_front": front.tolist()},
                )

    finally:
        # Always close MuJoCo workers cleanly to avoid orphaned procs.
        try:
            vec_env.close()
        except Exception:
            pass


# --------------------------------------------------------------------- #
# Entry point — *must* be guarded for ``spawn`` workers to import safely
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # necesario si algún día compilas a .exe
    train()