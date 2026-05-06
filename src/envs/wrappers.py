"""
Multi-objective wrappers for MuJoCo continuous-control environments.

This module exposes:

* :class:`MOHalfCheetahWrapper` -- a Gymnasium wrapper that converts the scalar
  reward emitted by ``HalfCheetah-v5`` into a 2-dimensional vector reward
  ``[reward_run, reward_ctrl]``. These two components are the same terms that
  the underlying MuJoCo task internally combines into the scalar reward, and
  are recovered from ``info`` so that they can be optimised independently by a
  multi-objective RL algorithm such as PCN.
* :func:`make_env` -- thunk-style factory that builds a single seeded MO env.
* :func:`make_vector_envs` -- factory that builds an
  :class:`gymnasium.vector.AsyncVectorEnv` configured with the ``"spawn"``
  multiprocessing context. The ``spawn`` context is mandatory when working
  with MuJoCo: ``fork`` duplicates GL/EGL/MuJoCo handles across processes,
  which causes segmentation faults and slow memory leaks. ``spawn`` starts
  fresh interpreters so each worker initialises its own MuJoCo instance in
  isolation.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv


# Dimensionality of the reward vector produced by :class:`MOHalfCheetahWrapper`.
# Index 0 -> forward/run reward, index 1 -> control cost (negated).
NUM_OBJECTIVES: int = 2


class MOHalfCheetahWrapper(gym.Wrapper):
    """Turn ``HalfCheetah-v5`` into a multi-objective environment.

    The native HalfCheetah task computes its scalar reward as::

        reward = reward_run + reward_ctrl

    where ``reward_run`` is the forward velocity reward and ``reward_ctrl`` is
    a (negative) control-effort penalty. Both terms are exposed in the
    ``info`` dictionary as ``reward_run`` / ``reward_forward`` and
    ``reward_ctrl`` respectively. This wrapper replaces the scalar reward
    with the vector ``np.array([reward_run, reward_ctrl], dtype=np.float32)``
    so downstream MORL algorithms can optimise both objectives independently.

    Notes
    -----
    * The original scalar reward is preserved in ``info["original_reward"]``
      for logging convenience.
    * The vector reward is stored in ``info["vector_reward"]`` as well, which
      is the key convention used by ``mo-gymnasium``.
    * The observation space is left untouched.
    * Rewards are divided by ``REWARD_SCALE`` (=100) to keep both objectives
      in a comparable ~O(1) range. This stabilises the supervised regression
      that PCN performs (return-conditioned Gaussian NLL) and prevents
      numerical overflow when the buffer accumulates large episode returns.
      Multiply HV / returns by ``REWARD_SCALE`` if you need the original
      MuJoCo scale for reporting.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.reward_dim: int = NUM_OBJECTIVES
        # Expose a reward_space attribute that mirrors the mo-gymnasium API.
        self.reward_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_OBJECTIVES,),
            dtype=np.float32,
        )

    def step(self, action):  # type: ignore[override]
        obs, scalar_reward, terminated, truncated, info = self.env.step(action)

        # HalfCheetah-v5 exposes both terms in ``info``; fall back gracefully
        # if a future version changes the key names.
        reward_run = float(
            info.get("reward_run", info.get("reward_forward", 0.0))
        )
        reward_ctrl = float(info.get("reward_ctrl", 0.0))

        # Normalisation (divide by 100) to stabilise the regression target
        # and keep both objectives in a comparable scale.
        vector_reward = np.array(
            [reward_run / 100.0, reward_ctrl / 100.0],
            dtype=np.float32,
        )

        info["original_reward"] = float(scalar_reward)
        info["vector_reward"] = vector_reward

        return obs, vector_reward, terminated, truncated, info


def make_env(env_id: str, seed: int) -> Callable[[], gym.Env]:
    """Build a thunk that creates a seeded multi-objective env instance.

    The returned callable takes no arguments and produces a fresh wrapped
    environment when invoked. This deferred-construction pattern is the one
    expected by Gymnasium's vector-env constructors (``AsyncVectorEnv``,
    ``SyncVectorEnv``), which need to instantiate the environment *inside*
    each worker process.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id. The wrapper currently targets
        ``"HalfCheetah-v5"`` but accepts any HalfCheetah-like env that
        exposes ``reward_run`` and ``reward_ctrl`` in ``info``.
    seed : int
        Seed used to initialise the underlying RNG via
        :py:meth:`gym.Env.reset`. Each worker must receive a unique seed in
        the vectorised setting.

    Returns
    -------
    Callable[[], gym.Env]
        A no-arg thunk that returns the wrapped environment.
    """

    def _thunk() -> gym.Env:
        env = gym.make(env_id)
        env = MOHalfCheetahWrapper(env)
        # ``reset`` with a seed propagates to the action/observation spaces
        # and to the underlying MuJoCo simulator's RNG.
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _thunk


def make_vector_envs(
    env_id: str,
    num_envs: int,
    seed: int,
) -> AsyncVectorEnv:
    """Create an asynchronous vector of MO environments.

    Each sub-environment runs in its own process, started with the
    ``"spawn"`` multiprocessing context. ``spawn`` is required for MuJoCo:

    * It launches a brand-new Python interpreter for every worker, so MuJoCo,
      GLFW/EGL contexts, and the underlying C state are *not* inherited
      from the parent. ``fork`` would duplicate those handles and lead to
      undefined behaviour, segmentation faults, or progressive RAM/GPU
      memory leaks across long training runs.
    * It is the only context guaranteed to work on macOS and on CUDA-enabled
      Linux setups in combination with MuJoCo.

    Each worker is given a deterministic but distinct seed
    (``seed, seed + 1, ..., seed + num_envs - 1``) so the rollouts are
    reproducible while remaining decorrelated.

    Parameters
    ----------
    env_id : str
        Gymnasium environment id (e.g. ``"HalfCheetah-v5"``).
    num_envs : int
        Number of parallel worker processes to spawn. Must be ``>= 1``.
    seed : int
        Base seed; worker ``i`` is seeded with ``seed + i``.

    Returns
    -------
    gymnasium.vector.AsyncVectorEnv
        A vector environment whose :py:meth:`step` returns batched
        ``(obs, vector_reward, terminated, truncated, info)`` tuples, where
        ``vector_reward`` has shape ``(num_envs, NUM_OBJECTIVES)``.

    Raises
    ------
    ValueError
        If ``num_envs < 1``.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    env_fns = [make_env(env_id, seed + i) for i in range(num_envs)]

    # Explicitly request the spawn context to isolate MuJoCo per worker.
    # Gymnasium's AsyncVectorEnv expects the *method name* (string) rather
    # than a context object -- it calls ``mp.get_context(method)`` itself.
    # Validate the name early so we fail fast with a clearer error.
    mp.get_context("spawn")

    import platform
    ctx = "spawn"  # único disponible en Windows, también correcto en Linux/Mac
    return AsyncVectorEnv(
        env_fns,
        context=ctx,
        shared_memory=False,
        copy=True,
    )