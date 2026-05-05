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