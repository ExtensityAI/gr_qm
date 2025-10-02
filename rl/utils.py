
from __future__ import annotations
import os
import numpy as np
from typing import Callable, Dict, Any

from stable_baselines3.common.monitor import Monitor

from simulation.bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec


def make_env_factory(target: TargetSpec, env_conf: EnvConfig, seed: int,
                     monitor_dir: str) -> Callable[[], BlackHoleParamEnv]:
    """
    Factory returning a thunk to create a monitored BlackHoleParamEnv.
    Suitable for SubprocVecEnv / DummyVecEnv.
    """
    os.makedirs(monitor_dir, exist_ok=True)
    def _thunk():
        env = BlackHoleParamEnv(target=target, config=env_conf)
        env = Monitor(env, filename=os.path.join(monitor_dir, f"monitor-seed{seed}.csv"))
        env.reset(seed=seed)
        return env
    return _thunk


def evaluate_policy_detailed(model, eval_env, n_eval_episodes: int = 10, deterministic: bool = True):
    """
    Roll out evaluation episodes and compute mean reward plus auxiliary stats.
    """
    import numpy as np

    episode_rewards = []
    chi2s = []
    params = []

    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_r = 0.0
        ep_chi2 = []
        last_info = None
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = eval_env.step(action)
            ep_r += float(reward)
            if isinstance(info, (list, tuple)) and len(info) > 0:
                last_info = info[0]
                if "chi2" in last_info:
                    ep_chi2.append(float(last_info["chi2"]))
                if "param" in last_info:
                    params.append(float(last_info["param"]))
            if done:
                break
        episode_rewards.append(ep_r)
        if len(ep_chi2) > 0:
            chi2s.append(np.mean(ep_chi2))

    stats = {
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_chi2": float(np.mean(chi2s)) if chi2s else None,
        "std_chi2": float(np.std(chi2s)) if chi2s else None,
        "mean_param": float(np.mean(params)) if params else None,
        "std_param": float(np.std(params)) if params else None,
    }
    return stats
