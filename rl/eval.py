
from __future__ import annotations
import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from simulation.bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec
from utils import make_env_factory, evaluate_policy_detailed


def load_and_eval(model_dir="models", vecnorm_path=None,
                  target=None, env_conf=None,
                  n_eval_episodes=20):
    model_path = os.path.join(model_dir, "ppo_blackhole.zip")
    model = PPO.load(model_path)
    env_fn = make_env_factory(target, env_conf, seed=123, monitor_dir="eval_monitors")
    eval_env = DummyVecEnv([env_fn])
    if vecnorm_path and os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    stats = evaluate_policy_detailed(model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    # Example: fill with your target and env config
    target = TargetSpec(f220_Hz=None, tau220_s=None, M_solar=70.0)
    env_conf = EnvConfig()
    load_and_eval("models", vecnorm_path="models/vecnormalize.pkl",
                  target=target, env_conf=env_conf, n_eval_episodes=10)
