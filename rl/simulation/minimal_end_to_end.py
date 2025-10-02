from __future__ import annotations

import argparse
import os
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from simulation.bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec
from utils import evaluate_policy_detailed


def make_env(target: TargetSpec, config: EnvConfig):
    def _thunk():
        return BlackHoleParamEnv(target=target, config=config)
    return _thunk


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--tb", type=str, default="tb_minimal")
    args = p.parse_args()

    target = TargetSpec(
        f220_Hz=None,
        tau220_s=None,
        M_solar=70.0,
        distance_m=None,
        theta_shadow_rad=None,
        b_c_m=None,
    )

    env_conf = EnvConfig(
        core_model="fractional",
        eps_min=0.0,
        eps_max=0.15,
        max_steps=64,
        chi2_converged=1e-2,
        step_size=0.02,
        use_loglike_reward=True,
        render_device="cpu",
        enable_render_overlays=False,
    )

    venv = DummyVecEnv([make_env(target, env_conf)])
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        tensorboard_log=args.tb,
        seed=args.seed,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps, tb_log_name=f"minimal_seed{args.seed}")

    stats: dict[str, Any] = evaluate_policy_detailed(model, venv, n_eval_episodes=5, deterministic=True)
    print(stats)


if __name__ == "__main__":
    main()

