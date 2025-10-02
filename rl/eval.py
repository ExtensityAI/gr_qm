
from __future__ import annotations
import os
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from simulation.bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec
from utils import make_env_factory, evaluate_policy_detailed
from omegaconf import OmegaConf
import argparse


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
    ap = argparse.ArgumentParser(description="Load a trained PPO and evaluate with chosen difficulty profile.")
    ap.add_argument("--model-dir", type=str, default="models")
    ap.add_argument("--vecnorm-path", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--profile", type=str, default=None, choices=[None, "medium", "hard"], help="Difficulty profile to apply from config.yaml presets")
    args = ap.parse_args()

    cfg = OmegaConf.load("config.yaml")
    env_base = cfg.env
    env_over = {}
    if args.profile:
        preset = cfg.get("difficulty_presets", {}).get(args.profile, {})
        env_over = dict(preset.get("env", {}))

    def gv(name, default=None):
        return env_over.get(name, env_base.get(name, default))

    env_conf = EnvConfig(
        core_model=gv("core_model"),
        eps_min=gv("eps_min"),
        eps_max=gv("eps_max"),
        L0_min=gv("L0_min"),
        L0_max=gv("L0_max"),
        max_steps=gv("max_steps"),
        chi2_converged=gv("chi2_converged"),
        step_size=gv("step_size"),
        use_loglike_reward=gv("use_loglike_reward"),
        render_width=gv("render_width"),
        render_height=gv("render_height"),
        render_device=gv("render_device"),
        enable_render_overlays=gv("enable_render_overlays"),
        init_center_eps=gv("init_center_eps"),
        init_spread=gv("init_spread"),
        default_M_solar=gv("default_M_solar"),
        difficulty=gv("difficulty", "LOW"),
        fidelity_low_noise_mult=gv("fidelity_low_noise_mult"),
        fidelity_high_noise_mult=gv("fidelity_high_noise_mult"),
        fidelity_low_step_cost=gv("fidelity_low_step_cost"),
        fidelity_high_step_cost=gv("fidelity_high_step_cost"),
        budget_init=gv("budget_init"),
        cost_weight=gv("cost_weight"),
        cost_query_f220=gv("cost_query_f220"),
        cost_query_tau220=gv("cost_query_tau220"),
        cost_query_theta=gv("cost_query_theta"),
        cost_query_bc=gv("cost_query_bc"),
        hard_terminal_full_eval=gv("hard_terminal_full_eval"),
        include_masks_in_obs=gv("include_masks_in_obs"),
        hard_noise_scale=gv("hard_noise_scale"),
    )

    target = TargetSpec(
        f220_Hz=cfg.target.f220_Hz,
        sigma_f_Hz=cfg.target.sigma_f_Hz,
        tau220_s=cfg.target.tau220_s,
        sigma_tau_s=cfg.target.sigma_tau_s,
        M_solar=cfg.target.M_solar,
        distance_m=cfg.target.distance_m,
        theta_shadow_rad=cfg.target.theta_shadow_rad,
        sigma_theta_rad=cfg.target.sigma_theta_rad,
        b_c_m=cfg.target.b_c_m,
        sigma_b_c_m=cfg.target.sigma_b_c_m,
    )

    load_and_eval(args.model_dir, vecnorm_path=args.vecnorm_path,
                  target=target, env_conf=env_conf, n_eval_episodes=int(args.episodes))
