
from __future__ import annotations
import os
import sys
import math
import multiprocessing as mp
from dataclasses import asdict
from typing import Optional, Dict, Any, Callable

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

# Ensure we can import the env and simulation helpers
PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# If the user keeps bh_gym_env.py in project root, this import works.
from simulation.bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from callbacks import MetricsCallback, TrainingTBCallback, DebugTBHistCallback
from utils import make_env_factory, evaluate_policy_detailed


def _build_target(cfg) -> TargetSpec:
    t = cfg.target
    return TargetSpec(
        f220_Hz=t.get("f220_Hz"),
        sigma_f_Hz=t.get("sigma_f_Hz"),
        tau220_s=t.get("tau220_s"),
        sigma_tau_s=t.get("sigma_tau_s"),
        M_solar=t.get("M_solar"),
        distance_m=t.get("distance_m"),
        theta_shadow_rad=t.get("theta_shadow_rad"),
        sigma_theta_rad=t.get("sigma_theta_rad"),
        b_c_m=t.get("b_c_m"),
        sigma_b_c_m=t.get("sigma_b_c_m"),
    )


def _autobootstrap_target_if_empty(target: TargetSpec, env_conf: EnvConfig) -> TargetSpec:
    empty = (
        target.f220_Hz is None and
        target.tau220_s is None and
        target.theta_shadow_rad is None and
        target.b_c_m is None
    )
    if not empty:
        return target

    # Import simulator backend
    try:
        from simulation import simulation_rl as _sim
    except Exception as e:
        raise RuntimeError("Failed to import simulation backend for target bootstrap.") from e

    Ms = float(target.M_solar or env_conf.default_M_solar)

    # Choose a ground-truth param offset from init center to avoid immediate convergence
    if env_conf.core_model == "fractional":
        gt = float(min(max(env_conf.init_center_eps + 0.03, env_conf.eps_min), env_conf.eps_max))
        _sim.set_core_params(eps=gt, model="fractional")
    else:
        mid = 0.5 * (env_conf.L0_min + env_conf.L0_max)
        gt = float(min(max(mid + 0.1 * (env_conf.L0_max - env_conf.L0_min), env_conf.L0_min), env_conf.L0_max))
        _sim.set_core_params(L0=gt, model="absolute")

    obs = _sim.bh_observables(M_solar=Ms, distance_m=1.0)
    f = obs.get("f220_Hz", None)
    tau = obs.get("tau220_s", None)
    bc = obs.get("b_c_m", None)
    # Prefer also theta if available
    theta = obs.get("theta_shadow_rad", None)

    # Conservative sigmas (relative)
    def rel_sigma(val: float | None, rel: float) -> float | None:
        return (rel * float(val)) if (val is not None) else None

    sigma_f = rel_sigma(f, 0.01)
    sigma_tau = rel_sigma(tau, 0.05)
    sigma_bc = rel_sigma(bc, 0.02)
    sigma_theta = rel_sigma(theta, 0.02)

    print("[bootstrap] Generated synthetic target at ground-truth param:", gt)

    return TargetSpec(
        f220_Hz=(float(f) if f is not None else None),
        sigma_f_Hz=(float(sigma_f) if sigma_f is not None else None),
        tau220_s=(float(tau) if tau is not None else None),
        sigma_tau_s=(float(sigma_tau) if sigma_tau is not None else None),
        M_solar=Ms,
        distance_m=1.0,
        theta_shadow_rad=(float(theta) if theta is not None else None),
        sigma_theta_rad=(float(sigma_theta) if sigma_theta is not None else None),
        b_c_m=(float(bc) if bc is not None else None),
        sigma_b_c_m=(float(sigma_bc) if sigma_bc is not None else None),
    )


def _get_preset_overrides(cfg) -> tuple[dict, dict]:
    prof = getattr(cfg, "difficulty_profile", None)
    if not prof:
        return {}, {}
    presets = getattr(cfg, "difficulty_presets", {})
    if prof not in presets:
        return {}, {}
    env_over = dict(presets[prof].get("env", {}))
    ppo_over = dict(presets[prof].get("ppo", {}))
    return env_over, ppo_over


def _build_env_config(cfg) -> EnvConfig:
    e = cfg.env
    env_over, _ = _get_preset_overrides(cfg)
    def gv(name, default=None):
        return env_over.get(name, e.get(name, default))
    return EnvConfig(
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


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Train PPO on the BlackHoleParamEnv. Returns the final evaluation mean reward so
    hydra-optuna-sweeper (if enabled) can optimize it.
    """
    # Pretty print the config
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Reproducibility
    seed = int(cfg.seed)
    set_random_seed(seed)

    # Set up environment factory
    target = _build_target(cfg)
    target = _autobootstrap_target_if_empty(target, env_conf=_build_env_config(cfg))
    env_conf = _build_env_config(cfg)
    env_id = "BlackHoleParamEnv-v0"

    def env_fn(rank: int) -> Callable[[], BlackHoleParamEnv]:
        return make_env_factory(target, env_conf, seed + rank, monitor_dir=cfg.logging.monitor_dir)

    # Vectorized environments
    n_envs = int(cfg.n_envs) if cfg.n_envs > 0 else max(1, mp.cpu_count() - 1)
    if n_envs == 1:
        venv = DummyVecEnv([env_fn(0)])
    else:
        venv = SubprocVecEnv([env_fn(i) for i in range(n_envs)], start_method="fork")

    # Optional VecNormalize (obs and/or reward normalization)
    if cfg.vecnormalize.enabled:
        venv = VecNormalize(
            venv,
            norm_obs=cfg.vecnormalize.norm_obs,
            norm_reward=cfg.vecnormalize.norm_reward,
            clip_obs=cfg.vecnormalize.clip_obs,
            gamma=cfg.vecnormalize.gamma,
            epsilon=1e-8
        )

    # PPO model
    env_over, ppo_over = _get_preset_overrides(cfg)
    ppo_cfg = cfg.ppo
    policy_kwargs = dict(
        net_arch=ppo_cfg.net_arch,
        activation_fn=__import__("torch").nn.Tanh,  # default to Tanh; configurable if needed
    )
    model = PPO(
        ppo_cfg.policy,
        venv,
        learning_rate=ppo_over.get("learning_rate", ppo_cfg.learning_rate),
        n_steps=ppo_over.get("n_steps", ppo_cfg.n_steps),
        batch_size=ppo_over.get("batch_size", ppo_cfg.batch_size),
        n_epochs=ppo_over.get("n_epochs", ppo_cfg.n_epochs),
        gamma=ppo_over.get("gamma", ppo_cfg.gamma),
        gae_lambda=ppo_over.get("gae_lambda", ppo_cfg.gae_lambda),
        clip_range=ppo_over.get("clip_range", ppo_cfg.clip_range),
        ent_coef=ppo_over.get("ent_coef", ppo_cfg.ent_coef),
        vf_coef=ppo_over.get("vf_coef", ppo_cfg.vf_coef),
        max_grad_norm=ppo_over.get("max_grad_norm", ppo_cfg.max_grad_norm),
        tensorboard_log=cfg.logging.tensorboard_dir,
        policy_kwargs=policy_kwargs,
        device=cfg.device,
        seed=seed,
        verbose=1,
    )

    # Callbacks: metrics, eval, and checkpoints
    callbacks = []

    metrics_cb = MetricsCallback(log_param=True)  # logs chi2, residuals, current param
    callbacks.append(metrics_cb)
    callbacks.append(TrainingTBCallback())  # logs LR, schedules, counters
    callbacks.append(DebugTBHistCallback(histogram_every=1000, buffer_size=4096))  # residual/param histograms

    # Build evaluation env (single, deterministic)
    eval_env = DummyVecEnv([env_fn(10_000)])  # different seed
    if cfg.vecnormalize.enabled:
        # Sync normalization stats from train env
        eval_env = VecNormalize(eval_env, training=False, norm_obs=cfg.vecnormalize.norm_obs, norm_reward=False)
        eval_env.obs_rms = venv.obs_rms

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg.logging.best_model_dir,
        log_path=cfg.logging.eval_log_dir,
        eval_freq=cfg.eval.eval_freq,
        n_eval_episodes=cfg.eval.n_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_cb)

    if cfg.checkpoint.enabled:
        ckpt_cb = CheckpointCallback(
            save_freq=cfg.checkpoint.save_freq,
            save_path=cfg.logging.checkpoint_dir,
            name_prefix=cfg.checkpoint.name_prefix,
            save_replay_buffer=False,
            save_vecnormalize=cfg.vecnormalize.enabled,
        )
        callbacks.append(ckpt_cb)

    # Train
    total_timesteps = int(cfg.train.total_timesteps)
    # Name the TB run for easier comparison
    tb_name = f"ppo_{_build_env_config(cfg).difficulty.lower()}_seed{seed}_nenv{n_envs}"
    model.learn(total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=cfg.train.progress_bar,
                tb_log_name=tb_name,
                log_interval=10)

    # Save artifacts
    os.makedirs(cfg.logging.model_dir, exist_ok=True)
    model.save(os.path.join(cfg.logging.model_dir, "ppo_blackhole"))
    if cfg.vecnormalize.enabled:
        venv.save(os.path.join(cfg.logging.model_dir, "vecnormalize.pkl"))

    # Final evaluation (detailed) – also returns mean episode reward
    eval_stats = evaluate_policy_detailed(
        model,
        eval_env,
        n_eval_episodes=cfg.eval.n_episodes,
        deterministic=True
    )
    mean_reward = float(eval_stats["mean_reward"])

    # Log final evaluation stats to TensorBoard as well
    try:
        for k, v in eval_stats.items():
            if v is not None:
                model.logger.record(f"final_eval/{k}", float(v))
        # Flush at the current timestep for visibility in TensorBoard
        model.logger.dump(int(model.num_timesteps))
    except Exception:
        pass

    # Write summary to disk for Hydra artifacts
    import json
    with open(os.path.join(cfg.logging.model_dir, "final_eval.json"), "w") as f:
        json.dump(eval_stats, f, indent=2)

    print(f"[RESULT] mean_reward={mean_reward:.6f}")
    return mean_reward


if __name__ == "__main__":
    main()
