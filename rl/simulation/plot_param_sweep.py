from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from simulation.bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec
from . import simulation_rl as _sim


def make_self_consistent_target(msolar: float, gt_eps: float) -> TargetSpec:
    _sim.set_core_params(eps=float(gt_eps), model="fractional")
    obs = _sim.bh_observables(M_solar=float(msolar), distance_m=1.0)
    f = float(obs["f220_Hz"]) if "f220_Hz" in obs else None
    tau = float(obs["tau220_s"]) if "tau220_s" in obs else None
    bc = float(obs.get("b_c_m", 0.0)) if obs.get("b_c_m", None) is not None else None

    sigma_f = 0.005 * f if f is not None else None
    sigma_tau = 0.05 * tau if tau is not None else None
    sigma_bc = 0.02 * bc if bc is not None else None

    return TargetSpec(
        f220_Hz=f, sigma_f_Hz=sigma_f,
        tau220_s=tau, sigma_tau_s=sigma_tau,
        M_solar=float(msolar),
        distance_m=1.0,
        theta_shadow_rad=None,
        b_c_m=bc, sigma_b_c_m=sigma_bc,
    )


def chi2_for_eps(env: BlackHoleParamEnv, eps: float) -> float:
    env._param = float(eps)
    env._apply_param_to_sim(env._param)
    chi2, _ = env._chi2_and_residuals()
    return float(chi2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eps-min", type=float, default=0.0)
    p.add_argument("--eps-max", type=float, default=0.15)
    p.add_argument("--points", type=int, default=101)
    p.add_argument("--msolar", type=float, default=70.0)
    p.add_argument("--gt-eps", type=float, default=0.05)
    p.add_argument("--out", type=str, default="data/chi2_sweep.png")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    target = make_self_consistent_target(args.msolar, args.gt_eps)
    env_conf = EnvConfig(core_model="fractional", eps_min=args.eps_min, eps_max=args.eps_max)
    env = BlackHoleParamEnv(target=target, config=env_conf)
    env.reset(seed=args.seed)

    xs = np.linspace(args.eps_min, args.eps_max, args.points)
    ys = [chi2_for_eps(env, float(x)) for x in xs]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, label=r"$\chi^2(\varepsilon)$")
    plt.axvline(args.gt_eps, color="r", linestyle="--", label=f"gt eps={args.gt_eps}")
    plt.xlabel(r"$\varepsilon = L/r_s$")
    plt.ylabel(r"$\chi^2$")
    plt.title("Chi-square vs. fractional core size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

