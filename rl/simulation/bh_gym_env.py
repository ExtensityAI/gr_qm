
"""
BlackHoleParamEnv (Gymnasium) for RL-based parameter fitting.

This environment is optimized for *speed* and *physical fidelity*
when training RL agents to infer core parameters (epsilon = L / r_s, etc.)
from measurable observables (ringdown f220, tau220; shadow diameter).

It depends on `simulation_rl.py` (or the original `simulation.py` if you
manually add the same helper functions there).
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Literal

try:
    import gymnasium as gym
except Exception as e:
    raise RuntimeError("Please install gymnasium: pip install gymnasium") from e

# Import the simulation helpers (fast, non-rendering)
from . import simulation_rl as _sim


@dataclass
class TargetSpec:
    # Provide any subset of targets. If sigma_* is None, unit-weights are used.
    # Ringdown (LIGO-like)
    f220_Hz: Optional[float] = None
    sigma_f_Hz: Optional[float] = None
    tau220_s: Optional[float] = None
    sigma_tau_s: Optional[float] = None
    M_solar: Optional[float] = None  # mass used for baseline scaling (must be > 0 for ringdown)

    # Shadow (EHT-like)
    distance_m: Optional[float] = None
    theta_shadow_rad: Optional[float] = None
    sigma_theta_rad: Optional[float] = None

    # You can also target b_c directly (meters) if desired
    b_c_m: Optional[float] = None
    sigma_b_c_m: Optional[float] = None


@dataclass
class EnvConfig:
    # Parameterization: we support fractional ("fractional": eps = L/rs) or absolute ("absolute": L0)
    core_model: Literal["fractional", "absolute"] = "fractional"

    # Bounds (physics-informed)
    # Epsilon is small in the "cubic response" regime; keep within ~0..0.15 by default
    eps_min: float = 0.0
    eps_max: float = 0.15

    # If using absolute-L model, this bound is in "scene units". Only relevant if core_model=="absolute".
    L0_min: float = 0.0
    L0_max: float = 0.6

    # Steps per episode and termination
    max_steps: int = 64
    # Considered converged when the (weighted) chi2 per DOF falls below this threshold
    chi2_converged: float = 1e-2

    # Action is a delta in normalized space (-1..1). step_size scales it in parameter units.
    step_size: float = 0.02

    # Reward shaping
    # Use Gaussian log-likelihood by default (i.e., reward = -0.5 * chi2 - const)
    use_loglike_reward: bool = True

    # Rendering controls (kept intentionally minimal & cheap)
    render_width: int = 256
    render_height: int = 144
    render_device: Optional[str] = "cpu"  # 'cuda' or 'cpu'
    enable_render_overlays: bool = False  # disable post/bloom/photon overlays for speed

    # Random init
    init_center_eps: float = 0.05   # good default within linear-cubic validity
    init_spread: float = 0.02       # uniform random perturbation around center

    # Mass prior for ringdown scaling if not present in TargetSpec (falls back to 70 Msun)
    default_M_solar: float = 70.0


class BlackHoleParamEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "none"]}

    def __init__(self,
                 target: TargetSpec,
                 config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.target = target

        # Validate we can compute what we need
        self._ensure_target_consistency()

        # Action/Observation spaces
        if self.cfg.core_model == "fractional":
            # Single control parameter: epsilon
            self.param_low = np.array([self.cfg.eps_min], dtype=np.float32)
            self.param_high = np.array([self.cfg.eps_max], dtype=np.float32)
        else:
            self.param_low = np.array([self.cfg.L0_min], dtype=np.float32)
            self.param_high = np.array([self.cfg.L0_max], dtype=np.float32)

        # We use delta-actions normalized to [-1, 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observations: current param (normalized 0..1 within bounds) and residual features
        # We include residuals for any targets provided, standardized by sigmas if available.
        self.obs_size = 1  # param
        self._residual_keys = []
        if self.target.f220_Hz is not None:
            self._residual_keys.append("f220")
        if self.target.tau220_s is not None:
            self._residual_keys.append("tau220")
        if self.target.theta_shadow_rad is not None or self.target.b_c_m is not None:
            if self.target.theta_shadow_rad is not None:
                self._residual_keys.append("theta")
            if self.target.b_c_m is not None:
                self._residual_keys.append("bc")
        self.obs_size += len(self._residual_keys)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.obs_size,), dtype=np.float32)

        # Internal state
        self._step_count = 0
        self._param = None  # current parameter in native units

        # Safety: ensure at least one residual target is active
        if len(self._residual_keys) == 0:
            raise ValueError(
                "No targets provided: specify at least one of f220_Hz, tau220_s, "
                "theta_shadow_rad (with distance_m), or b_c_m in TargetSpec."
            )

        # Initialize sim with minimal visual overhead
        self._apply_visual_defaults()

    # --------------- Gym API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        # Initialize parameter near center
        if self.cfg.core_model == "fractional":
            center = np.clip(self.cfg.init_center_eps, self.cfg.eps_min, self.cfg.eps_max)
        else:
            center = 0.5 * (self.cfg.L0_min + self.cfg.L0_max)
        spread = self.cfg.init_spread * (self.param_high[0] - self.param_low[0])
        self._param = float(np.clip(center + self.np_random.uniform(-spread, spread),
                                    self.param_low[0], self.param_high[0]))

        # Apply to simulation
        self._apply_param_to_sim(self._param)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        # Delta in native units
        delta = float(action[0]) * self.cfg.step_size * (self.param_high[0] - self.param_low[0])
        new_param = float(np.clip(self._param + delta, self.param_low[0], self.param_high[0]))
        self._param = new_param
        self._apply_param_to_sim(self._param)

        # Compute reward & info
        chi2, resid = self._chi2_and_residuals()
        if self.cfg.use_loglike_reward:
            reward = -0.5 * chi2  # constant term drops out for RL
        else:
            reward = -chi2

        terminated = (chi2 <= self.cfg.chi2_converged)  # solved to precision
        truncated = (self._step_count >= self.cfg.max_steps)

        # Include useful diagnostics in info
        Ms = self.target.M_solar or self.cfg.default_M_solar
        sim_obs = _sim.bh_observables(M_solar=Ms, distance_m=(self.target.distance_m if (self.target.distance_m is not None) else 1.0))
        info = {
            "chi2": chi2,
            "residuals": resid,
            "param": self._param,
            "has_horizon": bool(sim_obs.get("has_horizon", True)),
        }
        return self._get_obs(resid=resid), reward, terminated, truncated, info

    def render(self):
        # Very low-cost render: CPU, low res, 1 spp, overlays off
        # We temporarily apply minimal visual params, render, and return the image.
        # Delay heavy effects for RL speed.
        # Save current params and restore after
        prev = _sim.current_params_dict()
        try:
            low_cost = dict(
                MAX_STEPS=128, _comment="low-cost render overrides",
                SAMPLES_PER_PIXEL=1,
                ENABLE_PHOTON_RING=False,
                BLOOM_GAIN=0.0, BLOOM_SIGMA=0.0,
                SKY_DENSITY=0.0, STAR_GAIN=0.0,
                DUST_GAIN=0.0, DUST_ALPHA=0.0,
                NEBULA_FLOW_RATE=0.0, WARP_TIME_RATE=0.0, TWINKLE_RATE=0.0,
                QUIET=True,
            )
            _sim.apply_params(low_cost)
            img = _sim.render_image(self.cfg.render_width, self.cfg.render_height,
                                    time_s=0.0, device=self.cfg.render_device or "cpu")
            # img is a numpy array HxWx3 float32
            return img
        finally:
            _sim.apply_params(prev)  # restore

    # --------------- Internals ---------------
    def _ensure_target_consistency(self):
        # Ringdown requires M
        if (self.target.f220_Hz is not None or self.target.tau220_s is not None):
            if not self.target.M_solar:
                self.target.M_solar = self.cfg.default_M_solar

        # Shadow requirements
        if (self.target.theta_shadow_rad is not None) or (self.target.b_c_m is not None):
            if not self.target.distance_m and self.target.theta_shadow_rad is not None:
                raise ValueError("distance_m is required to compare shadow angle.")

    def _apply_visual_defaults(self):
        # Turn off cost-heavy visuals globally
        vis = dict(
            SAMPLES_PER_PIXEL=1,
            ENABLE_PHOTON_RING=False,
            BLOOM_GAIN=0.0, BLOOM_SIGMA=0.0,
            SKY_DENSITY=0.0, STAR_GAIN=0.0,
            DUST_GAIN=0.0, DUST_ALPHA=0.0,
            NEBULA_FLOW_RATE=0.0, WARP_TIME_RATE=0.0, TWINKLE_RATE=0.0,
        )
        _sim.apply_params(vis)

    def _apply_param_to_sim(self, value: float):
        if self.cfg.core_model == "fractional":
            _sim.set_core_params(eps=float(value), model="fractional")
        else:
            _sim.set_core_params(L0=float(value), model="absolute")

    def _get_obs(self, resid: Optional[Dict[str, float]] = None) -> np.ndarray:
        # Param normalized 0..1 within bounds
        x = (self._param - self.param_low[0]) / (self.param_high[0] - self.param_low[0] + 1e-12)
        obs = [x]
        if resid is None:
            _, resid = self._chi2_and_residuals()
        # Append residuals (standardized)
        for k in self._residual_keys:
            obs.append(float(resid[k]))
        return np.array(obs, dtype=np.float32)

    def _chi2_and_residuals(self) -> Tuple[float, Dict[str, float]]:
        # Compute observables using current sim parameters
        Ms = self.target.M_solar or self.cfg.default_M_solar
        obs = _sim.bh_observables(M_solar=Ms,
                                  distance_m=(self.target.distance_m if (self.target.distance_m is not None) else 1.0))

        chi2 = 0.0
        resid = {}

        # Ringdown
        if self.target.f220_Hz is not None:
            df = obs["f220_Hz"] - float(self.target.f220_Hz)
            sigma = float(self.target.sigma_f_Hz) if self.target.sigma_f_Hz else 1.0
            resid["f220"] = df / sigma
            chi2 += (df / sigma)**2
        if self.target.tau220_s is not None:
            dtau = obs["tau220_s"] - float(self.target.tau220_s)
            sigma = float(self.target.sigma_tau_s) if self.target.sigma_tau_s else 1.0
            resid["tau220"] = dtau / sigma
            chi2 += (dtau / sigma)**2

        # Shadow
        if self.target.theta_shadow_rad is not None:
            theta = obs.get("theta_shadow_rad", None)
            if theta is None:
                raise RuntimeError("theta_shadow_rad not computed – distance missing?")
            dth = theta - float(self.target.theta_shadow_rad)
            sigma = float(self.target.sigma_theta_rad) if self.target.sigma_theta_rad else 1.0
            resid["theta"] = dth / sigma
            chi2 += (dth / sigma)**2

        if self.target.b_c_m is not None:
            bc = obs.get("b_c_m", None)
            if bc is None:
                # Compute directly without angle
                bc, _ = _sim.photon_ring_shadow_diameter(Ms, distance_m=1.0)  # distance cancels for bc
            dbc = bc - float(self.target.b_c_m)
            sigma = float(self.target.sigma_b_c_m) if self.target.sigma_b_c_m else 1.0
            resid["bc"] = dbc / sigma
            chi2 += (dbc / sigma)**2

        # Horizon sanity term (optional): penalize if no horizon
        if not obs["has_horizon"]:
            chi2 += 10.0  # strong penalty to discourage horizonless configs

        return float(chi2), resid
