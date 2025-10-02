
"""
BlackHoleParamEnv (Gymnasium) for RL-based parameter fitting — with difficulty modes.

This environment is optimized for *speed* and *physical fidelity* when training RL agents to infer
core parameters (epsilon = L / r_s, etc.) from measurable observables (ringdown f220, tau220; shadow size).
It now supports three logical difficulties:

  1) LOW / SIMPLE  — (default) Backward-compatible: single-parameter control, smooth objective,
                     Gaussian log-likelihood reward; no extra costs or discrete controls.
  2) MEDIUM        — Adds *multi-fidelity control* (discrete, non-differentiable) and per-step compute cost.
                     The agent chooses a fidelity each step; higher fidelity reduces noise/uncertainty but
                     costs more. Objective becomes piecewise/discontinuous ⇒ gradient-based fits struggle.
  3) HARD          — Full sequential-design POMDP: *query selection* (which observable to acquire),
                     *multi-fidelity*, *budgeted costs*, and *optional early terminate*. Measurements
                     persist across steps (fixed noise realizations), and reward uses only acquired data
                     until termination (final eval can optionally include full chi²). RL excels here.

All additions are *opt-in* through EnvConfig and preserve legacy behavior when difficulty="LOW".

Depends on `simulation_rl.py` (fast, non-rendering helpers).
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Literal, List, Set

try:
    import gymnasium as gym
except Exception as e:
    raise RuntimeError("Please install gymnasium: pip install gymnasium") from e

# Import the simulation helpers (fast, non-rendering) — using relative import if part of a package
try:
    from . import simulation_rl as _sim  # if placed under a package (e.g., simulation/)
except Exception:  # fallback to flat import
    import simulation_rl as _sim


Difficulty = Literal["LOW", "MEDIUM", "HARD"]


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

    # =========================
    # Difficulty extensions
    # =========================
    difficulty: Difficulty = "LOW"

    # --- MEDIUM/HARD: multi-fidelity controls (discrete via action thresholding) ---
    # We model two fidelity levels for simplicity: low/high with different costs and noise multipliers.
    fidelity_low_noise_mult: float = 1.0
    fidelity_high_noise_mult: float = 0.25
    fidelity_low_step_cost: float = 0.0
    fidelity_high_step_cost: float = 1.0

    # --- HARD: sequential querying and budget ---
    # Initial budget and cost weights. Each *query* consumes cost; fidelity also costs (per step).
    budget_init: float = 10.0
    cost_weight: float = 0.05  # λ multiplying cost in the reward (reward = -0.5 χ² - λ * cost)
    # Per-query base costs (before fidelity multiplier); set to 0 to disable.
    cost_query_f220: float = 1.0
    cost_query_tau220: float = 1.0
    cost_query_theta: float = 2.0
    cost_query_bc: float = 1.5

    # Whether terminal step should evaluate full chi² (all targets) for the final reward
    hard_terminal_full_eval: bool = True

    # Whether to include mask bits & diagnostics in observations (HARD)
    include_masks_in_obs: bool = True

    # Seed offset for per-episode noise in HARD (noise is fixed after acquisition)
    hard_noise_scale: float = 1.0  # scales sigma used to draw fixed measurement noise


class BlackHoleParamEnv(gym.Env):
    """
    Difficulty modes:

      - LOW:
            * Action space: Box(1,) → Δparam only
            * Reward: -0.5 χ² (or -χ²), smooth, no costs
            * Obs: [param_norm, standardized residuals...]
      - MEDIUM:
            * Action space: Box(2,) → [Δparam, fidelity_ctrl]
              fidelity_ctrl>0 ⇒ high fidelity; else low.
            * Reward: (-0.5 χ²) - λ * cost_fidelity (per step)
            * Obs: [param_norm, residuals..., fidelity_flag(0/1), budget_norm]
      - HARD:
            * Action space: Box(3,) → [Δparam, fidelity_ctrl, query_ctrl]
              - fidelity_ctrl>0 ⇒ high; else low
              - query_ctrl discretized to {f220, tau220, theta, bc, terminate}
            * State: set of acquired measurements with fixed noise (sampled when acquired)
            * Reward: (-0.5 χ²_acquired) - λ * (cost_fidelity + cost_query)     [or -χ² if configured]
            * Termination: when 'terminate' chosen, budget exhausted, or max_steps
            * Optional final bonus: evaluate full χ² on termination (hard_terminal_full_eval=True)
            * Obs: [param_norm, residuals(acquired or zeros), masks(if enabled), fidelity_flag, budget_norm]
    """
    metadata = {"render_modes": ["rgb_array", "none"]}

    def __init__(self,
                 target: TargetSpec,
                 config: EnvConfig | None = None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.target = target

        # Validate we can compute what we need
        self._ensure_target_consistency()

        # Parse residual keys in a canonical order
        self._residual_keys: List[str] = []
        if self.target.f220_Hz is not None:
            self._residual_keys.append("f220")
        if self.target.tau220_s is not None:
            self._residual_keys.append("tau220")
        if (self.target.theta_shadow_rad is not None) or (self.target.b_c_m is not None):
            if self.target.theta_shadow_rad is not None:
                self._residual_keys.append("theta")
            if self.target.b_c_m is not None:
                self._residual_keys.append("bc")

        if len(self._residual_keys) == 0 and self.cfg.difficulty != "HARD":
            # In HARD you could choose queries later; for LOW/MEDIUM require at least one target
            raise ValueError(
                "No targets provided: specify at least one of f220_Hz, tau220_s, "
                "theta_shadow_rad (with distance_m), or b_c_m in TargetSpec."
            )

        # Define action & observation spaces
        self._setup_spaces()

        # Internal state
        self._step_count: int = 0
        self._param: Optional[float] = None  # current parameter in native units

        # HARD-mode state
        self._budget: float = 0.0
        self._fidelity_high: bool = False
        self._acquired: Set[str] = set()     # which keys have been acquired
        self._meas_noise: Dict[str, float] = {}  # fixed measurement noise per key (sampled at acquisition)

        # Initialize sim with minimal visual overhead
        self._apply_visual_defaults()

    # --------------- Gym API ---------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._step_count = 0
        # Parameter bounds per mode (computed in _setup_spaces)
        # Initialize parameter near center
        if self.cfg.core_model == "fractional":
            center = float(np.clip(self.cfg.init_center_eps, self.cfg.eps_min, self.cfg.eps_max))
        else:
            center = 0.5 * (self.cfg.L0_min + self.cfg.L0_max)
        spread = float(self.cfg.init_spread) * (self.param_high[0] - self.param_low[0])
        self._param = float(np.clip(center + self.np_random.uniform(-spread, spread),
                                    self.param_low[0], self.param_high[0]))
        self._apply_param_to_sim(self._param)

        # Difficulty-specific resets
        if self.cfg.difficulty == "LOW":
            pass
        elif self.cfg.difficulty == "MEDIUM":
            self._fidelity_high = False  # default start on low fidelity
            self._budget = float("inf")  # cost is logged but no budget constraint
        else:  # HARD
            self._fidelity_high = False
            self._budget = float(self.cfg.budget_init)
            self._acquired.clear()
            self._meas_noise.clear()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._step_count += 1

        # Map action to controls depending on difficulty
        delta_param, fidelity_high, query_idx, want_terminate = self._map_action(action)

        # Update parameter
        new_param = float(np.clip(self._param + delta_param, self.param_low[0], self.param_high[0]))
        self._param = new_param
        self._apply_param_to_sim(self._param)

        # Compute reward, chi2, residuals under current difficulty
        cost = 0.0
        terminated = False
        truncated = False

        if self.cfg.difficulty == "LOW":
            chi2, resid = self._chi2_and_residuals(active_keys=None, use_fixed_noise=False)
            reward = self._rl_reward_from_chi2(chi2)
            terminated = (chi2 <= self.cfg.chi2_converged)

        elif self.cfg.difficulty == "MEDIUM":
            # fidelity control + per-step cost
            self._fidelity_high = bool(fidelity_high)
            chi2, resid = self._chi2_and_residuals(active_keys=None, use_fixed_noise=False)
            step_cost = self.cfg.fidelity_high_step_cost if self._fidelity_high else self.cfg.fidelity_low_step_cost
            cost += step_cost
            reward = self._rl_reward_from_chi2(chi2) - self.cfg.cost_weight * cost
            terminated = (chi2 <= self.cfg.chi2_converged)

        else:  # HARD
            # Apply fidelity choice
            self._fidelity_high = bool(fidelity_high)

            # Optional query acquisition (discrete)
            if (query_idx is not None) and (not want_terminate):
                qkey = self._query_key_from_index(query_idx)
                if qkey is not None:
                    # Acquire this measurement if target exists and not yet acquired
                    if (qkey in self._residual_keys) and (qkey not in self._acquired):
                        # Sample a fixed measurement noise for this key (based on sigma*scale)
                        noise = self._sample_measurement_noise(qkey)
                        self._meas_noise[qkey] = noise
                        self._acquired.add(qkey)
                        cost += self._query_cost(qkey)

            # Fidelity per-step cost (encourages staying low unless beneficial)
            cost += (self.cfg.fidelity_high_step_cost if self._fidelity_high else self.cfg.fidelity_low_step_cost)

            # Compute chi2 over *acquired* data only
            active_keys = list(self._acquired) if len(self._acquired) > 0 else []
            chi2, resid = self._chi2_and_residuals(active_keys=active_keys, use_fixed_noise=True)

            # Base reward
            reward = self._rl_reward_from_chi2(chi2) - self.cfg.cost_weight * cost

            # Handle termination choices/budget
            if want_terminate:
                terminated = True
                if self.cfg.hard_terminal_full_eval:
                    # Evaluate full chi2 as a terminal adjustment (encourages informative acquisitions)
                    full_chi2, _ = self._chi2_and_residuals(active_keys=None, use_fixed_noise=True)
                    # add *difference* so final step reflects full-information quality
                    reward += self._rl_reward_from_chi2(full_chi2) - self._rl_reward_from_chi2(chi2)
            # Budget update (after costs are incurred)
            self._budget -= cost
            if self._budget <= 0.0 and not terminated:
                terminated = True  # out of budget

        truncated = truncated or (self._step_count >= self.cfg.max_steps)

        # Diagnostics
        Ms = self.target.M_solar or self.cfg.default_M_solar
        sim_obs = _sim.bh_observables(M_solar=Ms,
                                      distance_m=(self.target.distance_m if (self.target.distance_m is not None) else 1.0))
        info = {
            "chi2": float(chi2),
            "residuals": {k: float(v) for k, v in resid.items()},
            "param": float(self._param),
            "has_horizon": bool(sim_obs.get("has_horizon", True)),
            "difficulty": self.cfg.difficulty,
        }
        if self.cfg.difficulty in ("MEDIUM", "HARD"):
            info.update({
                "fidelity_high": bool(self._fidelity_high),
                "step_cost": float(cost),
            })
        if self.cfg.difficulty == "HARD":
            info.update({
                "budget": float(self._budget),
                "acquired": list(sorted(self._acquired)),
            })

        return self._get_obs(resid=resid), float(reward), bool(terminated), bool(truncated), info

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

        # HARD mode can start with empty targets (queries may acquire some), but at least *one*
        # of the potential targets must be specified in TargetSpec for scoring to make sense.
        if self.cfg.difficulty == "HARD":
            if all(getattr(self.target, nm) is None for nm in
                   ["f220_Hz", "tau220_s", "theta_shadow_rad", "b_c_m"]):
                raise ValueError("HARD mode: provide at least one potential target in TargetSpec to query.")

        # Action/Observation spaces depend on difficulty; compute here
    def _setup_spaces(self):
        # Parameter bounds per mode
        if self.cfg.core_model == "fractional":
            self.param_low = np.array([self.cfg.eps_min], dtype=np.float32)
            self.param_high = np.array([self.cfg.eps_max], dtype=np.float32)
        else:
            self.param_low = np.array([self.cfg.L0_min], dtype=np.float32)
            self.param_high = np.array([self.cfg.L0_max], dtype=np.float32)

        # Action spaces by difficulty
        if self.cfg.difficulty == "LOW":
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        elif self.cfg.difficulty == "MEDIUM":
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:  # HARD
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: base param + residuals; extra features added depending on difficulty
        base_dim = 1 + len(self._residual_keys)  # param + standardized residuals (order = self._residual_keys)
        extra = 0
        if self.cfg.difficulty in ("MEDIUM", "HARD"):
            extra += 1  # fidelity flag
            extra += 1  # budget normalized (LOW omits to keep backward-compat)
        if self.cfg.difficulty == "HARD" and self.cfg.include_masks_in_obs:
            extra += len(self._residual_keys)  # mask bits for which residuals are currently acquired
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(base_dim + extra,), dtype=np.float32)

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

    # ---------- Reward/chi2 ----------
    def _rl_reward_from_chi2(self, chi2: float) -> float:
        return float(-0.5 * chi2) if self.cfg.use_loglike_reward else float(-chi2)

    def _chi2_and_residuals(self,
                            active_keys: Optional[List[str]] = None,
                            use_fixed_noise: bool = False) -> Tuple[float, Dict[str, float]]:
        """
        Compute standardized residuals and chi².

        - If active_keys is None → use *all* configured targets (legacy behavior).
        - If active_keys is [] → empty acquisition set: chi²=0 (no info yet).
        - If use_fixed_noise is True → add persistent, pre-sampled measurement noise (HARD mode).
        """
        Ms = self.target.M_solar or self.cfg.default_M_solar
        # distance 1.0 is fine for b_c (cancels), needed for theta if present
        obs = _sim.bh_observables(M_solar=Ms,
                                  distance_m=(self.target.distance_m if (self.target.distance_m is not None) else 1.0))

        keys = self._residual_keys if (active_keys is None) else list(active_keys)
        if len(keys) == 0:
            return 0.0, {k: 0.0 for k in self._residual_keys}

        chi2 = 0.0
        resid: Dict[str, float] = {}

        for k in self._residual_keys:
            if k not in keys:
                # still include a placeholder residual for observation vector (0.0)
                resid[k] = 0.0
                continue

            if k == "f220":
                if self.target.f220_Hz is None: continue
                df = float(obs["f220_Hz"]) - float(self.target.f220_Hz)
                sigma = float(self.target.sigma_f_Hz) if self.target.sigma_f_Hz else 1.0
                if use_fixed_noise and ("f220" in self._meas_noise):
                    df = df + float(self._meas_noise["f220"])
                r = df / sigma

            elif k == "tau220":
                if self.target.tau220_s is None: continue
                dtau = float(obs["tau220_s"]) - float(self.target.tau220_s)
                sigma = float(self.target.sigma_tau_s) if self.target.sigma_tau_s else 1.0
                if use_fixed_noise and ("tau220" in self._meas_noise):
                    dtau = dtau + float(self._meas_noise["tau220"])
                r = dtau / sigma

            elif k == "theta":
                if self.target.theta_shadow_rad is None: continue
                theta = obs.get("theta_shadow_rad", None)
                if theta is None:
                    raise RuntimeError("theta_shadow_rad not computed – distance missing?")
                dth = float(theta) - float(self.target.theta_shadow_rad)
                sigma = float(self.target.sigma_theta_rad) if self.target.sigma_theta_rad else 1.0
                if use_fixed_noise and ("theta" in self._meas_noise):
                    dth = dth + float(self._meas_noise["theta"])
                r = dth / sigma

            elif k == "bc":
                if self.target.b_c_m is None:
                    # Compute directly if not provided empirically (distance cancels)
                    bc, _ = _sim.photon_ring_shadow_diameter(Ms, distance_m=1.0)
                    # If no target.b_c_m, skip residual but keep placeholder
                    resid["bc"] = 0.0
                    continue
                bc = obs.get("b_c_m", None)
                if bc is None:
                    bc, _ = _sim.photon_ring_shadow_diameter(Ms, distance_m=1.0)
                dbc = float(bc) - float(self.target.b_c_m)
                sigma = float(self.target.sigma_b_c_m) if self.target.sigma_b_c_m else 1.0
                if use_fixed_noise and ("bc" in self._meas_noise):
                    dbc = dbc + float(self._meas_noise["bc"])
                r = dbc / sigma

            else:
                continue

            resid[k] = float(r)
            chi2 += float(r*r)

        # Horizon sanity term (LOW/MEDIUM legacy behavior: always applied; HARD: only when scoring any keys)
        if (active_keys is None) or (len(keys) > 0):
            if not bool(obs.get("has_horizon", True)):
                chi2 += 10.0

        return float(chi2), resid

    # ---------- Difficulty-specific helpers ----------
    def _map_action(self, action: np.ndarray) -> Tuple[float, Optional[bool], Optional[int], bool]:
        """
        Returns (delta_param, fidelity_high?, query_index?, want_terminate?)
        - For LOW:     (Δ, None, None, False)
        - For MEDIUM:  (Δ, high?, None, False)
        - For HARD:    (Δ, high?, q_idx or None, terminate?)
        """
        # Δ in native units
        delta = float(action[0]) * self.cfg.step_size * (self.param_high[0] - self.param_low[0])

        if self.cfg.difficulty == "LOW":
            return delta, None, None, False

        if self.cfg.difficulty == "MEDIUM":
            fidelity_high = bool(action[1] > 0.0)
            return delta, fidelity_high, None, False

        # HARD
        fidelity_high = bool(action[1] > 0.0)
        # Discretize query selector into bins over available keys + one extra for TERMINATE
        n_bins = len(self._residual_keys) + 1
        x = float(action[2])
        # map [-1,1] -> {0,1,...,n_bins-1}
        idx = int(np.floor((x + 1.0) / 2.0 * n_bins))
        idx = int(np.clip(idx, 0, n_bins - 1))
        want_terminate = (idx == n_bins - 1)
        q_idx = None if want_terminate else idx
        return delta, fidelity_high, q_idx, want_terminate

    def _query_key_from_index(self, idx: int) -> Optional[str]:
        if idx < 0 or idx >= len(self._residual_keys):
            return None
        return self._residual_keys[idx]

    def _sigma_for_key(self, key: str) -> float:
        if key == "f220":
            return float(self.target.sigma_f_Hz) if self.target.sigma_f_Hz else 1.0
        if key == "tau220":
            return float(self.target.sigma_tau_s) if self.target.sigma_tau_s else 1.0
        if key == "theta":
            return float(self.target.sigma_theta_rad) if self.target.sigma_theta_rad else 1.0
        if key == "bc":
            return float(self.target.sigma_b_c_m) if self.target.sigma_b_c_m else 1.0
        return 1.0

    def _sample_measurement_noise(self, key: str) -> float:
        """
        Draw a fixed additive noise for a newly acquired measurement.
        Scaled by the key's sigma and fidelity noise multiplier.
        """
        base_sigma = self._sigma_for_key(key)
        mult = self.cfg.fidelity_high_noise_mult if self._fidelity_high else self.cfg.fidelity_low_noise_mult
        std = max(1e-12, float(base_sigma) * float(mult) * float(self.cfg.hard_noise_scale))
        # Use the gym RNG for reproducibility under seed
        return float(self.np_random.normal(loc=0.0, scale=std))

    def _query_cost(self, key: str) -> float:
        base = {
            "f220":  self.cfg.cost_query_f220,
            "tau220": self.cfg.cost_query_tau220,
            "theta": self.cfg.cost_query_theta,
            "bc":    self.cfg.cost_query_bc,
        }.get(key, 0.0)
        fidelity_mult = 1.0 if (not self._fidelity_high) else 2.0  # high fidelity twice as expensive
        return float(base) * float(fidelity_mult)

    # ---------- Observation packing ----------
    def _get_obs(self, resid: Optional[Dict[str, float]] = None) -> np.ndarray:
        # Param normalized 0..1 within bounds
        x = (self._param - self.param_low[0]) / (self.param_high[0] - self.param_low[0] + 1e-12)
        obs: List[float] = [float(x)]

        if resid is None:
            # LOW/MEDIUM: full set; HARD: acquired-only (fixed noise) but return placeholders for others
            if self.cfg.difficulty == "HARD":
                _, resid = self._chi2_and_residuals(active_keys=list(self._acquired), use_fixed_noise=True)
            else:
                _, resid = self._chi2_and_residuals(active_keys=None, use_fixed_noise=False)

        # Residuals in canonical order (placeholders already included by _chi2_and_residuals)
        for k in self._residual_keys:
            obs.append(float(resid.get(k, 0.0)))

        # Extra features for MEDIUM/HARD
        if self.cfg.difficulty in ("MEDIUM", "HARD"):
            obs.append(1.0 if self._fidelity_high else 0.0)  # fidelity flag
            # Normalized budget (LOW omits; MEDIUM logs +inf as 1.0)
            if self.cfg.difficulty == "MEDIUM":
                obs.append(1.0)  # treat as full budget (no constraint)
            else:
                obs.append(float(max(0.0, self._budget) / (self.cfg.budget_init + 1e-12)))

        # Mask bits (HARD only, optional)
        if self.cfg.difficulty == "HARD" and self.cfg.include_masks_in_obs:
            for k in self._residual_keys:
                obs.append(1.0 if (k in self._acquired) else 0.0)

        return np.array(obs, dtype=np.float32)
