# BlackHoleParamEnv — RL Environment for Parameter Inference

This document specifies the Gymnasium environment `BlackHoleParamEnv` used to train RL agents to infer compact-object core parameters from observables (ringdown frequencies/decays and black-hole shadow size). The environment is implemented in `simulation/bh_gym_env.py` and backed by the fast, non-rendering simulation helpers in `simulation/simulation_rl.py`.

The environment is designed for speed and physical plausibility, with a concise observation/action interface that is friendly to model-free RL (e.g., PPO in Stable-Baselines3).

**Key Goals**
- Infer a core deformation parameter from data-like targets.
- Support two parameterizations: fractional epsilon (L/rs) and absolute length L0.
- Provide well-scaled observations and a statistically grounded objective (Gaussian log-likelihood).

## Motivation & Intuition
We frame parameter inference as a control problem: an agent adjusts a single “core deformation” knob and watches how astrophysical signatures react. The knob is either a fractional deformation `ε = L/rs` (dimensionless, tied to the Schwarzschild scale) or an absolute length `L0` (scene units). The simulator returns data-like observables: the dominant ringdown mode (`f220`, `τ220`) and the shadow size (either as an angular diameter `θ` with known distance, or as the physical critical impact parameter `b_c`). With optional uncertainties, we standardize residuals and optimize a Gaussian log-likelihood. Intuitively, the agent is climbing a smooth landscape: by nudging `ε` (or `L0`) so that all residuals approach zero, it minimizes chi-square and maximizes likelihood.

Why this matters: jointly leveraging ringdown and shadow connects independent probes (gravitational-wave spectroscopy and horizon-scale imaging) to test strong-field gravity, constrain deviations from GR, and bound exotic compact-object structure. A single, well-scaled objective lets heterogeneous measurements inform one another coherently via their uncertainties.

How to interpret signals: residuals near zero signal agreement with the target; `env/chi2 = Σ r_k^2` summarizes mismatch; the reward is `-0.5·chi2` so higher is better. `env/param` shows the current location in parameter space; stable, low-variance residuals and a decreasing `chi2` indicate convergence. `env/has_horizon` flags unphysical regions; persistent zeros suggest bounds or step size should be revisited.

Paper parameters and what we optimize:
- Deformation: `ε = L/rs` (fractional) or `L0` (absolute) — the scalar controlled by the agent.
- Ringdown: `f220` (Hz) and `τ220` (s) of the `l = m = 2, n = 0` mode; residuals `(obs − target)/σ` when uncertainties are provided.
- Shadow: `θ` (radians) with `distance_m`, or the physical `b_c` (meters). Either can be used alone or combined with ringdown.
- Mass scale: `M_solar` anchors ringdown baselines; defaults are used if not specified.
- Objective: minimize `χ² = Σ r_k²` (equivalently maximize Gaussian log-likelihood). Early stop when `χ² ≤ chi2_converged`.

**Notation and Units**
- `rs` denotes the Schwarzschild radius; in the fractional model, the control parameter is `ε = L / rs`.
- Mass `M_solar` is in solar masses for ringdown scaling.
- Distances are in meters. Shadow angular size uses radians.

---

**Action & Observation Spaces**
- Action: `Box(low=-1.0, high=1.0, shape=(1,), dtype=float32)`
  - Interpreted as a delta update for the scalar parameter:
    - `Δparam = action[0] * step_size * (param_high - param_low)`, then clipped to bounds.
- Observation: `Box(low=-inf, high=inf, shape=(1 + K,), dtype=float32)`
  - `[0]`: normalized parameter x in [0, 1]: `x = (param - low) / (high - low + 1e-12)`
  - `[1:1+K]`: standardized residuals for the enabled targets (see Residuals). Keys appear in a fixed order determined by provided targets: `['f220']`, `['tau220']`, `['theta']`, `['bc']`.

---

**Parameterization (EnvConfig.core_model)**
- `"fractional"` (default): controls `ε = L / rs` within `[eps_min, eps_max]`.
  - Applied via `_sim.set_core_params(eps=<value>, model="fractional")`.
- `"absolute"`: controls absolute length `L0` within `[L0_min, L0_max]` (scene units used in the simulator).
  - Applied via `_sim.set_core_params(L0=<value>, model="absolute")`.

Bounds are physics-informed; the default fractional regime is small (`ε ∈ [0, 0.15]`) where a cubic response approximation for ringdown is valid.

---

**Targets (TargetSpec)**
- Ringdown (LIGO-like):
  - `f220_Hz`, `sigma_f_Hz` — 220 mode frequency and its standard deviation (Hz)
  - `tau220_s`, `sigma_tau_s` — 220 mode damping time and its standard deviation (s)
  - `M_solar` — source mass used for ringdown scaling; if omitted, defaults to `EnvConfig.default_M_solar` (70.0)
- Shadow (EHT-like):
  - `distance_m` — luminosity distance (m), required to compare against angular diameter.
  - `theta_shadow_rad`, `sigma_theta_rad` — observed angular shadow size (radians) and its standard deviation.
  - Alternatively (direct physical size): `b_c_m`, `sigma_b_c_m` — impact-parameter critical radius (meters) and its standard deviation.

Provide any subset of targets; residuals are computed only for those entries, and weights default to 1.0 if sigmas are not given.

---

**Residuals, Chi-square, and Reward**
- Observables are queried from the backend via:
  - `obs = _sim.bh_observables(M_solar=..., distance_m=...)`
  - Expected keys include: `f220_Hz`, `tau220_s`, optionally `theta_shadow_rad`, `b_c_m`, and `has_horizon`.

- Standardized residuals (only for provided targets):
  - Ringdown:
    - `r_f = (obs.f220_Hz - f220_Hz) / (sigma_f_Hz or 1.0)`
    - `r_tau = (obs.tau220_s - tau220_s) / (sigma_tau_s or 1.0)`
  - Shadow (angular): requires `distance_m` to be set in `TargetSpec`:
    - `r_theta = (obs.theta_shadow_rad - theta_shadow_rad) / (sigma_theta_rad or 1.0)`
  - Shadow (physical size):
    - if `obs.b_c_m` is unavailable, fall back to `_sim.photon_ring_shadow_diameter(M_solar, distance_m=1.0)` where distance cancels for `b_c`.
    - `r_bc = (obs.b_c_m - b_c_m) / (sigma_b_c_m or 1.0)`

- Chi-square and reward:
  - `$\chi^2 = \sum_k r_k^2$`
  - If `use_loglike_reward=True` (default): `reward = -0.5 * chi2` (Gaussian log-likelihood up to a constant).
  - Else: `reward = -chi2`.

- Horizon sanity term:
  - If `obs.has_horizon` is `False`, a penalty of `+10.0` is added to `chi2` (discourages horizonless configurations).

- Termination & truncation:
  - `terminated = (chi2 <= chi2_converged)`
  - `truncated = (step_count >= max_steps)`

---

**Environment Configuration (EnvConfig)**
- `core_model`: `"fractional" | "absolute"` — parameterization mode.
- `eps_min`, `eps_max`: parameter bounds for fractional mode.
- `L0_min`, `L0_max`: parameter bounds for absolute mode (scene units).
- `max_steps`: episode cap for truncation.
- `chi2_converged`: threshold for early termination when the fit is “good enough”.
- `step_size`: scales the action delta; effective delta is `step_size * (high - low)`.
- `use_loglike_reward`: if `True`, use `-0.5 * chi2` (statistically grounded for Gaussian errors).
- Rendering/overlays (kept minimal for RL speed):
  - `render_width`, `render_height`, `render_device` (`"cpu"` or `"cuda"`), `enable_render_overlays`.
  - Internal defaults (applied at init) disable heavy effects: `SAMPLES_PER_PIXEL=1`, `BLOOM_GAIN=0`, `ENABLE_PHOTON_RING=False`, zeroed sky/nebula/dust/twinkle.
- Initialization controls:
  - `init_center_eps`: center for the initial parameter when fractional.
  - `init_spread`: uniform perturbation radius (in bounds space) around the center.
  - `default_M_solar`: fallback mass for ringdown scaling if `TargetSpec.M_solar` is not provided.

---

**Backend Interface (simulation_rl.py)**
The environment relies on a few backend functions and conventions (all provided by `simulation/simulation_rl.py`):
- `set_core_params(eps=<float>, model="fractional")` or `set_core_params(L0=<float>, model="absolute")`
- `apply_params(dict_like)` and `current_params_dict()` for non-rendering adjustments and state snapshots.
- `bh_observables(M_solar: float, distance_m: float)` → `dict`
  - Must include `f220_Hz`, `tau220_s`, `has_horizon`.
  - Should include `theta_shadow_rad` if `distance_m` is provided; may include `b_c_m`.
- `photon_ring_shadow_diameter(M_solar: float, distance_m: float)` → `(b_c_m, theta_shadow_rad)`
- `render_image(w: int, h: int, time_s: float, device: str)` → `ndarray[H,W,3]` (float32 RGB in [0,1])

All rendering paths are minimized during RL. The environment’s `render()` method returns a low-cost image and restores the simulation state afterward.

---

**What the Agent Learns to Optimize**
The agent controls a single scalar parameter (either `ε` or `L0`), with the objective of minimizing `$\chi^2$` (or maximizing the Gaussian log-likelihood). It receives both the current normalized parameter and standardized residuals as observations, enabling gradient-free exploration of the parameter landscape via PPO or similar algorithms.

Given the small-`ε` cubic response in the fractional model, the reward landscape is smooth around the true value, supporting stable on-policy optimization.

---

**Ringdown Scaling Model**
Let `$M$` be the source mass in solar masses and `$\varepsilon = L/ r_s$` the fractional core size. The backend uses a cubic-response approximation around small `$\varepsilon$` for the dominant QNM,

- Baseline (Schwarzschild) scalings with mass:
  - `$f_{220,0}(M) = f_{220,ref} \, \frac{M_{ref}}{M}$`
  - `$\tau_{220,0}(M) = \tau_{220,ref} \, \frac{M}{M_{ref}}$`

- Cubic fractional shifts for small `$\varepsilon$`:
  - `$f_{220}(M, \varepsilon) \approx f_{220,0}(M) \, [1 + C_f \, \varepsilon^3]$`
  - `$\tau_{220}(M, \varepsilon) \approx \tau_{220,0}(M) \, [1 + C_\tau \, \varepsilon^3]$`

Here `$C_f$` and `$C_\tau$` are dimensionless coefficients set by the backend (see `simulation_rl.py`). For absolute-length mode, `$\varepsilon$` may be derived via `$\varepsilon = L_0 / r_s$` with `$r_s$` the Schwarzschild radius under the scene’s units. Shadow observables scale accordingly; for the critical impact parameter `$b_c$`, the backend can provide both `$b_c$` and the corresponding angular size `$\theta$` given a `distance_m`.

These relations motivate the use of a Gaussian log-likelihood reward with standardized residuals, which produces a smooth objective suitable for PPO.

---

**Usage Patterns (Stable-Baselines3)**
Example (single-process evaluation):

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from simulation.bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec

target = TargetSpec(
    f220_Hz=None, tau220_s=None, M_solar=70.0,
    distance_m=None, theta_shadow_rad=None,
    b_c_m=None
)
env_conf = EnvConfig(core_model="fractional", eps_min=0.0, eps_max=0.15)

def make_env():
    env = BlackHoleParamEnv(target, env_conf)
    return env

venv = DummyVecEnv([make_env])
model = PPO("MlpPolicy", venv, verbose=1)
model.learn(total_timesteps=10000)
```

For vectorized training (`SubprocVecEnv`) and normalization (`VecNormalize`), see `train.py` in the repository.

---

**Minimal End-to-End Example**
A short PPO training run is provided in `simulation/minimal_end_to_end.py`.

Run:

```bash
python simulation/minimal_end_to_end.py --timesteps 20000 --seed 123
```

It prints evaluation statistics and writes TensorBoard logs to `tb_minimal` in the current working directory.

---

**Plot: Parameter Sweep vs. Chi-square**
`simulation/plot_param_sweep.py` constructs a self-consistent target by sampling the simulator at a ground-truth `$\varepsilon$`, then sweeps `$\varepsilon$` to plot `$\chi^2(\varepsilon)$`.

Run (requires `matplotlib`):

```bash
python simulation/plot_param_sweep.py --eps-min 0.0 --eps-max 0.15 --points 101 --seed 1337 \
  --msolar 70 --gt-eps 0.05 --out data/chi2_sweep.png
```

This helps verify the residual scaling and the location/shape of the optimum.

---

**Best Practices & Tips**
- Start with fractional mode and `ε ∈ [0, 0.15]`. Widen bounds only if the task requires it.
- Provide realistic sigmas in `TargetSpec` to shape a well-scaled objective; unit weights are assumed if omitted.
- Keep `step_size` modest (e.g., 0.01–0.02 in normalized units) to avoid oscillations.
- Enable `use_loglike_reward=True` to align training with a statistical objective.
- Inspect TensorBoard metrics:
  - `env/chi2`, `env/residual_*`, `env/param` (from environment infos)
  - `train/learning_rate`, `train/clip_range_effective`, `rollout/ep_rew_mean`, etc.
- Watch for `has_horizon=False` penalties; if persistent, the parameter bounds may be outside physical regime.

---

**Troubleshooting**
- ImportError for `simulation_rl`: ensure imports are package-qualified (`from . import simulation_rl as _sim`).
- Angular shadow residuals require `TargetSpec.distance_m`.
- If no TensorBoard entries appear, confirm `tensorboard` is installed and that the logger directory is pointed to Hydra’s run output.
- On macOS sandboxes, OMP/SHM restrictions can block imports; run outside restrictive sandboxes when validating.

---

**API Reference (Summary)**
- `class BlackHoleParamEnv(gym.Env)`
  - `reset(seed: int | None, options: dict | None) -> (obs, info)`
  - `step(action: np.ndarray) -> (obs, reward, terminated, truncated, info)`
    - `info`: `{ "chi2": float, "residuals": dict[str, float], "param": float }`
  - `render() -> np.ndarray[H,W,3]`
- `class EnvConfig`: see Environment Configuration.
- `class TargetSpec`: see Targets.

---

For further details, inspect `simulation/bh_gym_env.py` (environment logic) and `simulation/simulation_rl.py` (backend implementation and observables).

---

## Schematic
```mermaid
flowchart LR
    P[Parameter: ε = L/rs or L0] --> S[Simulator]
    S --> O[Observables: f220, τ220, θ | b_c]
    O --> R[Residuals r_k = (obs−target)/σ]
    R --> C[Chi-square χ² = Σ r_k²]
    C --> W[Reward = −0.5 · χ²]
    W --> A[PPO Agent]
    A -- action Δparam --> P
```
