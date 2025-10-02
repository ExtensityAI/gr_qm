
# Black-Hole Parameter Fitting with RL (Gymnasium + PPO)

This package provides a fast, **research-grade** reinforcement learning setup to infer and optimize
UV-regular core parameters of a black hole model from **ringdown** and/or **shadow** observables.
It wraps the simulation in a light-weight **Gymnasium** environment optimized for training speed
(no cinematic rendering) and bundles a **Stable-Baselines3 PPO** training pipeline controlled by **Hydra**.

> Core idea: the agent adjusts the core parameter (fractional `epsilon=L/r_s` or absolute `L0`)
> to minimize a physically motivated error (χ²) between model observables and the target data.

---

## Contents

```
rl/
├─ train.py                 # Hydra entrypoint (returns mean eval reward for sweeps)
├─ eval.py                  # Load and evaluate a saved model
├─ callbacks.py             # Custom logging of χ², residuals, current param
├─ utils.py                 # Env factory and detailed evaluator
├─ requirements.txt         # All deps (Gymnasium, SB3, Hydra, Optuna, PyTorch, etc.)
├─ conf/
│  ├─ config.yaml           # Master config (env/targets/PPO/logging)
│  └─ hydra/
│     └─ optuna.yaml        # Optuna sweeper defaults
└─ simulation/
   ├─ __init__.py
   ├─ bh_gym_env.py
   ├─ simulation_rl.py
   └─ example_usage.py
```
Project root:
```
bh_gym_env.py               # The Gymnasium environment
simulation_rl.py            # Simulation + RL-friendly helpers (no behavior changes)
Ringdown_Bound_paper.pdf    # Reference paper (derives cubic-response coefficients)
```

---

## Install

```bash
# from your project root
pip install -r requirements.txt
```

Python 3.10+ recommended. Make sure `bh_gym_env.py` and `simulation_rl.py` are importable (stay at repo root).

---

## Physics model (short)

- **Fractional core size**:  \(\epsilon = L / r_s\) (default). Small \(\epsilon \in [0, 0.15]\) keeps the **cubic-response** regime valid.
- **Ringdown 220 mode (Schwarzschild baseline)** using Berti et al.:
  \[ \omega M \simeq 0.37367168 - 0.08896232\,i \]
  \[ f_0(M) = \frac{\Re(\omega)}{2\pi}\frac{c^3}{G M}, \qquad \tau_0(M)=\frac{1}{|\Im(\omega)|}\frac{G M}{c^3} \]
- **Core response (cubic)** from the paper (already in the sim):
  \[ \frac{\delta f}{f} = {\tt CORE\_CF}\,\epsilon^3,\quad \frac{\delta\tau}{\tau} = {\tt CORE\_CTAU}\,\epsilon^3 \]
  (defaults: `CORE_CF=0.248`, `CORE_CTAU=0.608`; adjust in your simulation if required).
- **Photon ring / shadow** (approx. with cubic correction):
  \[ b_c \approx 3\sqrt{3}\,M\left(1 + \frac{8}{27}\epsilon^3\right), \qquad
     \theta_{\rm sh} \approx \frac{2\,b_c}{D} \]
- Horizon sanity: if no horizon is detected, the environment adds a penalty to χ² (+10 by default).

Units:
- \(f_{220}\) in Hz, \(\tau_{220}\) in seconds, \(M\) in solar masses (`M_solar`), distance \(D\) in meters,
  \(b_c\) in meters, \(\theta\) in radians.

---

## The Gym Environment

**Class**: `BlackHoleParamEnv(target: TargetSpec, config: EnvConfig)` in `bh_gym_env.py`.

### Observation space
A 1D vector:
`[ param_normalized ; standardized residuals...]`

- `param_normalized ∈ [0, 1]` — the current parameter (ε or L0) linearly normalized by bounds.
- Residuals are included **only** for targets you provide, each divided by its σ when given:
  - `f220` — \((f_{220}^{\text{pred}} - f_{220}^{\text{target}})/\sigma_f\)
  - `tau220` — \((\tau^{\text{pred}} - \tau^{\text{target}})/\sigma_\tau\)
  - `theta` — \((\theta_{\rm sh}^{\text{pred}} - \theta_{\rm sh}^{\text{target}})/\sigma_\theta\)
  - `bc` — \((b_c^{\text{pred}} - b_c^{\text{target}})/\sigma_{b_c}\)

### Action space
`Box(low=-1, high=1, shape=(1,))` — a **delta** action in normalized space; internally scaled by `step_size`
and mapped to native units (ε or L0).

### Reward
By default:
\[
r = -\frac{1}{2}\chi^2, \quad \text{where }\chi^2=\sum_k \left(\frac{\Delta_k}{\sigma_k}\right)^2
\]
If a σ is not provided, \( \sigma_k=1 \) is used (unit-weight). Set `env.use_loglike_reward=false` to use `-χ²`.

### Difficulty Modes (LOW | MEDIUM | HARD)
The environment supports progressively harder settings via `env.difficulty`:
- `LOW` (default): legacy behavior — 1D action (Δparam), smooth objective, no costs.
- `MEDIUM`: adds multi-fidelity control; action is 2D `[Δparam, fidelity_ctrl]`, per-step fidelity cost added; observations include fidelity flag and normalized budget placeholder.
- `HARD`: adds query selection and budget; action is 3D `[Δparam, fidelity_ctrl, query_ctrl]` where query discretizes to {f220, tau220, theta, bc, terminate}. Observations include acquired residuals with fixed noise, masks, fidelity flag, and normalized budget.

Key env fields (see `config.yaml`): `difficulty`, `cost_weight`, fidelity noise/costs, `budget_init`, `hard_terminal_full_eval`, `include_masks_in_obs`.

### Termination / Truncation
- `terminated` when `chi2 <= env.chi2_converged` (solved to a precision).
- `truncated` when `step_count >= env.max_steps`.

### `info` dictionary
- `chi2`: current χ²
- `residuals`: dict by component (`f220`, `tau220`, `theta`, `bc`)
- `param`: the current parameter value in **native units** (ε or L0)

### Cheap rendering (optional)
`env.render()` returns a **low-res CPU image** (256×144 by default) with all costly effects disabled
(bloom/nebula/stars etc.). Not used in training; intended for occasional debugging/visual checks.

---

## Configuration (Hydra)

The master config lives in `conf/config.yaml`. You can override any value on the CLI.

### Top-level
| Key | Type | Description |
|---|---|---|
| `seed` | int | RNG seed for reproducibility |
| `device` | str | `auto`, `cpu`, or `cuda` |
| `n_envs` | int | Number of parallel vector envs (`SubprocVecEnv` if >1) |

### `env` (EnvConfig)
| Key | Type | Default | Description |
|---|---|---:|---|
| `core_model` | str | `fractional` | `"fractional"` (ε = L/rs) or `"absolute"` (L0 in scene units) |
| `eps_min, eps_max` | float | `0.0, 0.15` | Bounds for ε (cubic regime recommended) |
| `L0_min, L0_max` | float | `0.0, 0.6` | Bounds for absolute-L model (scene units) |
| `max_steps` | int | `64` | Episode horizon |
| `chi2_converged` | float | `1e-2` | Success threshold on χ² |
| `step_size` | float | `0.01` | Scales delta action in native units |
| `use_loglike_reward` | bool | `true` | Use `-0.5*χ²` if true; else `-χ²` |
| `difficulty` | str | `LOW` | Difficulty: `LOW`, `MEDIUM`, `HARD` |
| Fidelity and cost params | — | see file | Multi-fidelity costs/noise, query costs, budget |
| `render_width,height` | int | `256, 144` | Cheap render size (debug only) |
| `render_device` | str | `cpu` | Renderer device for debug previews |
| `enable_render_overlays` | bool | `false` | Keep false for speed |
| `init_center_eps` | float | `0.05` | Initial ε center (if fractional model) |
| `init_spread` | float | `0.02` | Random init spread around center |
| `default_M_solar` | float | `70.0` | Mass used if target omits `M_solar` |

### `target` (TargetSpec)
Provide any subset; residuals computed only for the values you set.

| Key | Type | Units | Notes |
|---|---|---|---|
| `f220_Hz`, `sigma_f_Hz` | float | Hz | Ringdown frequency and σ |
| `tau220_s`, `sigma_tau_s` | float | s | Ringdown damping time and σ |
| `M_solar` | float | \(M_\odot\) | Required for ringdown scaling if ringdown targets present |
| `distance_m` | float | m | Needed if using shadow angle |
| `theta_shadow_rad`, `sigma_theta_rad` | float | rad | Shadow angular diameter and σ |
| `b_c_m`, `sigma_b_c_m` | float | m | Photon-ring critical impact parameter and σ |

### `ppo` (Stable-Baselines3)
| Key | Default | Notes |
|---|---:|---|
| `policy` | `MlpPolicy` | |
| `learning_rate` | `3e-4` | Can be a float or a callable schedule |
| `n_steps` | `2048` | Rollout length per env |
| `batch_size` | `64` | |
| `n_epochs` | `10` | |
| `gamma` | `0.995` | Discount |
| `gae_lambda` | `0.95` | |
| `clip_range` | `0.2` | |
| `ent_coef` | `0.0` | |
| `vf_coef` | `0.5` | |
| `max_grad_norm` | `0.5` | |
| `net_arch` | `[256,256]` | Policy & value MLP layers |

### `train`
| Key | Default | Notes |
|---|---:|---|
| `total_timesteps` | `250000` | Total env steps across vector envs |
| `progress_bar` | `true` | SB3 progress bar |

### `eval`
| Key | Default | Notes |
|---|---:|---|
| `eval_freq` | `10000` | Steps between evals |
| `n_episodes` | `10` | Per-eval rollouts |

Evaluate a saved model with a profile:
- `python eval.py --model-dir models --episodes 20 --profile hard`

### `vecnormalize`
| Key | Default | Notes |
|---|---:|---|
| `enabled` | `true` | Use SB3 `VecNormalize` |
| `norm_obs` | `true` | Normalize observations |
| `norm_reward` | `false` | Keep raw reward (already well-scaled) |
| `clip_obs` | `10.0` | |
| `gamma` | `0.995` | Discount used for returns norm |

### `logging`
All paths are relative to the Hydra run dir.

| Key | Default | Notes |
|---|---|---|
| `tensorboard_dir` | `tb` | TensorBoard logs |
| `monitor_dir` | `monitors` | Gym Monitor CSVs |
| `checkpoint_dir` | `checkpoints` | Periodic model checkpoints |
| `best_model_dir` | `best` | Best-by-eval model |
| `eval_log_dir` | `eval_logs` | Eval callback logs |

### Difficulty Presets
`config.yaml` includes `difficulty_presets` and a `difficulty_profile` selector:
- `difficulty_profile: medium` applies MEDIUM env settings and suggests PPO tweaks (e.g., `ent_coef`).
- `difficulty_profile: hard` applies HARD env, budgeted queries, and PPO tweaks (longer rollouts).

Use from CLI (Hydra overrides):
- MEDIUM: `python train.py difficulty_profile=medium`
- HARD: `python train.py difficulty_profile=hard`
Or set `env.difficulty=MEDIUM|HARD` directly with manual overrides.
| `model_dir` | `models` | Final model & summary JSON |

### `checkpoint`
| Key | Default | Notes |
|---|---:|---|
| `enabled` | `true` | |
| `save_freq` | `50000` | Steps between saves |
| `name_prefix` | `"ppo_blackhole"` | |

---

## Quick Start (single run)

1) Edit targets in `conf/config.yaml` (see examples above).
2) Train:
```bash
python train.py \
  env.core_model=fractional env.eps_min=0.0 env.eps_max=0.15 \
  n_envs=8 seed=123 \
  ppo.learning_rate=3e-4 ppo.n_steps=2048 ppo.batch_size=64 \
  train.total_timesteps=750000
```
3) TensorBoard:
```bash
tensorboard --logdir outputs
```
4) Evaluate the saved model:
```bash
python eval.py
```

---

## Hyperparameter Search

### Hydra multirun (grid/random)
```bash
python train.py -m \
  seed=0,1,2 \
  n_envs=4,8 \
  ppo.learning_rate=1e-4,3e-4,1e-3 \
  ppo.n_steps=1024,2048 \
  ppo.batch_size=64,128
```

### Optuna sweeps
Activate the Optuna sweeper:
```bash
python train.py -m hydra/sweeper=optuna \
  hydra.sweeper.n_trials=30 hydra.sweeper.direction=maximize
```
The default search space is defined in `conf/hydra/optuna.yaml`. You can override any bound on the CLI.

**Objective**: `train.py` returns the **mean evaluation reward** so Optuna maximizes it automatically.

---

## Best Practices (to maximize reward / minimize χ²)

- **Stay in the cubic-response regime** for fractional cores: ε in **[0, 0.15]** by default.
- Provide realistic **σ** for each observable to weight residuals correctly.
- Use `vecnormalize.enabled=true` (default) so the policy sees a stationary residual distribution.
- Scale `n_envs` and `n_steps` together; keep `batch_size` s.t. \(n_{\rm envs}\cdot n_{\rm steps} \approx 8\times\) `batch_size` as a rule of thumb.
- If convergence stalls, try: increase `clip_range` slightly (0.25–0.3), reduce `ent_coef`, or simplify `net_arch` to `[128,128]`.
- Don’t render in the training loop. Rendering is off-policy and disabled by default in the env.

---

## Reproducibility & Artifacts

Hydra creates a new timestamped run directory under `outputs/…` containing:
- TensorBoard logs (`tb/`)
- Gym Monitor CSVs (`monitors/`)
- Checkpoints (`checkpoints/`), best model (`best/`)
- Final model and normalization (`models/ppo_blackhole.zip`, `models/vecnormalize.pkl`)
- Final evaluation summary (`models/final_eval.json`)

To evaluate later:
```bash
python eval.py
# or programmatically: see eval.py:load_and_eval(...)
```

---

## Notes / Troubleshooting

- **PyTorch deprecation**: add `dim=-1` to all `torch.cross` uses (see code comments), or switch to `torch.linalg.cross`.
- **Horizon penalty**: If you intentionally explore horizonless configs, decrease or remove the +10 penalty in the env.
- **Absolute core model**: Set `env.core_model=absolute` and adjust `L0_min/L0_max` (scene units). Make sure your downstream observables are consistent with absolute scaling.
- **Numerical stability**: If your targets are extremely precise (very small σ), consider reducing `step_size`, increasing `n_steps`, and enabling more envs to stabilize GAE.

---


---

## Nested layout: `simulation/`

If you prefer to keep the environment and sim helpers together, you can place them under
`simulation/` (this repository includes that layout):

```
rl/
├─ train.py
├─ eval.py
├─ callbacks.py
├─ utils.py
├─ requirements.txt
├─ conf/
│  ├─ config.yaml
│  └─ hydra/optuna.yaml
└─ simulation/
   ├─ __init__.py
   ├─ bh_gym_env.py
   ├─ simulation_rl.py
   └─ example_usage.py
```

**Imports:** you already updated imports for this layout (e.g., `from simulation.bh_gym_env import BlackHoleParamEnv`).
As a reminder, you have two options:

1) **Run from `rl/` as CWD** (recommended):
```bash
cd rl
python train.py
python simulation/example_usage.py
```

2) **Or set `PYTHONPATH`** to include `rl/` and run from anywhere:
```bash
export PYTHONPATH=/path/to/your/repo/rl:$PYTHONPATH
python /path/to/your/repo/rl/train.py
```

Hydra will still create per-run output directories under `outputs/...`. The environment functionality and
all physics/observables are identical; this is just an organizational change.

## References

- **Paper**: *Ringdown Bounds on UV‑Regularized Black‑Hole Cores* — see `Ringdown_Bound_paper.pdf` (derives cubic response in \(f_{220}\) and \(\tau\) and photon-ring correction).
- **Simulation helpers**: `simulation_rl.py` provides:
  `qnm_220_baseline(M_solar)`, `qnm_220_with_core(M_solar)`, `photon_ring_shadow_diameter(M_solar, distance_m)`, `bh_observables(...)`, `has_horizon()`, and `set_core_params(...)`.

---

## Citation
If you use this environment or training pipeline in a publication, please cite the paper above and reference this repository.

---

## Batch Multi-Seed Runs with Confidence Intervals

Use `scripts/run_seeds.py` to run multiple seeds, aggregate TensorBoard scalars, and plot mean ± 95% CI.

Examples:
- Train and aggregate 4 seeds:
  - `python scripts/run_seeds.py --seeds 0 1 2 3 --total-timesteps 750000 --n-envs 8`
- Aggregate existing runs only:
  - `python scripts/run_seeds.py --no-train`
- Override PPO hyperparameters:
  - `python scripts/run_seeds.py --learning-rate 3e-4 --batch-size 64 --n-steps 2048`

Smart defaults:
- Auto-discovers TensorBoard runs under `outputs/` (falls back to `tb/`).
- Auto-selects common metrics present (e.g., `env/chi2`, `rollout/ep_rew_mean`, `eval/mean_reward`).
- Picks the latest run per base name (e.g., `ppo_seed123_nenv8_*`).

Smoothing and resampling options:
- Per-run smoothing before aggregation:
  - Moving average: `--smooth-window 21`
  - Exponential moving average: `--ema-alpha 0.2`
- Resample each run to a common step grid for fair alignment:
  - Number of points: `--resample 200` (default)
  - Domain: `--resample-domain overlap|union` (default `overlap`)

Outputs are saved in `outputs/aggregates/` as CSVs and PNGs, plus a `summary.json` with discovered runs and metrics.

---

## Sweeps (Optuna) with Difficulty

Run Optuna sweeps over PPO and env parameters, including difficulty:

- General sweep (searches env.difficulty across LOW/MEDIUM/HARD):
  - `python train.py -m hydra/sweeper=optuna hydra.sweeper.n_trials=40`
- Fix difficulty to MEDIUM (restricts search to MEDIUM):
  - `python train.py -m hydra/sweeper=optuna env.difficulty=MEDIUM hydra.sweeper.n_trials=40`
- Fix difficulty to HARD and expand rollout length candidates:
  - `python train.py -m hydra/sweeper=optuna env.difficulty=HARD ppo.n_steps=4096,6144,8192 hydra.sweeper.n_trials=40`

Profiles (templates):
- Medium template ranges: see `optuna_medium.yaml`
- Hard template ranges: see `optuna_hard.yaml`

You can copy these ranges into CLI overrides, e.g. (MEDIUM):
- `python train.py -m hydra/sweeper=optuna env.difficulty=MEDIUM \
    ppo.n_steps=2048,4096 ppo.batch_size=64,128 ppo.ent_coef=interval(0.005,0.02) \
    env.cost_weight=interval(0.01,0.08) env.fidelity_high_step_cost=0.5,1.0 \
    env.fidelity_low_step_cost=0.0,0.5 hydra.sweeper.n_trials=40`

And for HARD:
- `python train.py -m hydra/sweeper=optuna env.difficulty=HARD \
    ppo.n_steps=4096,6144,8192 ppo.ent_coef=interval(0.01,0.05) ppo.gamma=interval(0.99,0.999) \
    env.cost_weight=interval(0.02,0.10) env.budget_init=10.0,20.0,30.0 \
    env.fidelity_high_step_cost=1.0,2.0 hydra.sweeper.n_trials=60`

Notes:
- optuna.yaml defines search spaces for PPO (LR, n_steps, batch_size, etc.) and MEDIUM/HARD env cost knobs.
- For fair comparisons across difficulties, prefer using the same total_timesteps and similar n_envs.
