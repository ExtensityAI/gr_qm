# UV‑Saturated Regular‑Core Black Holes: Ringdown, Eikonal/WKB, and Hierarchical Inference

<video width="900" controls>
  <source src="https://youtu.be/ONgwK0xHVCE" type="video/mp4">
  Your browser does not support the video tag.
</video>

Comprehensive, reproducible code to study how short‑distance, UV‑regular black‑hole cores modify quasinormal modes (QNMs) and related observables, and to infer bounds from gravitational‑wave ringdown and ancillary data. The code focuses on a one‑scale Hayward‑like mass profile and small‑core expansions, with both geodesic (eikonal) and wave‑based (WKB and time‑domain) validations, plus a lightweight hierarchical analysis for LVK O3b.

If you want a one‑shot demo, see the commands under “Run the Experiments”.

## Papers‑with‑Code Style Summary

- Task: Constrain a UV‑regular core scale `L` (or fractional size `ε = L/r_s`) using BH ringdown (220) frequencies and damping times, cross‑checked by analytic limits (eikonal) and semi‑analytic wave computations (WKB), with time‑domain validation and a hierarchical posterior across multiple GW events.
- Method: Model a one‑scale interior via `m(r) = M r^3/(r^3 + L^3)`, derive eikonal shifts `δΩ/Ω`, compute wave QNMs from WKB1/3 using barrier derivatives at the deformed peak, and validate with time‑domain Regge–Wheeler evolutions. Build per‑event residuals against GR fits (Berti–Nakano) and infer posteriors on `ε` and on absolute `L0` via a simple likelihood with optional slope systematics and start‑time mixtures.
- Key findings (as produced by this repo’s scripts): fractional QNM shifts scale close to linear+cubic in `ε` at small cores; time‑domain and WKB trends agree at small `L/rs`; hierarchical posteriors are dominated by small shifts consistent with GR within current uncertainties, yielding informative upper bounds on `ε` and `L0` at 95%.
- Code highlights: self‑contained Kerr (220) Leaver continued‑fraction solver, Schwarzschild TD evolution + Prony, symbolic eikonal/WKB scaffolds, and compact scripts to reproduce figures and CSV outputs under `data/`.

Note: For precise numerical values, run the scripts below; they generate the tables/figures in `data/` and `figures/` directly from this codebase and small CSV inputs.

## Repository Structure

Only the analysis code and supporting utilities are summarized here. Simulation, paper build files, and AI‑agent notes are intentionally omitted.

- `src/`
  - `td_rw.py`: Time‑domain Regge–Wheeler evolution for Schwarzschild axial perturbations; includes `evolve_td_RW(...)`, `prony_single(...)`, and helpers.
  - `teukolsky_cf.py`: Kerr (l=2, m=2, n=0) QNM via Leaver’s angular+radial continued fractions; robust initial guess from Berti–Cardoso–Will fits.
  - `leaver_schwarzschild.py`: Convenience Schwarzschild (220) frequency using the Kerr CF in the `a→0` limit.
  - `eikonal_level.py` and `re-eikonal_level.py`: Symbolic and numeric eikonal helpers for photon‑sphere `r_c`, frequency `Ω_c`, Lyapunov `λ_c`, and small‑`ε` fits.
  - `uv_core_qnm_validation.py`: End‑to‑end eikonal and WKB1 validation for a Hayward‑like core; produces plots and fit coefficients.
  - `wkb_pade.py`: WKB1/3 formulae and Pade(1,1) utilities; used in barrier‑based QNM estimates.
  - `hayward_core.py`: Core mass function and axial potential for Hayward‑type deformation.
  - `hierarchical.py` and `hierarchical_cov.py`: Per‑event residual construction, slope extraction from TD maps, hierarchical posteriors over `ε` and `L0`, leave‑one‑out diagnostics; `hierarchical_cov.py` adds simple start‑time mixtures and covariance options.
  - `kerr_eikonal.py`, `kerr_surrogate.py`: Eikonal reference for Kerr and small‑spin scaling surrogate for slope propagation.
  - `entropy_calculation.py`: EHT‑style shadow residuals and an entropy budget sanity check at the inferred bounds (toy calculation).
  - `NCBH_numerical_constants.py`: Numerics for a related non‑commutative BH profile and thermodynamic quantities (used in ancillary scans).
  - `pending_modify_case_c.py`: Scratchpad for alternative model cases and fits.
- `scripts/`
  - `run_hier.py`: Reproduce the hierarchical analysis end‑to‑end; writes `data/hier/` CSVs and PDFs.
  - `run_posteriors.py`: Quick posterior prototype from a linearized TD surrogate; produces `data/hayward_L_over_rs_posterior.csv` and a PDF.
  - `run_reomega.py`: TD Hayward QNMs across `L/rs` and calibrated WKB‑3 sweep; produces comparison figures and CSVs in `data/`.
  - `cf_audit.py`: Sanity checks for Schwarzschild/Kerr CF frequencies; writes `data/cf_audit/report.json`.
  - `run_diagnostics.py`: Full‑covariance hierarchical posteriors and barrier diagnostics; writes under `data/hier/` and `data/diagnostics/`.
  - `plot_barrier_mprime.py`, `run_kerr_surrogate.py`: Helper diagnostics and slope‑scaling plots.
- `tests/`
  - `test_schwarzschild_qnm.py`: Smoke tests comparing TD‑extracted (220) with a continued‑fraction solve and known “gold” values.
- `data/`
  - Small inputs and generated artifacts (CSVs, PDFs). Key inputs include `data/hier/events_o3b_tablexiii.csv` and TD/WKB sweeps generated by scripts.
- `outputs/`
  - Optional aggregated results.
- `requirements.txt`, `Makefile`
  - Minimal Python deps and convenience targets to run key experiments.

## Installation

Use Python 3.10+.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional: for notebooks, `pip install jupyter`.

## Run the Experiments

The following commands reproduce the main figures and CSVs. All outputs go under `data/` (and `figures/` where applicable).

- Core audit and quick posterior prototype:
  - `python3 scripts/cf_audit.py` → `data/cf_audit/report.json` with CF frequencies.
  - `python3 scripts/run_posteriors.py` → TD surrogate, posterior over `L/r_s`, figures in `data/`.
- Time‑domain vs WKB comparison across `L/rs`:
  - `python3 scripts/run_reomega.py` → `data/hayward_td_qnms_dense.csv`, `data/hayward_wkb3_qnms_dense.csv`, and comparison figures.
- Hierarchical ringdown inference (LVK O3b‑style table):
  - `python3 scripts/run_hier.py` → `data/hier/per_event_deltas.csv`, `data/hier/summary.json`, and posterior PDFs.
  - `python3 scripts/run_diagnostics.py` → covariance/mixture variants and barrier diagnostics in `data/hier/` and `data/diagnostics/`.
- Symbolic eikonal + WKB1 validation and fits:
  - `python3 src/uv_core_qnm_validation.py` → figures under `figures/` and `figures/fit_coefficients.json`.

Makefile shortcuts bundle several of the above:

```
make data     # runs key scripts to regenerate CSVs/plots under data/
make hier     # only hierarchical analysis
make post     # posterior prototype and diagnostics
```

Tip: Some TD and CF sweeps are modestly intensive. Start with the defaults in each script; they are sized to finish quickly on a laptop.

## How the Pieces Fit Together

- Theory side: `src/eikonal_level.py`, `src/uv_core_qnm_validation.py`, and `src/wkb_pade.py` derive/validate how small cores perturb photon‑sphere quantities and the barrier, capturing the leading `ε` scaling (often linear+cubic fits).
- Wave side: `src/td_rw.py` provides direct TD evolutions and QNM extraction; `src/teukolsky_cf.py`/`src/leaver_schwarzschild.py` provide independent frequency checks using continued fractions.
- Inference side: `src/hierarchical*.py` turn observed (f, τ) into fractional residuals relative to GR fits and combine them across events to posteriors on `ε` or `L0` with simple systematics.

## Reproducing Paper‑Style Results

The repo recreates, from scratch, the core figures/tables needed for a “ringdown bound on regular cores” analysis:

- TD and WKB‑3 trends: `data/fig_Reomega_vs_L.pdf`, `data/fig_Imomega_vs_L.pdf`, residuals in `data/fig_residuals_vs_L.pdf`.
- Hierarchical posteriors: `data/hier/posterior_eps_frac.pdf` and `data/hier/posterior_L0_abs.pdf` plus LOO diagnostics.
- Eikonal/WKB fits: `figures/eikonal_caseA_deltaOmega.pdf`, `figures/wkb1_deltaReomega.pdf`, etc., with fit coefficients in `figures/fit_coefficients.json`.

Exact numerical values may differ slightly across environments due to precision and solver settings; each script exposes tolerances and depths you can adjust.

## Testing

Tests live under `tests/`. After installing `pytest`, run:

```
pytest -q
```

Note: The CF API in `tests/test_schwarzschild_qnm.py` assumes a convenience function that may differ from the current `leaver_schwarzschild.py` surface. The TD gold‑value test is the primary smoke check.

## Data and Reproducibility

- Inputs are small and versioned under `data/` (e.g., `data/hier/events_o3b_tablexiii.csv`). Scripts generate all other CSVs/figures.
- Paths are relative; no absolute paths are required. Precision knobs (e.g., CF depth, TD window) are encoded in each script. Randomness is not used.

## Citation

If you use this repository in academic work, please cite the associated manuscript

```bibtex
@misc{Dinu_Ringdown_2025,
  author = {Dinu, Marius-Constantin},
  title = {{Ringdown Bounds on UV-Regularized Black-Hole Cores}},
  url = {https://github.com/ExtensityAI/gr_qm},
  month = {10},
  year = {2025}
}
```

and, where appropriate, the original QNM/CF/WKB references used in the code (Leaver 1985; Berti–Cardoso–Will 2009; Schutz–Will 1985; Iyer–Will 1987).
