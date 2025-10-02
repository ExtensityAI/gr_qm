from __future__ import annotations
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import re
import json
from pathlib import Path


@dataclass
class RunConfig:
    seeds: list[int]
    total_timesteps: int
    n_envs: int
    learning_rate: float | None
    batch_size: int | None
    n_steps: int | None
    device: str | None
    extra_overrides: list[str]
    outputs_dir: str
    metrics: list[str] | None
    smooth_window: int | None
    ema_alpha: float | None
    resample: int | None
    resample_domain: str


def run_training(cfg: RunConfig) -> None:
    for seed in cfg.seeds:
        overrides = [
            f"seed={seed}",
            f"n_envs={cfg.n_envs}",
            f"train.total_timesteps={cfg.total_timesteps}",
            "train.progress_bar=false",
        ]
        if cfg.learning_rate is not None:
            overrides.append(f"ppo.learning_rate={cfg.learning_rate}")
        if cfg.batch_size is not None:
            overrides.append(f"ppo.batch_size={cfg.batch_size}")
        if cfg.n_steps is not None:
            overrides.append(f"ppo.n_steps={cfg.n_steps}")
        if cfg.device is not None:
            overrides.append(f"device={cfg.device}")
        overrides.extend(cfg.extra_overrides)

        cmd = [sys.executable, "train.py"] + overrides
        print(f"[run] seed={seed} cmd={' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def _contains_tfevents(path: str) -> bool:
    try:
        for p in Path(path).iterdir():
            if p.is_file() and p.name.startswith("events.out.tfevents"):
                return True
    except Exception:
        return False
    return False


def find_tb_runs(outputs_dir: str, prefix: str | None) -> list[str]:
    # Smart discovery: list subdirs containing tfevents; optionally filter by prefix
    base = Path(outputs_dir)
    if not base.is_dir():
        return []
    candidates = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        # run dirs have tfevents inside
        if _contains_tfevents(str(p)):
            candidates.append(p)
    # optional filter by prefix
    if prefix:
        candidates = [p for p in candidates if p.name.startswith(prefix)]
    # Prefer latest per base-name (drop numeric suffix like _1, _2)
    latest_by_base: dict[str, Path] = {}
    for p in candidates:
        m = re.match(r"^(.*?)(?:_\d+)?$", p.name)
        base_name = m.group(1) if m else p.name
        prev = latest_by_base.get(base_name)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            latest_by_base[base_name] = p
    runs = [str(p) for p in sorted(latest_by_base.values(), key=lambda q: q.name)]
    return runs


def load_scalar_series(tb_dir: str, tag: str) -> pd.DataFrame:
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return pd.DataFrame(columns=["step", "value"])  # empty
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [float(e.value) for e in events]
    return pd.DataFrame({"step": steps, "value": values}).sort_values("step").drop_duplicates(subset=["step"]) 


def smooth_df(df: pd.DataFrame, window: int | None = None, ema_alpha: float | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    s = df["value"].astype(float)
    if window and window > 1:
        sm = s.rolling(window=window, min_periods=1, center=False).mean()
    elif ema_alpha and 0.0 < ema_alpha < 1.0:
        sm = s.ewm(alpha=ema_alpha, adjust=False).mean()
    else:
        return df
    out = df.copy()
    out["value"] = sm.values
    return out


def resample_to_grid(df: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    if df.empty or grid.size == 0:
        return pd.DataFrame({"step": grid, "value": np.full_like(grid, np.nan, dtype=float)})
    x = df["step"].to_numpy(dtype=float)
    y = df["value"].to_numpy(dtype=float)
    if x.size < 2:
        # replicate last value across grid
        return pd.DataFrame({"step": grid, "value": np.full(grid.shape, y[-1] if y.size else np.nan)})
    # Interpolate within bounds, NaN outside
    y_interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return pd.DataFrame({"step": grid, "value": y_interp})


def aggregate_metrics(
    runs: list[str],
    metrics: list[str],
    smooth_window: int | None = None,
    ema_alpha: float | None = None,
    resample: int | None = None,
    resample_domain: str = "overlap",
) -> dict[str, pd.DataFrame]:
    agg: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        dfs = []
        for run in runs:
            df = load_scalar_series(run, metric)
            if not df.empty:
                df = smooth_df(df.copy(), window=smooth_window, ema_alpha=ema_alpha)
                df["run"] = os.path.basename(run)
                dfs.append(df)
        if not dfs:
            agg[metric] = pd.DataFrame()
            continue
        if resample and resample > 1:
            # Determine common grid
            mins = [d["step"].min() for d in dfs]
            maxs = [d["step"].max() for d in dfs]
            if resample_domain == "overlap":
                lo = float(max(mins))
                hi = float(min(maxs))
            else:  # union
                lo = float(min(mins))
                hi = float(max(maxs))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                agg[metric] = pd.DataFrame()
                continue
            grid = np.linspace(lo, hi, int(resample))
            aligned = []
            for d in dfs:
                rd = resample_to_grid(d[["step", "value"]], grid)
                aligned.append(rd["value"].to_numpy())
            mat = np.vstack(aligned)
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            n = np.sum(~np.isnan(mat), axis=0)
            ci95 = 1.96 * std / np.sqrt(np.clip(n, 1, None))
            grouped = pd.DataFrame({"step": grid, "mean": mean, "std": std, "n": n, "ci95": ci95})
            agg[metric] = grouped
        else:
            big = pd.concat(dfs, ignore_index=True)
            grouped = big.groupby("step").agg(
                mean=("value", "mean"),
                std=("value", "std"),
                n=("value", "count"),
            ).reset_index()
            grouped["ci95"] = 1.96 * grouped["std"] / grouped["n"].clip(lower=1).pow(0.5)
            agg[metric] = grouped
    return agg


def plot_with_ci(df: pd.DataFrame, x: str, y: str, ci: str, title: str, out_png: str) -> None:
    if df.empty:
        print(f"[warn] No data for plot: {title}")
        return
    plt.figure(figsize=(7, 4))
    plt.plot(df[x], df[y], label="mean")
    y_lo = df[y] - df[ci]
    y_hi = df[y] + df[ci]
    plt.fill_between(df[x], y_lo, y_hi, alpha=0.2, label="95% CI")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[plot] wrote {out_png}")


def discover_metrics(runs: list[str]) -> list[str]:
    # Prioritized default tags if present
    priority = [
        "time/num_timesteps",
        "env/chi2",
        "env/param",
        "env/step_reward_mean",
        "rollout/ep_rew_mean",
        "train/learning_rate",
        "train/clip_range_effective",
        "eval/mean_reward",
        "eval/mean_ep_length",
    ]
    available: set[str] = set()
    for run in runs:
        try:
            ea = EventAccumulator(run)
            ea.Reload()
            available.update(ea.Tags().get('scalars', []))
        except Exception:
            continue
    # keep those in priority that are available; include time/num_timesteps if present
    selected = [m for m in priority if m in available]
    # Ensure we always have at least one metric
    if not selected:
        selected = list(sorted(available))[:5]
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multiple seeds and aggregate TensorBoard scalars with 95% CI.")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3], help="List of seeds")
    ap.add_argument("--total-timesteps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--override", type=str, nargs="*", default=[], help="Extra Hydra overrides, e.g. env.core_model=fractional")
    ap.add_argument("--outputs-dir", type=str, default="outputs", help="TensorBoard base dir (matches config.logging.tensorboard_dir)")
    ap.add_argument("--metrics", type=str, nargs="*", default=None, help="Scalar tags to aggregate; default auto-discovers useful tags")
    ap.add_argument("--prefix", type=str, default=None, help="Run dir prefix; default auto-detects all TB runs in outputs-dir")
    # smoothing
    ap.add_argument("--smooth-window", type=int, default=None, help="Moving-average window over values (per run) before aggregation")
    ap.add_argument("--ema-alpha", type=float, default=None, help="EMA alpha (0<alpha<1). Use instead of window if preferred.")
    # resampling
    ap.add_argument("--resample", type=int, default=200, help="Resample each run to N points across step domain before aggregation (default 200)")
    ap.add_argument("--resample-domain", type=str, default="overlap", choices=["overlap", "union"], help="Use shared step overlap or full union for grid")
    ap.add_argument("--no-train", action="store_true", help="Skip training, only aggregate existing runs")
    args = ap.parse_args()

    cfg = RunConfig(
        seeds=list(args.seeds),
        total_timesteps=int(args.total_timesteps),
        n_envs=int(args.n_envs),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        device=args.device,
        extra_overrides=list(args.override),
        outputs_dir=args.outputs_dir,
        metrics=list(args.metrics) if args.metrics is not None else None,
        smooth_window=args.smooth_window,
        ema_alpha=args.ema_alpha,
        resample=int(args.resample) if args.resample else None,
        resample_domain=str(args.resample_domain),
    )

    if not args.no_train:
        run_training(cfg)

    # Find runs and aggregate
    runs = find_tb_runs(cfg.outputs_dir, args.prefix)
    if not runs:
        # legacy fallback to tb/ base if outputs-dir has no runs
        legacy = find_tb_runs("tb", args.prefix)
        if legacy:
            print(f"[info] Falling back to tb/: found {len(legacy)} runs")
            runs = legacy
    if not runs:
        print(f"[error] No runs found under {cfg.outputs_dir} with prefix '{args.prefix}'")
        sys.exit(1)
    print(f"[info] Found {len(runs)} runs: {runs}")

    # Aggregate metrics by step
    metrics = cfg.metrics or discover_metrics(runs)
    print(f"[info] Aggregating metrics: {metrics}")
    agg = aggregate_metrics(
        runs,
        metrics,
        smooth_window=cfg.smooth_window,
        ema_alpha=cfg.ema_alpha,
        resample=cfg.resample,
        resample_domain=cfg.resample_domain,
    )

    # Use time/num_timesteps as x; join accordingly for plots
    # We expect other tags to be logged at the same steps; if not, each plot uses its own step index
    out_dir = os.path.join(cfg.outputs_dir, "aggregates")
    os.makedirs(out_dir, exist_ok=True)

    # Save CSVs and plots
    summary = {"runs": runs, "metrics": metrics}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for metric, df in agg.items():
        out_csv = os.path.join(out_dir, f"agg_{metric.replace('/', '_')}.csv")
        df.to_csv(out_csv, index=False)
        print(f"[csv] wrote {out_csv}")
        if metric != "time/num_timesteps":
            plot_with_ci(df, x="step", y="mean", ci="ci95", title=metric, out_png=os.path.join(out_dir, f"{metric.replace('/', '_')}.png"))


if __name__ == "__main__":
    main()
