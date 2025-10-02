
from __future__ import annotations
from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """
    Logs environment-specific metrics exposed in 'info' dict:
      - chi2
      - residuals.{f220,tau220,theta,bc}
      - param
    Use with SB3's logger (CSV/TensorBoard).
    """
    def __init__(self, log_param: bool = True, verbose: int = 0):
        super().__init__(verbose)
        self.log_param = log_param
        self._last_chi2 = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None or len(infos) == 0:
            return True
        # For vectorized envs, infos is a list
        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            chi2 = info.get("chi2", None)
            if chi2 is not None:
                self._last_chi2 = chi2
                self.logger.record("env/chi2", float(chi2))

            resid = info.get("residuals", None)
            if isinstance(resid, dict):
                for k, v in resid.items():
                    self.logger.record(f"env/residual_{k}", float(v))

            if self.log_param and ("param" in info):
                self.logger.record("env/param", float(info["param"]))
            if "has_horizon" in info:
                self.logger.record("env/has_horizon", 1.0 if bool(info["has_horizon"]) else 0.0)
        return True


class TrainingTBCallback(BaseCallback):
    """Additional TensorBoard logging for training introspection.

    Records optimizer learning rate, clip range, and update counters.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _safe_eval(self, val):
        try:
            # SB3 schedules accept progress_remaining in [0,1]
            prog = getattr(self.model, "_current_progress_remaining", None)
            return float(val(prog)) if callable(val) and prog is not None else float(val)
        except Exception:
            return None

    def _on_step(self) -> bool:
        # Step reward (mean across vectorized envs)
        try:
            rews = self.locals.get("rewards", None)
            if rews is not None:
                import numpy as _np
                self.logger.record("env/step_reward_mean", float(_np.mean(rews)))
        except Exception:
            pass

        # Learning rate from optimizer
        try:
            opt = getattr(self.model.policy, "optimizer", None)
            if opt and opt.param_groups:
                lr = float(opt.param_groups[0].get("lr", None))
                if lr is not None:
                    self.logger.record("train/learning_rate", lr)
        except Exception:
            pass

        # Clip range (if schedule)
        clipr = getattr(self.model, "clip_range", None)
        if clipr is not None:
            v = self._safe_eval(clipr)
            if v is not None:
                self.logger.record("train/clip_range_effective", v)

        # Update counters
        n_updates = getattr(self.model, "_n_updates", None)
        if n_updates is not None:
            self.logger.record("time/n_updates", int(n_updates))
        self.logger.record("time/num_timesteps", int(self.num_timesteps))
        return True


class DebugTBHistCallback(BaseCallback):
    """Adds TensorBoard histograms for residuals and param.

    Uses an internal buffer; writes every `histogram_every` steps.
    """
    def __init__(self, histogram_every: int = 1000, buffer_size: int = 4096, verbose: int = 0):
        super().__init__(verbose)
        self.histogram_every = int(histogram_every)
        self.buffer_size = int(buffer_size)
        self._buf: Dict[str, list[float]] = {}
        self._writer = None

    def _on_training_start(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            logdir = getattr(self.logger, "dir", None)
            if logdir:
                self._writer = SummaryWriter(log_dir=logdir)
        except Exception:
            self._writer = None

    def _push(self, key: str, value: float) -> None:
        b = self._buf.setdefault(key, [])
        b.append(float(value))
        if len(b) > self.buffer_size:
            del b[: len(b) - self.buffer_size]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None or len(infos) == 0:
            return True
        for info in infos:
            if not isinstance(info, dict):
                continue
            # residuals dict
            resid = info.get("residuals", None)
            if isinstance(resid, dict):
                for k, v in resid.items():
                    self._push(f"env/residual_{k}", float(v))
            # param
            if "param" in info:
                self._push("env/param", float(info["param"]))
            # has_horizon as 0/1
            if "has_horizon" in info:
                self._push("env/has_horizon", 1.0 if bool(info["has_horizon"]) else 0.0)

        if self._writer and self.histogram_every > 0 and (self.num_timesteps % self.histogram_every == 0):
            try:
                import numpy as _np
                for k, vals in self._buf.items():
                    if len(vals) > 1:
                        arr = _np.asarray(vals, dtype=float)
                        self._writer.add_histogram(k, arr, global_step=int(self.num_timesteps))
                self._writer.flush()
            except Exception:
                pass
        return True

    def _on_training_end(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
