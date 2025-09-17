
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from src.td_rw import evolve_td_RW, prony_single

ts, ys = evolve_td_RW()
omega = prony_single(ts, ys)
print(f"TD Schwarzschild l=2: Mω ≈ {omega.real:.6f} - {(-omega.imag):.6f} i")

os.makedirs("data", exist_ok=True)
pd.DataFrame({"t":ts, "psi":ys}).to_csv("data/schwarzschild_td_waveform.csv", index=False)
plt.figure(); plt.plot(ts, ys); plt.xlabel("t"); plt.ylabel("ψ"); plt.title("RW time-domain signal"); plt.grid(True); plt.savefig("data/fig_td_signal.pdf", dpi=150)
