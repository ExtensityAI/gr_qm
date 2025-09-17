# %%
# Write the kerr_surrogate module before importing it (previous cell wrote it but state reset).
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, sys

base = "."
sys.path.insert(0, base)

from src.hierarchical_cov import build_event_table, slopes_from_td, posterior_frac_full, posterior_abs_full, hpd, median_from_pdf
events_csv = f"{base}/data/hier/events_o3b_tablexiii.csv"
td_csv = f"{base}/data/hayward_td_qnms_dense.csv"
ev, covs = build_event_table(events_csv, cov_csv=None, default_rho=-0.5)
kf, kt = slopes_from_td(td_csv)

from src.kerr_surrogate import scale_slopes
a_grid = np.linspace(0, 0.95, 200)
kf_a = []; kt_a=[]
for a in a_grid:
    kfi, kti = scale_slopes(kf, kt, a, p_f=1.0, p_tau=1.0)
    kf_a.append(kfi); kt_a.append(kti)
kf_a = np.array(kf_a); kt_a = np.array(kt_a)

plt.figure(figsize=(6.2,4.0))
plt.plot(a_grid, kf_a, label=r'$k_f(a)$')
plt.plot(a_grid, kt_a, label=r'$k_\tau(a)$')
plt.xlabel(r'dimensionless spin $a$'); plt.ylabel('slope (arb. units)')
plt.title('Kerr small-spin surrogate for deformation slopes')
plt.legend(); plt.grid(True, ls=':'); plt.tight_layout()
plt.savefig(f"{base}/data/diagnostics/kerr_slope_surrogate.pdf", dpi=150); plt.close()

print("Wrote kerr_surrogate and produced slope plot.")
