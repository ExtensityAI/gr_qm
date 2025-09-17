# %%
# Recompute the hierarchical posteriors (full-cov + start-time mixture) and barrier diagnostics, then zip the repo.
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, sys
base = "."
sys.path.insert(0, base)
from src.hierarchical_cov import build_event_table, slopes_from_td, posterior_frac_full, posterior_abs_full, hpd, median_from_pdf

events_csv = f"{base}/data/hier/events_o3b_tablexiii.csv"
td_csv = f"{base}/data/hayward_td_qnms_dense.csv"
ev, covs = build_event_table(events_csv, cov_csv=None, default_rho=-0.5)
kf, kt = slopes_from_td(td_csv)

start_mix = [(0.25, np.array([0.0, 0.0])),
             (0.50, np.array([ 0.003, -0.003])),
             (0.25, np.array([-0.003,  0.003]))]

eps_grid, P_eps = posterior_frac_full(ev, covs, kf, kt, eps_grid=np.linspace(0,0.15,1201), start_mixture=start_mix)

r_s_min_km = float(np.min(2*6.67430e-11*(ev.Mz.values*1.988409902147041e30)/299792458.0**2)/1000.0)
prior_trunc_km = 0.3*r_s_min_km

L_grid, P_L = posterior_abs_full(ev, covs, kf, kt, L0_grid_km=np.linspace(0,60,1201), prior_trunc_km=prior_trunc_km)

def save_plot(x, y, xlabel, ylabel, title, path, vline=None, legend=None):
    plt.figure(figsize=(6.2,4.2))
    plt.plot(x, y, '-')
    if vline is not None:
        plt.axvline(vline, ls='--', alpha=0.6, label=legend if legend else None)
        if legend: plt.legend()
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, ls=':'); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

save_plot(eps_grid, P_eps, r'$\varepsilon=L/r_s$', 'Posterior density',
          'Hierarchical posterior (full covariance + start-time mixture)',
          f"{base}/data/hier/posterior_eps_frac_cov.pdf")

save_plot(L_grid, P_L, r'$L_0$ [km]', 'Posterior density',
          'Hierarchical posterior (absolute), truncated prior',
          f"{base}/data/hier/posterior_L0_abs_trunc.pdf", vline=prior_trunc_km, legend='prior truncation')

eps_med, (eps_lo, eps_hi) = median_from_pdf(eps_grid,P_eps), hpd(eps_grid,P_eps,0.95)
L_med, (L_lo, L_hi) = median_from_pdf(L_grid,P_L), hpd(L_grid,P_L,0.95)

summary_cov = {
    "kf": float(kf), "kt": float(kt),
    "eps_posterior": {"median": float(eps_med), "ci95":[float(eps_lo), float(eps_hi)]},
    "L0_posterior_km": {"median": float(L_med), "ci95":[float(L_lo), float(L_hi)]},
    "prior_trunc_km": float(prior_trunc_km)
}
with open(f"{base}/data/hier/summary_fullcov.json","w") as f:
    json.dump(summary_cov, f, indent=2)

# Barrier diagnostics
def m_of_r(r, M, L):
    return M * r**3 / (r**3 + L**3)
def f_of_r(r, M, L):
    return 1.0 - 2.0*m_of_r(r,M,L)/r
def V_ax(r, M, L, ell):
    f = f_of_r(r,M,L)
    return f * ( ell*(ell+1)/r**2 - 6.0*m_of_r(r,M,L)/r**3 )
def r_peak_and_height(M, L, ell):
    rmin, rmax = 2.2*M, 6.0*M
    rs = np.linspace(rmin, rmax, 2000)
    Vs = V_ax(rs, M, L, ell)
    idx = np.argmax(Vs)
    return rs[idx], Vs[idx]
def mprime_of_r(r,M,L):
    return (3*M*L**3*r**2)/(r**3+L**3)**2

M=1.0; ell=2
Ls = np.linspace(0.0, 0.14*2*M, 12)  # up to L/r_s=0.14
L_over_rs = Ls/(2*M)
rpk=[]; Vpk=[]; mpr_rel=[]
for L in Ls:
    rp, Vp = r_peak_and_height(M, L, ell)
    rpk.append(rp); Vpk.append(Vp)
    mpr_rel.append(mprime_of_r(rp, M, L)/M)
rpk = np.array(rpk); Vpk = np.array(Vpk); mpr_rel=np.array(mpr_rel)

plt.figure(figsize=(6.2,4.0))
plt.plot(L_over_rs, rpk/M, 'o-')
plt.xlabel(r'$L/r_s$'); plt.ylabel(r'$r_{\rm peak}/M$')
plt.title('Barrier peak location vs core size')
plt.grid(True, ls=':'); plt.tight_layout()
plt.savefig(f"{base}/data/diagnostics/barrier_rpeak_vs_L.pdf", dpi=150); plt.close()

plt.figure(figsize=(6.2,4.0))
plt.plot(L_over_rs, Vpk*M**2, 'o-')
plt.xlabel(r'$L/r_s$'); plt.ylabel(r'$V_{\rm peak}\,M^2$')
plt.title('Barrier peak height vs core size')
plt.grid(True, ls=':'); plt.tight_layout()
plt.savefig(f"{base}/data/diagnostics/barrier_Vpeak_vs_L.pdf", dpi=150); plt.close()
