
import numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt

# Physical constants
G = 6.67430e-11
c = 299792458.0
Msun = 1.988409902147041e30
t_Msun = G*Msun/c**3  # seconds

# Kerr 220 fits (Berti/Nakano)
f1, f2, f3 = 1.5251, -1.1568, 0.1292
q1, q2, q3 = 0.7000,  1.4187, -0.4990

def f_GR_Hz(Mz_Msun, chi):
    M_geo = Mz_Msun * t_Msun  # seconds
    F = f1 + f2*(1.0-chi)**f3
    return F/(2.0*np.pi*M_geo)

def tau_GR_s(Mz_Msun, chi):
    fR = f_GR_Hz(Mz_Msun, chi)
    Q = q1 + q2*(1.0-chi)**q3
    return Q/(np.pi * fR)

def load_events(csv_path):
    df = pd.read_csv(csv_path)
    return df

def per_event_deltas(df):
    rows = []
    for _, r in df.iterrows():
        f0 = r.f
        tau0 = r.tau_ms*1e-3
        Mz0 = r.Mz
        chi0 = float(np.clip(r.chi, 0.0, 0.9999))
        # symmetric 1σ
        sf = 0.5*(r.sf_up + r.sf_dn)
        st = 1e-3*0.5*(r.st_up + r.st_dn)
        sM = 0.5*(r.sM_up + r.sM_dn)
        schi = 0.5*(r.schi_up + r.schi_dn)
        # GR predictions
        fgr = f_GR_Hz(Mz0, chi0)
        tgr = tau_GR_s(Mz0, chi0)
        # deltas
        d_f = (f0 - fgr)/fgr
        d_t = (tau0 - tgr)/tgr
        # derivatives for uncertainty propagation
        dlnf_dM = -1.0/Mz0
        F = f1 + f2*(1.0-chi0)**f3
        Fp = -f2*f3*(1.0-chi0)**(f3-1.0)
        dlnf_dchi = Fp/F
        Q = q1 + q2*(1.0-chi0)**q3
        Qp = -q2*q3*(1.0-chi0)**(q3-1.0)
        dlnQ_dchi = Qp/Q
        dln_tau_dM = -dlnf_dM
        dln_tau_dchi = dlnQ_dchi - dlnf_dchi
        var_df = (sf/fgr)**2 + (dlnf_dM*sM)**2 + (dlnf_dchi*schi)**2
        var_dt = (st/tgr)**2 + (dln_tau_dM*sM)**2 + (dln_tau_dchi*schi)**2
        rows.append(dict(event=r.event, Mz=Mz0, chi=chi0, f_obs=f0, tau_obs=tau0, f_gr=fgr, tau_gr=tgr,
                         d_f=d_f, d_t=d_t, s_df=np.sqrt(var_df), s_dt=np.sqrt(var_dt)))
    return pd.DataFrame(rows)

def slopes_from_td(td_csv):
    df_td = pd.read_csv(td_csv).sort_values("L_over_rs")
    w0R = float(df_td.loc[df_td["L_over_rs"]==0.0, "Re_Mw"].iloc[0])
    w0I = float(df_td.loc[df_td["L_over_rs"]==0.0, "Im_Mw"].iloc[0])
    x = df_td["L_over_rs"].values
    wR = df_td["Re_Mw"].values
    wI = df_td["Im_Mw"].values
    delta_f = (wR - w0R)/w0R
    delta_tau = ( (w0I / wI) - 1.0 )
    # fit near zero
    mask = x <= (0.03 if np.any(x<=0.03) else np.sort(x)[min(3,len(x)-1)])
    k_f = np.polyfit(x[mask], delta_f[mask], 1)[0]
    k_t = np.polyfit(x[mask], delta_tau[mask], 1)[0]
    return k_f, k_t

def posterior_frac(ev, kf, kt, eps_grid=None):
    if eps_grid is None:
        eps_grid = np.linspace(0.0, 0.15, 1201)
    s2f = ev.s_df.values**2; s2t = ev.s_dt.values**2
    def loglike(eps):
        mu_f = kf*eps; mu_t = kt*eps
        z = ((ev.d_f.values-mu_f)**2/s2f + np.log(2*np.pi*s2f) +
             (ev.d_t.values-mu_t)**2/s2t + np.log(2*np.pi*s2t))
        return -0.5*np.sum(z)
    LL = np.array([loglike(x) for x in eps_grid])
    P = np.exp(LL-LL.max()); P/=np.trapz(P, eps_grid)
    return eps_grid, P

def posterior_abs(ev, kf, kt, L0_grid_km=None):
    if L0_grid_km is None:
        L0_grid_km = np.linspace(0.0, 50.0, 1201)
    s2f = ev.s_df.values**2; s2t = ev.s_dt.values**2
    rs = 2*G*(ev.Mz.values*Msun)/c**2
    def loglike(Lkm):
        L0_m = Lkm*1000.0
        xi = L0_m/rs
        mu_f = kf*xi; mu_t = kt*xi
        z = ((ev.d_f.values-mu_f)**2/s2f + np.log(2*np.pi*s2f) +
             (ev.d_t.values-mu_t)**2/s2t + np.log(2*np.pi*s2t))
        return -0.5*np.sum(z)
    LL = np.array([loglike(x) for x in L0_grid_km])
    P = np.exp(LL-LL.max()); P/=np.trapz(P, L0_grid_km)
    return L0_grid_km, P

def hpd_interval(grid, pdf, alpha=0.95):
    idx = np.argsort(pdf)[::-1]
    c = np.cumsum(pdf[idx]); c/=c[-1]
    thr = pdf[idx[np.searchsorted(c, alpha)]]
    mask = pdf>=thr
    return grid[mask].min(), grid[mask].max()

def median_from_pdf(grid, pdf):
    c = np.cumsum(pdf); c/=c[-1]
    return np.interp(0.5, c, grid)

def leave_one_out(ev, kf, kt, model="frac"):
    bounds = []
    for i in range(len(ev)):
        ev2 = ev.drop(ev.index[i]).reset_index(drop=True)
        if model=="frac":
            g,p = posterior_frac(ev2, kf, kt)
        else:
            g,p = posterior_abs(ev2, kf, kt)
        lo, hi = hpd_interval(g,p,0.95); med = median_from_pdf(g,p)
        bounds.append((ev.iloc[i].event, med, lo, hi))
    return bounds

def run_hier(repo_base, outdir_rel="data/hier", td_csv_rel="data/hayward_td_qnms_dense.csv",
             events_csv_rel="data/hier/events_o3b_tablexiii.csv", slope_systematic=0.2, make_plots=True):
    outdir = os.path.join(repo_base, outdir_rel)
    os.makedirs(outdir, exist_ok=True)
    # load
    ev_tbl = load_events(os.path.join(repo_base, events_csv_rel))
    ev = per_event_deltas(ev_tbl)
    # slopes
    td_csv = os.path.join(repo_base, td_csv_rel)
    if os.path.exists(td_csv):
        kf, kt = slopes_from_td(td_csv)
    else:
        # fallback to nominal slopes if TD map not present
        kf, kt = 0.20, 0.35
    # posteriors nominal
    ge, Pe = posterior_frac(ev, kf, kt)
    gL, PL = posterior_abs(ev, kf, kt)
    # systematics
    kf_lo, kt_lo = (1.0 - slope_systematic)*kf, (1.0 - slope_systematic)*kt
    kf_hi, kt_hi = (1.0 + slope_systematic)*kf, (1.0 + slope_systematic)*kt
    _, Pe_lo = posterior_frac(ev, kf_lo, kt_lo)
    _, Pe_hi = posterior_frac(ev, kf_hi, kt_hi)
    _, PL_lo = posterior_abs(ev, kf_hi, kt_hi)
    _, PL_hi = posterior_abs(ev, kf_lo, kt_lo)
    # LOO
    loo_eps = leave_one_out(ev, kf, kt, "frac")
    loo_L0  = leave_one_out(ev, kf, kt, "abs")
    # Summaries
    eps_med = median_from_pdf(ge, Pe); eps_lo, eps_hi = hpd_interval(ge, Pe, 0.95)
    L0_med  = median_from_pdf(gL, PL); L0_lo2, L0_hi2 = hpd_interval(gL, PL, 0.95)
    summary = {
        "slopes_from_td": {"k_f": float(kf), "k_tau": float(kt)},
        "eps_posterior": {"median": float(eps_med), "ci95": [float(eps_lo), float(eps_hi)]},
        "L0_posterior_km": {"median": float(L0_med), "ci95": [float(L0_lo2), float(L0_hi2)]},
        "loo_eps": [{"event":e,"median":float(m),"lo":float(lo),"hi":float(hi)} for e,m,lo,hi in loo_eps],
        "loo_L0":  [{"event":e,"median":float(m),"lo":float(lo),"hi":float(hi)} for e,m,lo,hi in loo_L0],
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ev.to_csv(os.path.join(outdir,"per_event_deltas.csv"), index=False)
    # Plots
    if make_plots:
        plt.figure(figsize=(6.2,4.2))
        plt.plot(ge, Pe, label="nominal")
        plt.plot(ge, Pe_lo, ls='--', label="slopes -{:.0f}%".format(slope_systematic*100))
        plt.plot(ge, Pe_hi, ls='--', label="slopes +{:.0f}%".format(slope_systematic*100))
        plt.xlabel(r'$\varepsilon = L/r_s$'); plt.ylabel('Posterior density')
        plt.title('Hierarchical posterior (fractional core)')
        plt.legend(); plt.grid(True, ls=':'); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "posterior_eps_frac.pdf"), dpi=150)
        plt.close()
        plt.figure(figsize=(6.2,4.2))
        plt.plot(gL, PL, label="nominal")
        plt.plot(gL, PL_lo, ls='--', label="slopes +{:.0f}%".format(slope_systematic*100))
        plt.plot(gL, PL_hi, ls='--', label="slopes -{:.0f}%".format(slope_systematic*100))
        plt.xlabel(r'$L_0$ [km]'); plt.ylabel('Posterior density')
        plt.title('Hierarchical posterior (absolute length)')
        plt.legend(); plt.grid(True, ls=':'); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "posterior_L0_abs.pdf"), dpi=150)
        plt.close()
        # LOO plots
        def plot_loo(bounds, xlab, title, fname):
            names = [b[0] for b in bounds]
            meds  = [b[1] for b in bounds]
            los   = [b[2] for b in bounds]
            his   = [b[3] for b in bounds]
            y = np.arange(len(names))
            plt.figure(figsize=(6.6, 0.35*len(names)+1.5))
            for i in range(len(names)):
                plt.plot([los[i], his[i]], [y[i], y[i]], '-', lw=2)
                plt.plot([meds[i]], [y[i]], 'o')
            plt.yticks(y, names)
            plt.xlabel(xlab); plt.title(title)
            plt.grid(axis='x', ls=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, fname), dpi=150)
            plt.close()
        plot_loo(leave_one_out(ev, kf, kt, "frac"), r'$\varepsilon$ (95\% CI)', 'LOO: fractional core size', "loo_eps.pdf")
        plot_loo(leave_one_out(ev, kf, kt, "abs"),  r'$L_0$ [km] (95\% CI)', 'LOO: absolute core length', "loo_L0.pdf")
    return summary
