
import numpy as np, pandas as pd, json, os
import matplotlib.pyplot as plt

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

def build_event_table(events_csv, cov_csv=None, default_rho=-0.5):
    df = pd.read_csv(events_csv)
    rows = []
    for _, r in df.iterrows():
        f0 = r.f
        tau0 = r.tau_ms*1e-3
        Mz0 = r.Mz
        chi0 = float(np.clip(r.chi, 0.0, 0.9999))
        sf = 0.5*(r.sf_up + r.sf_dn)
        st = 1e-3*0.5*(r.st_up + r.st_dn)
        sM = 0.5*(r.sM_up + r.sM_dn)
        schi = 0.5*(r.schi_up + r.schi_dn)
        fgr = f_GR_Hz(Mz0, chi0)
        tgr = tau_GR_s(Mz0, chi0)
        d_f = (f0 - fgr)/fgr
        d_t = (tau0 - tgr)/tgr
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
        rho = default_rho
        rows.append(dict(event=r.event, Mz=Mz0, chi=chi0, f_obs=f0, tau_obs=tau0, f_gr=fgr, tau_gr=tgr,
                         d_f=d_f, d_t=d_t, s_df=np.sqrt(var_df), s_dt=np.sqrt(var_dt), rho=rho))
    ev = pd.DataFrame(rows)
    cov_list = []
    for _, r in ev.iterrows():
        s1, s2, rho = r.s_df, r.s_dt, r.rho
        cov = np.array([[s1**2, rho*s1*s2],[rho*s1*s2, s2**2]], dtype=float)
        cov_list.append(cov)
    return ev, cov_list

def slopes_from_td(td_csv):
    df_td = pd.read_csv(td_csv).sort_values("L_over_rs")
    w0R = float(df_td.loc[df_td["L_over_rs"]==0.0, "Re_Mw"].iloc[0])
    w0I = float(df_td.loc[df_td["L_over_rs"]==0.0, "Im_Mw"].iloc[0])
    x = df_td["L_over_rs"].values
    wR = df_td["Re_Mw"].values
    wI = df_td["Im_Mw"].values
    delta_f = (wR - w0R)/w0R
    delta_tau = ( (w0I / wI) - 1.0 )
    mask = x <= (0.03 if np.any(x<=0.03) else np.sort(x)[min(3,len(x)-1)])
    k_f = np.polyfit(x[mask], delta_f[mask], 1)[0]
    k_t = np.polyfit(x[mask], delta_tau[mask], 1)[0]
    return k_f, k_t

def posterior_frac_full(ev, covs, kf, kt, eps_grid=None, start_mixture=None):
    if eps_grid is None:
        eps_grid = np.linspace(0.0, 0.15, 1201)
    if start_mixture is None:
        start_mixture = [(1.0, np.zeros(2))]
    data = ev[["d_f","d_t"]].values
    LL = np.zeros_like(eps_grid)
    for w, bias in start_mixture:
        shifted = data - bias[None,:]
        for i,eps in enumerate(eps_grid):
            mu = np.array([kf*eps, kt*eps])
            ll = 0.0
            for j in range(len(ev)):
                diff = shifted[j] - mu
                Cinv = np.linalg.inv(covs[j])
                ll += -0.5*(diff @ Cinv @ diff) - 0.5*np.log((2*np.pi)**2*np.linalg.det(covs[j]))
            LL[i] += w*np.exp(ll)
    P = LL.copy()
    P /= np.trapz(P, eps_grid)
    return eps_grid, P

def posterior_abs_full(ev, covs, kf, kt, L0_grid_km=None, prior_trunc_km=None):
    if L0_grid_km is None:
        L0_grid_km = np.linspace(0.0, 60.0, 1201)
    data = ev[["d_f","d_t"]].values
    rs = 2*6.67430e-11*(ev.Mz.values*Msun)/c**2  # meters
    LL = np.zeros_like(L0_grid_km)
    for i,Lkm in enumerate(L0_grid_km):
        if prior_trunc_km is not None and Lkm>prior_trunc_km:
            LL[i] = 0.0
            continue
        xi = (Lkm*1000.0)/rs
        mu = np.vstack((kf*xi, kt*xi)).T
        ll = 0.0
        for j in range(len(ev)):
            diff = data[j] - mu[j]
            Cinv = np.linalg.inv(covs[j])
            ll += -0.5*(diff @ Cinv @ diff) - 0.5*np.log((2*np.pi)**2*np.linalg.det(covs[j]))
        LL[i] = np.exp(ll)
    P = LL.copy()
    if np.trapz(P, L0_grid_km)>0:
        P /= np.trapz(P, L0_grid_km)
    return L0_grid_km, P

def hpd(grid, pdf, alpha=0.95):
    idx = np.argsort(pdf)[::-1]
    c = np.cumsum(pdf[idx]); c/=c[-1]
    thr = pdf[idx[np.searchsorted(c, alpha)]]
    mask = pdf>=thr
    return grid[mask].min(), grid[mask].max()

def median_from_pdf(grid, pdf):
    c = np.cumsum(pdf); c/=c[-1]
    return np.interp(0.5, c, grid)
