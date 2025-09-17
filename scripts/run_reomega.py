# %%
# Extend the analysis with:
# 1) TD QNM frequencies for Hayward core at multiple L/rs values
# 2) WKB-3 (calibrated to TD at L=0) computed for the Hayward potential (derivatives in r* at the deformed peak)
# 3) Comparison figures: Re ω and -Im ω vs L/rs; residuals TD - WKB3
# 4) Update the repo data and paper to include the new figures
import numpy as np, mpmath as mp, matplotlib.pyplot as plt, pandas as pd, os, sys

base = "."
assert os.path.exists(base)
sys.path.insert(0, base)

# --- import helpers from repo ---
from src.td_rw import r_from_rstar, evolve_td_RW, prony_single
from src.hayward_core import V_ax_hayward

M = 1.0
l = 2
rs = 2.0*M

# ---- 1) TD QNMs across L/rs ----
def evolve_td_RW_hayward(L_over_rs, l=2, M=1.0, rstar_obs=20.0, h=1.0, u_max=450.0, v_max=450.0, t1=90.0, t2=210.0):
    """Time-domain evolution with Hayward axial potential (plugged into RW stencil)."""
    L = L_over_rs*2.0*M
    Nu = int(u_max/h)+1; Nv=int(v_max/h)+1
    Psi = np.zeros((Nu, Nv), dtype=float)
    rstar0=10.0; w0=3.5
    for j in range(Nv):
        rstar = (j*h)/2.0
        Psi[0,j] = np.exp(-((rstar-rstar0)/w0)**2)
    Psi[:,0] = 0.0
    for i in range(Nu-1):
        for j in range(Nv-1):
            u = i*h; v = j*h
            rstar_S = 0.5*(v-u)
            r = r_from_rstar(rstar_S, M)
            Vs = V_ax_hayward(r, L, l, M)
            Psi[i+1,j+1] = Psi[i+1,j] + Psi[i,j+1] - Psi[i,j] - (h*h/8.0)*Vs*(Psi[i+1,j]+Psi[i,j+1])
    ts=[]; ys=[]
    for i in range(Nu):
        u=i*h; v=u+2.0*rstar_obs; j=int(round(v/h))
        if 0<=j<Nv:
            t=0.5*(u+j*h); ts.append(t); ys.append(Psi[i,j])
    ts=np.array(ts); ys=np.array(ys)
    om = prony_single(ts, ys, t1=t1, t2=t2)
    return om

L_grid = [0.0, 0.01, 0.02, 0.03, 0.05]
rows_td = []
for x in L_grid:
    om = evolve_td_RW_hayward(x)
    rows_td.append({"L_over_rs": x, "Re_Mw": om.real, "Im_Mw": om.imag})
df_td = pd.DataFrame(rows_td)
df_td.to_csv(f"{base}/data/hayward_td_qnms_dense.csv", index=False)

# ---- 2) WKB-3 on Hayward potential ----
def f_hayward(r, L, M=1.0):
    return 1.0 - 2.0*(M * r**3 / (r**3 + 2.0*M*L**2))/r

def V_ax_hay(r, L, l=2, M=1.0):
    m = M*r**3/(r**3+2.0*M*L**2)
    f = 1.0 - 2.0*m/r
    return f*(l*(l+1)/r**2 - 6.0*m/r**3)

def Dstar_numeric(g, r, L):
    h=1e-5
    # D_* g = f d g/dr
    dgr = (g(r+h, L)-g(r-h, L))/(2*h)
    return f_hayward(r,L)*dgr

def Dstar_power_V_hay(n, r, L):
    def g(rv, LL): return V_ax_hay(rv, LL, l, M)
    for _ in range(n):
        g_prev = g
        def g_tmp(rr, LL, gprev=g_prev):
            return Dstar_numeric(gprev, rr, LL)
        g = g_tmp
    return g(r, L)

def find_r0_hay(L):
    def DstarV(r):
        h=1e-6
        dV = (V_ax_hay(r+h, L, l, M)-V_ax_hay(r-h, L, l, M))/(2*h)
        return f_hayward(r,L)*dV
    return float(mp.findroot(DstarV, 3.0))

def wkb3_from_derivs(V0,V2,V3,V4,V5,V6,n=0, variant=1):
    alpha = n + 0.5
    R2 = V2
    A = V3/R2; B = V4/R2; C = V5/R2; D = V6/R2
    sqrt_term = np.sqrt(-2.0*R2)
    L2 = (1.0/sqrt_term)*( (1.0/8.0)*B*(alpha**2 + 0.25)
                           - (1.0/288.0)*(A**2)*(7.0 + 60.0*alpha**2) )
    L3 = (1.0/(-2.0*R2))*( (5.0/6912.0)*A**4*(77.0 + 188.0*alpha**2)
                           - (1.0/384.0)*A**2*B*(51.0 + 100.0*alpha**2)
                           + (1.0/2304.0)*B**2*(67.0 + 68.0*alpha**2)
                           + (1.0/288.0)*A*C*(19.0 + 28.0*alpha**2)
                           - (1.0/288.0)*D*(5.0 + 4.0*alpha**2) )
    if variant == 1:   alpha_eff = alpha + L2 + L3
    elif variant == 2: alpha_eff = alpha - (L2 + L3)
    elif variant == 3: alpha_eff = alpha + (L2 - L3)
    else:              alpha_eff = alpha - (L2 - L3)
    omega_sq = V0 - 1j*np.sqrt(-2.0*V2)*alpha_eff
    return np.sqrt(omega_sq)

# Calibrate variant to TD at L=0
L0 = 0.0; r0 = find_r0_hay(L0)
V0 = Dstar_power_V_hay(0, r0, L0)
V2 = Dstar_power_V_hay(2, r0, L0)
V3 = Dstar_power_V_hay(3, r0, L0)
V4 = Dstar_power_V_hay(4, r0, L0)
V5 = Dstar_power_V_hay(5, r0, L0)
V6 = Dstar_power_V_hay(6, r0, L0)
om_td0 = df_td[df_td["L_over_rs"]==0.0].iloc[0]
omega_td0 = om_td0["Re_Mw"] + 1j*om_td0["Im_Mw"]
cands = []
for v in [1,2,3,4]:
    w3 = wkb3_from_derivs(V0,V2,V3,V4,V5,V6,0,variant=v)
    cands.append((v, w3, abs(w3-omega_td0)))
v_best, w3_best, err = min(cands, key=lambda t: t[-1])

# Sweep L grid and compute WKB-3 at each deformed peak
rows_wkb = []
for x in L_grid:
    L = x*rs
    r0 = find_r0_hay(L)
    V0 = Dstar_power_V_hay(0, r0, L)
    V2 = Dstar_power_V_hay(2, r0, L)
    V3 = Dstar_power_V_hay(3, r0, L)
    V4 = Dstar_power_V_hay(4, r0, L)
    V5 = Dstar_power_V_hay(5, r0, L)
    V6 = Dstar_power_V_hay(6, r0, L)
    w3 = wkb3_from_derivs(V0,V2,V3,V4,V5,V6,0,variant=v_best)
    rows_wkb.append({"L_over_rs": x, "Re_Mw": w3.real, "Im_Mw": w3.imag})
df_wkb = pd.DataFrame(rows_wkb)
df_wkb.to_csv(f"{base}/data/hayward_wkb3_qnms_dense.csv", index=False)

# ---- 3) Comparison figures ----
# Plot Re ω and -Im ω vs L/rs
plt.figure(figsize=(6.2,4.6))
plt.plot(df_td["L_over_rs"], df_td["Re_Mw"], "o-", label="TD (Hayward)")
plt.plot(df_wkb["L_over_rs"], df_wkb["Re_Mw"], "s--", label="WKB-3 (cal.)")
plt.xlabel("L / r_s"); plt.ylabel("Re Mω"); plt.grid(True); plt.legend()
plt.title("Re Mω vs L/r_s (Schwarzschild, Hayward core)")
plt.tight_layout()
plt.savefig(f"{base}/data/fig_Reomega_vs_L.pdf", dpi=150)

plt.figure(figsize=(6.2,4.6))
plt.plot(df_td["L_over_rs"], -df_td["Im_Mw"], "o-", label="TD (Hayward)")
plt.plot(df_wkb["L_over_rs"], -df_wkb["Im_Mw"], "s--", label="WKB-3 (cal.)")
plt.xlabel("L / r_s"); plt.ylabel("-Im Mω (damping)"); plt.grid(True); plt.legend()
plt.title("-Im Mω vs L/r_s (Schwarzschild, Hayward core)")
plt.tight_layout()
plt.savefig(f"{base}/data/fig_Imomega_vs_L.pdf", dpi=150)

# Residuals TD - WKB3
plt.figure(figsize=(6.2,4.6))
res_re = df_td["Re_Mw"].values - df_wkb["Re_Mw"].values
res_im = -df_td["Im_Mw"].values - (-df_wkb["Im_Mw"].values)
plt.plot(df_td["L_over_rs"], res_re, "o-", label="Δ Re Mω (TD - WKB3)")
plt.plot(df_td["L_over_rs"], res_im, "s--", label="Δ (-Im Mω) (TD - WKB3)")
plt.axhline(0, color='k', lw=0.7)
plt.xlabel("L / r_s"); plt.ylabel("Residual"); plt.grid(True); plt.legend()
plt.title("Residuals: TD minus WKB-3 (calibrated at L=0)")
plt.tight_layout()
plt.savefig(f"{base}/data/fig_residuals_vs_L.pdf", dpi=150)
print("Wrote TD and WKB-3 Hayward QNMs and comparison figures.")