
import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from src.td_rw import evolve_td_RW, prony_single, r_from_rstar
from src.hayward_core import V_ax_hayward

def evolve_td_RW_hayward(L_over_rs, l=2, M=1.0, rstar_obs=20.0, h=1.0, u_max=400.0, v_max=400.0, t1=80.0, t2=200.0):
    rs = 2.0*M; L = L_over_rs*rs
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

def build_posterior():
    os.makedirs("data", exist_ok=True)
    # TD baseline
    ts, ys = evolve_td_RW()
    om0 = prony_single(ts, ys)
    # Coarse Hayward points
    xs = [0.0, 0.02, 0.05]
    oms = [evolve_td_RW_hayward(x) for x in xs]
    df = pd.DataFrame({"L_over_rs": xs, "Re_Mw": [z.real for z in oms], "Im_Mw": [z.imag for z in oms]})
    df.to_csv("data/hayward_td_qnm_points.csv", index=False)
    # Linear surrogate
    cR = np.polyfit(df["L_over_rs"], df["Re_Mw"], 1)
    cI = np.polyfit(df["L_over_rs"], df["Im_Mw"], 1)
    def w_lin(x): return np.polyval(cR, x) + 1j*np.polyval(cI, x)
    xfine = np.linspace(0.0, 0.08, 161)
    wvals = np.array([w_lin(x) for x in xfine])
    f0 = om0.real; tau0 = -1.0/om0.imag
    dff = (wvals.real - f0)/f0
    dtt = (-1.0/wvals.imag - tau0)/tau0
    mu_df, sigma_df = 0.02, 0.07/1.645
    mu_dt, sigma_dt = 0.10, 0.40/1.645
    logL = -0.5*((dff-mu_df)/sigma_df)**2 - 0.5*((dtt-mu_dt)/sigma_dt)**2
    post = np.exp(logL - np.max(logL)); post /= np.trapz(post, xfine)
    pd.DataFrame({"L_over_rs": xfine, "posterior": post, "delta_f_over_f": dff, "delta_tau_over_tau": dtt}).to_csv("data/hayward_L_over_rs_posterior.csv", index=False)
    # Quick plots
    plt.figure(); plt.plot(xfine, post); plt.xlabel("L/r_s"); plt.ylabel("Posterior"); plt.title("Posterior for L/r_s"); plt.grid(True); plt.savefig("data/fig_posterior_L_over_rs.pdf", dpi=150)
    return True

if __name__ == "__main__":
    build_posterior()
