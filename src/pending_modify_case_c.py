# %%
# Continue with Case C: modify F instead of G:
#   F_new = F * (1 + beta * (L/r)^p),   G_new = F_new  (keep one-function link)
#
import sympy as sp
import numpy as np
from mpmath import findroot
from math import sqrt

# Recreate the core definitions to ensure state is clean
r, M, L, eta, p, beta = sp.symbols('r M L eta p beta', positive=True, real=True)
m = M * r**3 / (r**3 + L**3)
F_base = 1 - 2*m/r

def make_eikonal_functions(F_expr, G_expr):
    Fp = sp.diff(F_expr, r)
    FG = sp.simplify(F_expr*G_expr)
    dFG = sp.diff(FG, r)
    eq_ph = sp.simplify(2*F_expr - r*Fp)
    def d2_drstar2_of(X):
        Xp = sp.diff(X, r); Xpp = sp.diff(X, r, 2)
        return sp.simplify(FG*Xpp + sp.Rational(1,2)*dFG*Xp)
    X = sp.simplify(F_expr/r**2)
    d2X_drstar2 = d2_drstar2_of(X)
    eq_ph_num = sp.lambdify((r, M, L, beta), eq_ph, 'mpmath')
    F_num = sp.lambdify((r, M, L, beta), F_expr, 'mpmath')
    d2X_num = sp.lambdify((r, M, L, beta), d2X_drstar2, 'mpmath')
    def photon_sphere_radius(Mv, Lv, betav=0.0, guess=None):
        if guess is None: guess = 3.0*Mv
        func = lambda rr: eq_ph_num(rr, Mv, Lv, betav)
        for g in [guess, 2.8*Mv, 3.2*Mv, 3.5*Mv, 2.5*Mv, 4.0*Mv]:
            try:
                rc = float(findroot(func, g))
                if rc > 2*Mv: return rc
            except: pass
        raise RuntimeError("Photon sphere search failed (Case C)")
    def omega_lambda(Mv, Lv, betav=0.0):
        rc = photon_sphere_radius(Mv, Lv, betav)
        Fc = F_num(rc, Mv, Lv, betav)
        Omega = sqrt(Fc)/rc
        d2Xc = d2X_num(rc, Mv, Lv, betav)
        lam_sq = 0.5 * ( - (rc**2 / Fc) * d2Xc )
        lam = sqrt(lam_sq) if lam_sq > 0 else 0.0
        return rc, Omega, lam
    return photon_sphere_radius, omega_lambda

def scan_case_C(pval=3, betaval=1.0, Mval=1.0, L_grid=None):
    if L_grid is None: L_grid = np.linspace(0.0, 0.4, 9)
    eps = L_grid/(2.0*Mval)
    # Build F_new, G_new with p fixed
    F_new = sp.simplify(F_base * (1 + beta*(L/r)**pval))
    G_new = F_new
    ph_rc_C, ph_omlam_C = make_eikonal_functions(F_new, G_new)
    rc, Om, Lam = [], [], []
    for Lx in L_grid:
        rci, Oi, li = ph_omlam_C(Mval, Lx, betaval)
        rc.append(rci); Om.append(Oi); Lam.append(li)
    return eps, np.array(L_grid), np.array(rc), np.array(Om), np.array(Lam)

def fractional(x):
    x0 = x[0]
    return (x - x0)/x0

def fit_lin_cubic(eps, d):
    A = np.vstack([eps, eps**3]).T
    coeffs, *_ = np.linalg.lstsq(A, d, rcond=None)
    return coeffs

# Run Case C for p=3 (regular) and p=1 (artificial long-range), beta=1 and beta=0.05
L_grid = np.linspace(0.0, 0.4, 9)

eps_C3_b1, LsC, rc_C3_b1, Om_C3_b1, Lam_C3_b1 = scan_case_C(pval=3, betaval=1.0, Mval=1.0, L_grid=L_grid)
eps_C1_b005, _, rc_C1_b005, Om_C1_b005, Lam_C1_b005 = scan_case_C(pval=1, betaval=0.05, Mval=1.0, L_grid=L_grid)

dOm_C3_b1 = fractional(Om_C3_b1)
dOm_C1_b005 = fractional(Om_C1_b005)

kf_C3_b1, cf_C3_b1 = fit_lin_cubic(eps_C3_b1, dOm_C3_b1)
kf_C1_b005, cf_C1_b005 = fit_lin_cubic(eps_C1_b005, dOm_C1_b005)

# Lyapunov/damping side
dlam_C3_b1 = fractional(Lam_C3_b1)
dlam_C1_b005 = fractional(Lam_C1_b005)
kλ_C3_b1, cλ_C3_b1 = fit_lin_cubic(eps_C3_b1, dlam_C3_b1)
kλ_C1_b005, cλ_C1_b005 = fit_lin_cubic(eps_C1_b005, dlam_C1_b005)

import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user
import matplotlib.pyplot as plt

summary_C = pd.DataFrame({
    "Case": ["C: F→F(1+β(L/r)^3), β=1", "C: F→F(1+β(L/r)^1), β=0.05"],
    "k_f (linear)": [kf_C3_b1, kf_C1_b005],
    "c_f (cubic)": [cf_C3_b1, cf_C1_b005],
    "k_λ (linear)": [kλ_C3_b1, kλ_C1_b005],
    "c_λ (cubic)": [cλ_C3_b1, cλ_C1_b005]
})
display_dataframe_to_user("Case C (modify F) — coefficients", summary_C)

# Plots for Ω shifts
plt.figure()
plt.scatter(eps_C3_b1, dOm_C3_b1, label="p=3, β=1")
plt.plot(eps_C3_b1, kf_C3_b1*eps_C3_b1 + cf_C3_b1*eps_C3_b1**3, label="fit")
plt.xlabel("ε"); plt.ylabel("δΩ/Ω"); plt.title("Case C: F modified, p=3"); plt.legend(); plt.show()

plt.figure()
plt.scatter(eps_C1_b005, dOm_C1_b005, label="p=1, β=0.05")
plt.plot(eps_C1_b005, kf_C1_b005*eps_C1_b005 + cf_C1_b005*eps_C1_b005**3, label="fit")
plt.xlabel("ε"); plt.ylabel("δΩ/Ω"); plt.title("Case C: F modified, p=1"); plt.legend(); plt.show()

# Print numeric values for quick reference
print("Case C p=3 β=1: k_f =", kf_C3_b1, " c_f =", cf_C3_b1, " | k_λ =", kλ_C3_b1, " c_λ =", cλ_C3_b1)
print("Case C p=1 β=0.05: k_f =", kf_C1_b005, " c_f =", cf_C1_b005, " | k_λ =", kλ_C1_b005, " c_λ =", cλ_C1_b005)
