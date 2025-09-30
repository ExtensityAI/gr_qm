# %%
# Re-run: Eikonal-level validation for general static, spherically symmetric metrics.
import sympy as sp
import numpy as np
from mpmath import findroot
from math import sqrt, pi

sp.init_printing()

# --------------------------
# Symbolic definitions
# --------------------------
r, M, L, eta, p = sp.symbols('r M L eta p', positive=True, real=True)

# Hayward mass and metric functions
m = M * r**3 / (r**3 + L**3)
F = 1 - 2*m/r
G = F  # one-function

# Two-function deformation
G_def = sp.simplify(F * (1 + eta*(L/r)**p))

# Generic builders
def make_eikonal_functions(F_expr, G_expr):
    Fp = sp.diff(F_expr, r)
    FG = sp.simplify(F_expr*G_expr)
    dFG = sp.diff(FG, r)

    # photon sphere condition: 2F = r F'
    eq_ph = sp.simplify(2*F_expr - r*Fp)

    # d^2/dr_*^2 for X with dr/dr_* = sqrt(FG)
    def d2_drstar2_of(X):
        Xp = sp.diff(X, r)
        Xpp = sp.diff(X, r, 2)
        return sp.simplify(FG*Xpp + sp.Rational(1,2)*dFG*Xp)

    X = sp.simplify(F_expr/r**2)
    d2X_drstar2 = d2_drstar2_of(X)

    eq_ph_num = sp.lambdify((r, M, L, eta), eq_ph, 'mpmath')
    F_num = sp.lambdify((r, M, L, eta), F_expr, 'mpmath')
    d2X_num = sp.lambdify((r, M, L, eta), d2X_drstar2, 'mpmath')

    def photon_sphere_radius(Mv, Lv, etav=0.0, guess=None):
        if guess is None:
            guess = 3.0*Mv
        func = lambda rr: eq_ph_num(rr, Mv, Lv, etav)
        for g in [guess, 2.8*Mv, 3.2*Mv, 3.5*Mv, 2.5*Mv, 4.0*Mv]:
            try:
                rc = float(findroot(func, g))
                if rc > 2*Mv:
                    return rc
            except:
                pass
        raise RuntimeError("Photon sphere search failed")

    def omega_lambda(Mv, Lv, etav=0.0):
        rc = photon_sphere_radius(Mv, Lv, etav)
        Fc = F_num(rc, Mv, Lv, etav)
        Omega = sqrt(Fc)/rc
        d2Xc = d2X_num(rc, Mv, Lv, etav)
        lam_sq = 0.5 * ( - (rc**2 / Fc) * d2Xc )
        lam = sqrt(lam_sq) if lam_sq > 0 else 0.0
        return rc, Omega, lam

    return photon_sphere_radius, omega_lambda

# Builders
ph_rc_A, ph_omlam_A = make_eikonal_functions(F, G)
ph_rc_B_template, ph_omlam_B_template = make_eikonal_functions(F, G_def)

# Scans
def scan_eps_eikonal(case='A', pval=3, etaval=1.0, Mval=1.0, L_vals=None):
    if L_vals is None:
        L_vals = np.linspace(0.0, 0.4, 9)
    eps = L_vals/(2.0*Mval)
    Om = np.zeros_like(L_vals, dtype=float)
    Lam = np.zeros_like(L_vals, dtype=float)
    rc_arr = np.zeros_like(L_vals, dtype=float)

    if case == 'A':
        for i, Lx in enumerate(L_vals):
            rc, Oc, lc = ph_omlam_A(Mval, Lx, 0.0)
            rc_arr[i], Om[i], Lam[i] = rc, Oc, lc
    elif case == 'B':
        # rebuild with fixed p
        G_def_p = sp.simplify(F * (1 + eta*(L/r)**pval))
        ph_rc_B, ph_omlam_B = make_eikonal_functions(F, G_def_p)
        for i, Lx in enumerate(L_vals):
            rc, Oc, lc = ph_omlam_B(Mval, Lx, etaval)
            rc_arr[i], Om[i], Lam[i] = rc, Oc, lc
    else:
        raise ValueError("case must be 'A' or 'B'")
    return eps, L_vals, rc_arr, Om, Lam

def fractional_shifts_from_omega(eps, Om):
    Om0 = Om[0]
    return (Om - Om0)/Om0

# Execute scans
L_grid = np.linspace(0.0, 0.4, 9)
eps_A, Ls_A, rc_A, Om_A, Lam_A = scan_eps_eikonal('A', Mval=1.0, L_vals=L_grid)
df_A = fractional_shifts_from_omega(eps_A, Om_A)

eps_B3, Ls_B3, rc_B3, Om_B3, Lam_B3 = scan_eps_eikonal('B', pval=3, etaval=1.0, Mval=1.0, L_vals=L_grid)
df_B3 = fractional_shifts_from_omega(eps_B3, Om_B3)

eps_B1, Ls_B1, rc_B1, Om_B1, Lam_B1 = scan_eps_eikonal('B', pval=1, etaval=0.05, Mval=1.0, L_vals=L_grid)
df_B1 = fractional_shifts_from_omega(eps_B1, Om_B1)

# Fit linear + cubic
import numpy as np
def fit_lin_cubic(eps, d):
    A = np.vstack([eps, eps**3]).T
    coeffs, *_ = np.linalg.lstsq(A, d, rcond=None)
    return coeffs

kf_A, cf_A = fit_lin_cubic(eps_A, df_A)
kf_B3, cf_B3 = fit_lin_cubic(eps_B3, df_B3)
kf_B1, cf_B1 = fit_lin_cubic(eps_B1, df_B1)

# LVK O3b δf_220 constraint and GW150914
deltaf90 = 0.07

def eps_max_from_cubic(deltaf, c3):
    return (abs(deltaf)/abs(c3))**(1.0/3.0)

eps_max_A = eps_max_from_cubic(deltaf90, cf_A)

# Convert ε_max to L_max for GW150914
G_SI = 6.67430e-11
c_SI = 299792458.0
M_sun = 1.98847e30
L_geom_per_Msun_m = G_SI*M_sun/c_SI**2
r_s_150914_m = 2.0 * L_geom_per_Msun_m * 62.0
L_max_150914_km = eps_max_A * r_s_150914_m / 1000.0

# MDR bound
def mdr_lambda_min(deltaf, alpha=1.0, ell=2, M_solar=62.0):
    M_geom_m = L_geom_per_Msun_m * M_solar
    r_c_m = 3.0 * M_geom_m
    Lambda_min_inv_m = ell / ( r_c_m * sqrt(2.0*deltaf/alpha) )
    return Lambda_min_inv_m

Lambda_min_inv_m_150914 = mdr_lambda_min(deltaf90, alpha=1.0, ell=2, M_solar=62.0)

# Present results
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user
import matplotlib.pyplot as plt

summary_A = pd.DataFrame({
    "Case": ["A: one-function (F=G)"],
    "k_f (linear coeff)": [kf_A],
    "c_f (cubic coeff)": [cf_A],
    "ε_max (90% from δf=0.07)": [eps_max_A],
    "L_max (GW150914, km)": [L_max_150914_km]
})

summary_B = pd.DataFrame({
    "Case": ["B: two-function, p=3 (η=1)", "B: two-function, p=1 (η=0.05)"],
    "k_f (linear coeff)": [kf_B3, kf_B1],
    "c_f (cubic coeff)": [cf_B3, cf_B1]
})

display_dataframe_to_user("Eikonal-fit coefficients and bounds", pd.concat([summary_A, summary_B], ignore_index=True))

scan_A_df = pd.DataFrame({"ε": eps_A, "δΩ/Ω": df_A, "r_c/M": rc_A/1.0})
scan_B3_df = pd.DataFrame({"ε": eps_B3, "δΩ/Ω": df_B3, "r_c/M": rc_B3/1.0})
scan_B1_df = pd.DataFrame({"ε": eps_B1, "δΩ/Ω": df_B1, "r_c/M": rc_B1/1.0})
display_dataframe_to_user("Case A: one-function (F=G) — fractional shifts", scan_A_df)
display_dataframe_to_user("Case B (p=3): two-function regular tail — fractional shifts", scan_B3_df)
display_dataframe_to_user("Case B (p=1): artificial long-range tail — fractional shifts", scan_B1_df)

plt.figure()
plt.scatter(eps_A, df_A, label="Case A data")
plt.plot(eps_A, kf_A*eps_A + cf_A*eps_A**3, label="fit a1 ε + a3 ε^3")
plt.xlabel("ε = L/r_s (M=1 ⇒ r_s=2)")
plt.ylabel("δΩ/Ω")
plt.title("Case A: one-function core (eikonal)")
plt.legend()
plt.show()

plt.figure()
plt.scatter(eps_B3, df_B3, label="Case B p=3 data")
plt.plot(eps_B3, kf_B3*eps_B3 + cf_B3*eps_B3**3, label="fit")
plt.xlabel("ε")
plt.ylabel("δΩ/Ω")
plt.title("Case B: two-function, p=3 (regular tail)")
plt.legend()
plt.show()

plt.figure()
plt.scatter(eps_B1, df_B1, label="Case B p=1 data")
plt.plot(eps_B1, kf_B1*eps_B1 + cf_B1*eps_B1**3, label="fit")
plt.xlabel("ε")
plt.ylabel("δΩ/Ω")
plt.title("Case B: two-function, p=1 (artificial long-range)")
plt.legend()
plt.show()

mdr_df = pd.DataFrame({
    "Event": ["GW150914"],
    "δf_220 90% (LVK O3b)": [deltaf90],
    "Λ_min (1/m), α=1, ℓ=2": [Lambda_min_inv_m_150914]
})
display_dataframe_to_user("MDR lower-bound (toy) from LVK δf_220", mdr_df)
