# %%
# Eikonal-level validation for general static, spherically symmetric metrics:
#   ds^2 = -F(r) dt^2 + dr^2/G(r) + r^2 dΩ^2
# Using Cardoso et al. (2009) / Konoplya & Stuchlík (2017):
#   ω_{l n} ≈ l Ω_c - i (n+1/2) λ_c  (eikonal / large-l),
# with photon-sphere radius r_c solving 2 F_c = r_c F'_c,
# Ω_c = sqrt(F_c)/r_c,
# λ_c = (1/√2) * sqrt[ -(r_c^2 / F_c) * d^2/dr_*^2 (F/r^2) |_{r_c} ],
# where dr/dr_* = sqrt(F G).
#
# We test:
#  (A) One-function regular core: F = G = 1 - 2 m(r)/r with Hayward mass m(r) = M r^3 / (r^3 + L^3).
#      Expect leading small-ε scaling ∝ ε^3 for δω/ω.
#  (B) Two-function interior with *regular-core tail* in G only:  G = F * (1 + η (L/r)^p).
#      For p=3 (regular tail), expect still ∝ ε^3. For artificial p=1 (unphysical tail), a linear term can appear.
#  (C) Add MDR propagation as an *additive* correction: (δω/ω)_MDR ≈ (α/2) (ℓ/(Λ r_c))^2.
#
# We then convert LVK O3b δf_220 (90% CI) into bounds on L for GW150914.
#
# NOTE: This code is self-contained and uses no external web access.
#
import sympy as sp
import numpy as np
from mpmath import findroot
from math import sqrt, pi

sp.init_printing()

# --------------------------
# Symbolic definitions
# --------------------------
r, M, L, eta, p = sp.symbols('r M L eta p', positive=True, real=True)

# Hayward mass function & one-function metric
m = M * r**3 / (r**3 + L**3)
F = 1 - 2*m/r
G = F  # one-function baseline

# Two-function G deformation: G = F*(1 + eta*(L/r)**p)
G_def = sp.simplify(F * (1 + eta*(L/r)**p))

# Useful derivatives
Fprime = sp.diff(F, r)
# General helper for eikonal quantities given generic F(r), G(r)
def make_eikonal_functions(F_expr, G_expr):
    Fp = sp.diff(F_expr, r)
    FG = sp.simplify(F_expr*G_expr)
    dFG = sp.diff(FG, r)

    # photon sphere condition: 2 F = r F'
    eq_ph = sp.simplify(2*F_expr - r*Fp)

    # Function to compute d^2/dr_*^2 of X with dr/dr_* = sqrt(F G)
    def d2_drstar2_of(X):
        Xp = sp.diff(X, r)
        Xpp = sp.diff(X, r, 2)
        return sp.simplify(FG*Xpp + sp.Rational(1,2)*dFG*Xp)

    X = sp.simplify(F_expr/r**2)
    d2X_drstar2 = d2_drstar2_of(X)

    # Lambdify numerics
    eq_ph_num = sp.lambdify((r, M, L, eta), eq_ph, 'mpmath')
    F_num = sp.lambdify((r, M, L, eta), F_expr, 'mpmath')
    G_num = sp.lambdify((r, M, L, eta), G_expr, 'mpmath')
    d2X_num = sp.lambdify((r, M, L, eta), d2X_drstar2, 'mpmath')

    def photon_sphere_radius(Mv, Lv, etav=0.0, guess=None):
        if guess is None:
            guess = 3.0*Mv  # near 3M
        func = lambda rr: eq_ph_num(rr, Mv, Lv, etav)
        # Try multiple guesses
        guesses = [guess, 2.8*Mv, 3.2*Mv, 3.5*Mv, 2.5*Mv, 4.0*Mv]
        for g in guesses:
            try:
                rc = float(findroot(func, g))
                if rc > 2*Mv:  # outside horizon
                    return rc
            except:
                pass
        raise RuntimeError("Could not find photon sphere radius")

    def omega_lambda(Mv, Lv, etav=0.0):
        rc = photon_sphere_radius(Mv, Lv, etav)
        Fc = F_num(rc, Mv, Lv, etav)
        # angular frequency
        Omega = sqrt(Fc)/rc
        # lambda via Konoplya-Stuchlik formula
        d2Xc = d2X_num(rc, Mv, Lv, etav)
        lam_sq = 0.5 * ( - (rc**2 / Fc) * d2Xc )
        # guard against small negative due to numerics
        lam = sqrt(lam_sq) if lam_sq > 0 else 0.0
        return rc, Omega, lam

    return photon_sphere_radius, omega_lambda

# Build eikonal helpers for:
# A) one-function (eta ignored)
ph_rc_A, ph_omlam_A = make_eikonal_functions(F, G)
# B) two-function with exponent p (we substitute p later)
ph_rc_B_template, ph_omlam_B_template = make_eikonal_functions(F, G_def)

# --------------------------
# Numeric studies
# --------------------------

def scan_eps_eikonal(case='A', pval=3, etaval=1.0, Mval=1.0, L_vals=None):
    if L_vals is None:
        L_vals = np.linspace(0.0, 0.4, 9)  # in geometric units with M=1 ⇒ r_s=2
    eps = L_vals/(2.0*Mval)
    Om = np.zeros_like(L_vals, dtype=float)
    Lam = np.zeros_like(L_vals, dtype=float)
    rc_arr = np.zeros_like(L_vals, dtype=float)

    if case == 'A':
        for i, Lx in enumerate(L_vals):
            rc, Oc, lc = ph_omlam_A(Mval, Lx, 0.0)
            rc_arr[i], Om[i], Lam[i] = rc, Oc, lc
    elif case == 'B':
        # We need to substitute p into the lambdified functions by rebuilding them with p value
        # Easiest approach: rebuild with p replaced numerically
        # Construct new sympy expressions with p->pval
        G_def_p = sp.simplify(F * (1 + eta*(L/r)**pval))
        ph_rc_B, ph_omlam_B = make_eikonal_functions(F, G_def_p)
        for i, Lx in enumerate(L_vals):
            rc, Oc, lc = ph_omlam_B(Mval, Lx, etaval)
            rc_arr[i], Om[i], Lam[i] = rc, Oc, lc
    else:
        raise ValueError("case must be 'A' or 'B'")
    return eps, L_vals, rc_arr, Om, Lam

# Reference (L=0) eikonal frequency proxy (for l=2, we still look at fractional scaling via Ω_c).
def fractional_shifts_from_omega(eps, Om):
    Om0 = Om[0]
    dOm = (Om - Om0)/Om0
    return dOm

# --------------------------
# Run scans: case A (one-function), case B with p=3 and p=1
# --------------------------
L_grid = np.linspace(0.0, 0.4, 9)
eps_A, Ls_A, rc_A, Om_A, Lam_A = scan_eps_eikonal('A', Mval=1.0, L_vals=L_grid)
df_A = fractional_shifts_from_omega(eps_A, Om_A)

eps_B3, Ls_B3, rc_B3, Om_B3, Lam_B3 = scan_eps_eikonal('B', pval=3, etaval=1.0, Mval=1.0, L_vals=L_grid)
df_B3 = fractional_shifts_from_omega(eps_B3, Om_B3)

eps_B1, Ls_B1, rc_B1, Om_B1, Lam_B1 = scan_eps_eikonal('B', pval=1, etaval=0.05, Mval=1.0, L_vals=L_grid)
df_B1 = fractional_shifts_from_omega(eps_B1, Om_B1)

# --------------------------
# Fit linear + cubic in ε:  d ≈ a1 ε + a3 ε^3
# --------------------------
def fit_lin_cubic(eps, d):
    A = np.vstack([eps, eps**3]).T
    coeffs, *_ = np.linalg.lstsq(A, d, rcond=None)
    return coeffs  # a1, a3

kf_A, cf_A = fit_lin_cubic(eps_A, df_A)
kf_B3, cf_B3 = fit_lin_cubic(eps_B3, df_B3)
kf_B1, cf_B1 = fit_lin_cubic(eps_B1, df_B1)

# --------------------------
# LVK O3b δf_220 constraint and GW150914 parameters (from LIGO papers)
#   δf_220 = 0.02^{+0.07}_{-0.07} (90% C.I.); use |δf|_90 ~ 0.07–0.09.
#   GW150914 source-frame: M_f ≈ 62 M_sun, a_f ≈ 0.67.
# Convert bound using cubic scaling from case A: |δf| ≈ c_f ε^3  ⇒ ε_max ≈ (|δf|/|c_f|)^{1/3}
# --------------------------
deltaf90 = 0.07  # use 0.07 (90% width), we can also show 0.09 for robustness

def eps_max_from_cubic(deltaf, c3):
    return (abs(deltaf)/abs(c3))**(1.0/3.0)

eps_max_A = eps_max_from_cubic(deltaf90, cf_A)

# Convert eps_max to L_max for GW150914
G_SI = 6.67430e-11
c_SI = 299792458.0
M_sun = 1.98847e30
L_geom_per_Msun_m = G_SI*M_sun/c_SI**2  # meters
r_s_150914_m = 2.0 * L_geom_per_Msun_m * 62.0
L_max_150914_km = eps_max_A * r_s_150914_m / 1000.0

# MDR lower bound on Λ using δf bound (toy model):  (δf/f)_MDR = (α/2) (ℓ/(Λ r_c))^2
# Take α=1, ℓ=2, r_c≈ 3M for Schwarzschild (M in geometric length). For GW150914, M_geom = G M/c^2
def mdr_lambda_min(deltaf, alpha=1.0, ell=2, M_solar=62.0):
    M_geom_m = L_geom_per_Msun_m * M_solar
    r_c_m = 3.0 * M_geom_m
    # deltaf >= (alpha/2) * (ell/(Λ r_c))^2  => Λ >= ell / (r_c * sqrt(2*deltaf/alpha))
    Lambda_min_inv_m = ell / ( r_c_m * sqrt(2.0*deltaf/alpha) )
    return Lambda_min_inv_m

Lambda_min_inv_m_150914 = mdr_lambda_min(deltaf90, alpha=1.0, ell=2, M_solar=62.0)

# --------------------------
# Present results in tables and simple plots
# --------------------------
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

# Detail scans
scan_A_df = pd.DataFrame({"ε": eps_A, "δΩ/Ω": df_A, "r_c/M": rc_A/1.0})
scan_B3_df = pd.DataFrame({"ε": eps_B3, "δΩ/Ω": df_B3, "r_c/M": rc_B3/1.0})
scan_B1_df = pd.DataFrame({"ε": eps_B1, "δΩ/Ω": df_B1, "r_c/M": rc_B1/1.0})
display_dataframe_to_user("Case A: one-function (F=G) — fractional shifts", scan_A_df)
display_dataframe_to_user("Case B (p=3): two-function regular tail — fractional shifts", scan_B3_df)
display_dataframe_to_user("Case B (p=1): artificial long-range tail — fractional shifts", scan_B1_df)

# Plots
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

# MDR bound summary
mdr_df = pd.DataFrame({
    "Event": ["GW150914"],
    "δf_220 90% (LVK O3b)": [deltaf90],
    "Λ_min (1/m), α=1, ℓ=2": [Lambda_min_inv_m_150914]
})
display_dataframe_to_user("MDR lower-bound (toy) from LVK δf_220", mdr_df)

# Done.
