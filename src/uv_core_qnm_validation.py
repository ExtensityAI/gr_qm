
"""
uv_core_qnm_validation.py
-------------------------
Comprehensive, self-contained derivations and validations for UV-regular BH cores:

- Geometry:
    F(r) = 1 - 2 m(r)/r,  with m(r) = M r^3 / (r^3 + L^3)
- Eikonal (geodesic) validation:
    ω_{ℓn} ≈ ℓ Ω_c - i (n+1/2) λ_c with photon-sphere r_c solving 2F = r F'.
- Wave-based validation (Schutz–Will WKB1) for axial perturbations:
    V_ℓ(r) = f(r)[ℓ(ℓ+1)/r^2 - 6 M / r^3],   f(r)=1-2 m(r)/r
    d^2/dr_*^2 = f^2 d^2/dr^2 + f f' d/dr
    ω^2 ≈ V0 - i (n+1/2) sqrt(-2 V0'')
- Fits: fractional shifts vs ε = L/r_s (r_s=2M) to a1 ε + a3 ε^3
- Figures saved as PDF in ./figures

Requirements: sympy, numpy, mpmath, matplotlib
"""

import os, math, json
import numpy as np
import sympy as sp
from mpmath import findroot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Utilities ----------
def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(ROOT), "figures")
ensure_dir(FIGDIR)

# ---------- Symbols & core geometry ----------
r, M, L = sp.symbols('r M L', positive=True, real=True)

# Hayward-like mass and metric function (one-scale, exterior-clean)
m_expr = M * r**3 / (r**3 + L**3)
f_expr = sp.simplify(1 - 2*m_expr/r)

# ---------- Eikonal (geodesic) functions for general F,G ----------
def build_eikonal_functions(F_expr, G_expr):
    """Return numeric helpers for photon-sphere radius, Ω_c, λ_c for given F,G."""
    Fp = sp.diff(F_expr, r)
    FG = sp.simplify(F_expr*G_expr)
    dFG = sp.diff(FG, r)
    # photon sphere: 2F = r F'
    eq_ph = sp.simplify(2*F_expr - r*Fp)
    # dr/dr_* = sqrt(FG);  d^2X/dr_*^2 = FG X'' + (1/2) (FG)' X'
    def d2_drstar2_of(X):
        Xp  = sp.diff(X, r)
        Xpp = sp.diff(X, r, 2)
        return sp.simplify(FG*Xpp + sp.Rational(1,2)*dFG*Xp)

    X = sp.simplify(F_expr/r**2)
    d2X = d2_drstar2_of(X)

    eq_ph_num = sp.lambdify((r, M, L), eq_ph, 'mpmath')
    F_num     = sp.lambdify((r, M, L), F_expr, 'mpmath')
    d2X_num   = sp.lambdify((r, M, L), d2X, 'mpmath')

    def photon_sphere_radius(Mv, Lv, guess=None):
        if guess is None: guess = 3.0*Mv
        func = lambda rr: eq_ph_num(rr, Mv, Lv)
        for g in [guess, 2.8*Mv, 3.2*Mv, 3.5*Mv, 2.5*Mv, 4.0*Mv]:
            try:
                rc = float(findroot(func, g))
                if rc > 2*Mv:
                    return rc
            except: pass
        raise RuntimeError("Photon sphere search failed")

    def omega_lambda(Mv, Lv):
        rc = photon_sphere_radius(Mv, Lv)
        Fc = F_num(rc, Mv, Lv)
        Omega = math.sqrt(Fc)/rc
        d2Xc = d2X_num(rc, Mv, Lv)
        lam_sq = 0.5 * ( - (rc**2 / Fc) * d2Xc )
        lam = math.sqrt(lam_sq) if lam_sq > 0 else 0.0
        return rc, Omega, lam

    return photon_sphere_radius, omega_lambda

# Case A: F=G=f_expr
ph_rc_A, ph_omlam_A = build_eikonal_functions(f_expr, f_expr)

def scan_eikonal_case_A(Mval=1.0, L_grid=None):
    if L_grid is None:
        L_grid = np.linspace(0.0, 0.4, 13)  # geometric units with M=1 ⇒ r_s=2
    eps = L_grid/(2.0*Mval)
    rc, Om, Lam = [], [], []
    for Lx in L_grid:
        rci, Oi, li = ph_omlam_A(Mval, Lx)
        rc.append(rci); Om.append(Oi); Lam.append(li)
    return eps, np.array(L_grid), np.array(rc), np.array(Om), np.array(Lam)

def frac_shift(arr):
    return (arr - arr[0]) / arr[0]

def fit_lin_cubic(eps, d):
    A = np.vstack([eps, eps**3]).T
    coeffs, *_ = np.linalg.lstsq(A, d, rcond=None)
    return coeffs  # [a1, a3]

# Plot helper (single plot per figure; no colors/styles imposed)
def plot_and_save(x, y, fit_coeffs, xlabel, ylabel, title, outpath):
    a1, a3 = fit_coeffs
    plt.figure()
    plt.scatter(x, y, label="data")
    plt.plot(x, a1*x + a3*(x**3), label="fit")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, format="pdf")
    plt.close()

# ---------- WKB1 wave-based computation ----------
l_val, n_val = 2, 0
# Regge–Wheeler axial potential with effective mass deformation at the barrier
V_expr = sp.simplify( f_expr * ( l_val*(l_val+1)/r**2 - 6*sp.Symbol('M')/r**3 ).subs({'M':M}) )
f_r = sp.diff(f_expr, r)
V_r = sp.diff(V_expr, r)
V_rr = sp.diff(V_expr, r, 2)
V_num  = sp.lambdify((r, M, L), V_expr, 'mpmath')
f_num  = sp.lambdify((r, M, L), f_expr, 'mpmath')
fr_num = sp.lambdify((r, M, L), f_r, 'mpmath')
Vr_num = sp.lambdify((r, M, L), V_r, 'mpmath')
Vrr_num= sp.lambdify((r, M, L), V_rr, 'mpmath')

def V_rstar_derivs(r0, Mv, Lv):
    f0  = f_num(r0, Mv, Lv)
    V0  = V_num(r0, Mv, Lv)
    Vp  = Vr_num(r0, Mv, Lv)
    Vpp = Vrr_num(r0, Mv, Lv)
    fp  = fr_num(r0, Mv, Lv)
    # d^2V/dr_*^2 = f^2 V'' + f f' V'
    V2  = f0**2 * Vpp + f0 * fp * Vp
    return float(V0), float(V2)

def r_peak(Mv, Lv):
    g = lambda rr: Vr_num(rr, Mv, Lv)
    for g0 in [3.0*Mv, 2.8*Mv, 3.2*Mv, 2.6*Mv, 3.6*Mv]:
        try:
            root = float(findroot(g, g0))
            if root > 2*Mv: return root
        except: pass
    raise RuntimeError("Barrier peak not found")

def wkb1_frequency(Mv, Lv, n=n_val):
    r0 = r_peak(Mv, Lv)
    V0, V2 = V_rstar_derivs(r0, Mv, Lv)
    if V2 >= 0:
        for s in [0.999, 1.001, 0.997, 1.003]:
            V0, V2 = V_rstar_derivs(r0*s, Mv, Lv)
            if V2 < 0:
                r0 *= s; break
    S = np.sqrt(max(0.0, -2.0*V2))
    omega_sq = V0 - 1j*(n+0.5)*S
    omega = np.sqrt(omega_sq)
    return r0, omega

def scan_wkb1(Mv=1.0, eps_vals=None):
    if eps_vals is None:
        eps_vals = np.linspace(0.0, 0.4, 17)
    Ls = eps_vals * (2.0*Mv)
    r0s, omeg = [], []
    for Lv in Ls:
        r0, om = wkb1_frequency(Mv, Lv)
        r0s.append(r0); omeg.append(om)
    return eps_vals, np.array(Ls), np.array(r0s), np.array(omeg)

def run_all_and_save():
    # ---- Eikonal Case A ----
    epsA, LsA, rcA, OmA, LamA = scan_eikonal_case_A(Mval=1.0, L_grid=np.linspace(0.0, 0.4, 13))
    dOmA   = frac_shift(OmA)
    dLamA  = frac_shift(LamA)
    kfA, cfA = fit_lin_cubic(epsA, dOmA)
    kLa, cLa = fit_lin_cubic(epsA, dLamA)
    # Save plots
    plot_and_save(epsA, dOmA, (kfA, cfA), r"$\varepsilon=L/r_s$", r"$\delta\Omega/\Omega$",
                  "Eikonal (F=G): frequency shift", os.path.join(FIGDIR, "eikonal_caseA_deltaOmega.pdf"))
    plot_and_save(epsA, dLamA, (kLa, cLa), r"$\varepsilon=L/r_s$", r"$\delta\lambda/\lambda$",
                  "Eikonal (F=G): damping (Lyapunov) shift", os.path.join(FIGDIR, "eikonal_caseA_deltaLambda.pdf"))

    # ---- WKB1 ----
    epsW, LsW, r0s, oms = scan_wkb1(Mv=1.0, eps_vals=np.linspace(0.0, 0.4, 17))
    wR = np.real(oms); wI = np.imag(oms)
    dRe = frac_shift(wR)
    dIm = frac_shift(np.abs(wI))
    kfW, cfW = fit_lin_cubic(epsW, dRe)
    ktW, ctW = fit_lin_cubic(epsW, dIm)
    plot_and_save(epsW, dRe, (kfW, cfW), r"$\varepsilon=L/r_s$", r"$\delta(\Re\omega)/\Re\omega$",
                  "WKB1: frequency shift", os.path.join(FIGDIR, "wkb1_deltaReomega.pdf"))
    plot_and_save(epsW, dIm, (ktW, ctW), r"$\varepsilon=L/r_s$", r"$\delta(|\Im\omega|)/|\Im\omega|$",
                  "WKB1: damping shift", os.path.join(FIGDIR, "wkb1_deltaImomega.pdf"))

    # Save summary JSON with fitted coefficients for reproducibility
    out = {
        "eikonal_caseA": {"k_f": float(kfA), "c_f": float(cfA), "k_lambda": float(kLa), "c_lambda": float(cLa)},
        "wkb1_l2n0": {"k_Re": float(kfW), "c_Re": float(cfW), "k_Im": float(ktW), "c_Im": float(ctW)},
        "grid": {"eps_eikonal": epsA.tolist(), "eps_wkb1": epsW.tolist()}
    }
    with open(os.path.join(FIGDIR, "fit_coefficients.json"), "w") as fh:
        json.dump(out, fh, indent=2)

if __name__ == "__main__":
    run_all_and_save()
