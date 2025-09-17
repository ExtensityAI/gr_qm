# uv_saturated_bh_qnms/src/teukolsky_cf.py
# ------------------------------------------------------------
# Kerr gravitational QNM (l=2, m=2, n=0) via Leaver continued fractions
# Angular CF: Leaver (1985) Proc. R. Soc. A 402, 285 — Eqs. (19)–(21)
# Radial  CF: Leaver (1985) Proc. R. Soc. A 402, 285 — Eqs. (24)–(27)
# This implementation follows those equations literally. Units M=1.
# Spin input is the usual Kerr \hat a = a/M in [0,1); internally we use
# Leaver's a_L = \hat a/2 (so the Kerr limit is a_L -> 1/2).
#
# References:
#   • E.W. Leaver, Proc. R. Soc. A 402, 285 (1985). (angular CF, radial CF)
#   • H.-P. Nollert, Phys. Rev. D 47, 5253 (1993). (CF tail improvement – not used here)
#   • E. Berti, V. Cardoso, A.O. Starinets, CQG 26, 163001 (2009). (useful fits/overview)
#
# NOTE:
#   This version uses deep truncation (large depth) rather than a Nollert tail.
#   For the fundamental (220) it converges well; for higher overtones you should
#   add a remainder (tail) accelerator. The interfaces below are compatible with
#   your cf_audit.py script.

import mpmath as mp

# ---------- Berti–Cardoso–Will fit for 220 (for initial guess) ----------
# M*omega_R(a) = f1 + f2*(1-a)^f3
# Q(a)         = q1 + q2*(1-a)^q3
# M*omega_I(a) = -(1/2Q) * M*omega_R(a)
_f1, _f2, _f3 = 1.5251, -1.1568, 0.1292
_q1, _q2, _q3 = 0.7000,  1.4187, -0.4990

def _w_init_from_fits(ahat):
    wr = _f1 + _f2*(1.0-ahat)**_f3
    Q  = _q1 + _q2*(1.0-ahat)**_q3
    wi = -0.5*wr/Q
    return wr + 1j*wi

# ---------- Angular CF (Leaver eqs. (19)–(21)) ----------
def _ang_abg(s, ell, m, aL, w, n):
    # k1 = |m - s|/2, k2 = |m + s|/2
    k1 = 0.5*abs(m - s)
    k2 = 0.5*abs(m + s)
    # Eq. (20): α_n^θ, β_n^θ, γ_n^θ; we keep A_lm outside and subtract it later
    alpha = -2*(n+1)*(n + 2*k1 + 1)
    beta  = n*(n-1) + 2*n*(k1 + k2 + 1 - 2*aL*w) \
            - ( 2*aL*w*(2*k1 + s + 1) - (k1 + k2)*(k1 + k2 + 1) ) \
            - ( (aL*w)**2 + s*(s+1) )
    gamma = 2*aL*w*(n + k1 + k2 + s)
    return alpha, beta, gamma

def _ang_cf_residual(A_lm, s, ell, m, aL, w, depth=200):
    # F(A) = β0 - α0 γ1 / (β1 - α1 γ2 / (β2 - ...)), with βn -> βn - A_lm
    def beta0():
        a0, b0, _ = _ang_abg(s, ell, m, aL, w, 0)
        return b0 - A_lm
    def abg(n):
        a, b, g = _ang_abg(s, ell, m, aL, w, n)
        return a, b - A_lm, g
    # bottom-up evaluation
    aN, bN, gN = abg(depth)
    f = bN
    tiny = 1e-30
    for n in range(depth-1, 0, -1):
        a, b, g = abg(n)
        denom = f if abs(f) > tiny else tiny
        f = b - a*g/denom
    a0, b0, g1 = _ang_abg(s, ell, m, aL, w, 0)[0], beta0(), _ang_abg(s, ell, m, aL, w, 1)[2]
    denom = f if abs(f) > tiny else tiny
    return b0 - a0*g1/denom

def _solve_Alm(s, ell, m, aL, w, depth=200):
    # Spherical limit: A ~ ell(ell+1) - s(s+1)
    A0 = ell*(ell+1) - s*(s+1)
    F = lambda A: mp.re(_ang_cf_residual(A, s, ell, m, aL, w, depth))
    # Robust real secant for A to avoid mpmath findroot quirks in some envs
    A_prev = mp.mpf(A0)
    A_curr = mp.mpf(A0) + mp.mpf('0.5')
    f_prev = F(A_prev)
    f_curr = F(A_curr)
    tiny = mp.mpf('1e-30')
    for _ in range(60):
        denom = f_curr - f_prev
        if abs(denom) < tiny:
            A_curr += mp.mpf('1e-3')
            f_curr = F(A_curr)
            denom = f_curr - f_prev
            if abs(denom) < tiny:
                A_prev -= mp.mpf('1e-3')
                f_prev = F(A_prev)
                denom = f_curr - f_prev
        A_next = A_curr - (A_curr - A_prev) * f_curr / denom
        if abs(A_next - A_curr) < mp.mpf('1e-12') and abs(f_curr) < mp.mpf('1e-8'):
            return mp.re(A_next)
        A_prev, f_prev = A_curr, f_curr
        A_curr, f_curr = A_next, F(A_next)
    return mp.re(A_curr)

# ---------- Radial CF (Leaver eqs. (24)–(27)) ----------
def _radial_c01234(s, ell, m, aL, w, A_lm):
    # b = sqrt(1 - 4 a^2) in Leaver's normalization (a in [0,1/2])
    b = mp.sqrt(1 - 4*(aL**2))
    # Eq. (26): c0..c4 (verbatim)
    c0 = 1 - s - 1j*w/b + (2j/b)*(w/2 - aL*m)
    c1 = -4 + 2j*w*(2 + b) + (4j/b)*(w/2 - aL*m)
    c2 = s + 3 - 3j*w - (2j/b)*(w/2 - aL*m)
    c3 = (w**2)*(4 + 2*b - aL**2) - 2*aL*m*w - s - 1 + (2 + b)*1j*w - A_lm \
         + ((4*w + 2j)/b)*(w/2 - aL*m)
    c4 = s + 1 - 2*(w**2) - (2*s + 3)*1j*w - ((4*w + 2j)/b)*(w/2 - aL*m)
    return c0, c1, c2, c3, c4

def _rad_abg(s, ell, m, aL, w, A_lm, n):
    c0, c1, c2, c3, c4 = _radial_c01234(s, ell, m, aL, w, A_lm)
    alpha = n**2 + (c0 + 1)*n + c0
    beta  = -2*n**2 + (c1 + 2)*n + c3
    gamma = n**2 + (c2 - 3)*n + c4 - c2 + 2
    return alpha, beta, gamma

def _radial_cf_residual(s, ell, m, aL, w, A_lm, depth=400):
    # G(w) = β0 - α0 γ1 / (β1 - α1 γ2 / (β2 - ...))
    def beta0():
        _, b0, _ = _rad_abg(s, ell, m, aL, w, A_lm, 0)
        return b0
    def abg(n): return _rad_abg(s, ell, m, aL, w, A_lm, n)
    aN, bN, gN = abg(depth)
    f = bN
    tiny = 1e-30
    for n in range(depth-1, 0, -1):
        a, b, g = abg(n)
        denom = f if abs(f) > tiny else tiny
        f = b - a*g/denom
    a0 = _rad_abg(s, ell, m, aL, w, A_lm, 0)[0]
    g1 = _rad_abg(s, ell, m, aL, w, A_lm, 1)[2]
    denom = f if abs(f) > tiny else tiny
    return beta0() - a0*g1/denom

# ---------- Public solver ----------
def qnm_l2m2n0_kerr(ahat, max_iter=60, tol=1e-10, depth_ang=200, depth_rad=400, verbose=False):
    """
    Compute the Kerr gravitational QNM frequency for (l,m,n)=(2,2,0) via Leaver CFs.
    Input:
      ahat       : usual dimensionless spin a/M in [0,1)
      max_iter   : max outer iterations (complex secant on the radial CF residual)
      tol        : tolerance on |Δω| and |residual|
      depth_ang  : truncation depth for angular CF
      depth_rad  : truncation depth for radial CF
    Returns:
      complex M*omega
    """
    s, ell, m = -2, 2, 2
    aL = ahat/2.0  # Leaver normalization
    # Initial guess from Berti fit
    w0 = _w_init_from_fits(ahat)
    # Complex secant on G(w) with nested angular solve A(w)
    def G(wc):
        Alm = _solve_Alm(s, ell, m, aL, wc, depth=depth_ang)
        return _radial_cf_residual(s, ell, m, aL, wc, Alm, depth=depth_rad)
    # Two starting points
    w_prev = w0
    w_curr = w0*(1 + 1e-3)
    G_prev = G(w_prev)
    G_curr = G(w_curr)
    for it in range(max_iter):
        denom = (G_curr - G_prev)
        if abs(denom) < 1e-30:
            w_curr = w_curr + (1e-3+1e-3j)
            G_curr = G(w_curr)
            denom = (G_curr - G_prev)
        w_next = w_curr - (w_curr - w_prev) * G_curr / denom
        if verbose:
            print(f"[{it}] w={w_curr}, |G|={abs(G_curr)}")
        if abs(w_next - w_curr) < tol and abs(G_curr) < 1e-6:
            return w_next
        w_prev, G_prev = w_curr, G_curr
        w_curr, G_curr = w_next, G(w_next)
    # Return best iterate if not converged to tol
    return w_curr
