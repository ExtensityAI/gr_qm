
import mpmath as mp
from .teukolsky_cf import qnm_l2m2n0_kerr

def qnm_l2n0_schw(max_iter=80, tol=1e-12, depth_ang=220, depth_rad=480, dps=80, verbose=False):
    """
    Schwarzschild gravitational QNM (l=2, n=0) via Leaver CF.
    Implementation delegates to the Kerr Teukolsky CF in the a->0 limit, which
    is equivalent and well conditioned for the fundamental mode.

    Parameters
    ----------
    max_iter : int
        Maximum outer iterations for the complex secant root finder.
    tol : float
        Tolerance on |Δω| and |residual| for convergence.
    depth_ang : int
        Truncation depth for the angular continued fraction (at a=0 this reduces to A=4,
        but we keep the angular solve to maintain identical numerics/interfaces).
    depth_rad : int
        Truncation depth for the radial continued fraction.
    dps : int
        mpmath working precision (decimal digits).
    verbose : bool
        Print iteration diagnostics.

    Returns
    -------
    complex
        Dimensionless frequency M*ω for the (l=2,n=0) fundamental Schwarzschild QNM.
    """
    mp.mp.dps = dps
    return qnm_l2m2n0_kerr(0.0, max_iter=max_iter, tol=tol,
                            depth_ang=depth_ang, depth_rad=depth_rad, verbose=verbose)
