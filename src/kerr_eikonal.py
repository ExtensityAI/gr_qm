
import numpy as np

def kerr_eikonal_qnm(a, M=1.0, m=2, n=0):
    """
    Crude eikonal surrogate for Kerr: ω ~ m Ω_c - i (n+1/2) λ,
    where Ω_c and λ are the circular null geodesic frequency and Lyapunov exponent.
    Here we return a small-a expansion placeholder.
    """
    omega_re = 0.37367168 + 0.05*(a/M)
    omega_im = -0.08896232 - 0.002*(a/M)
    return omega_re + 1j*omega_im
