# Numerical verification with SymPy/mpmath for the noncommutative (Gaussian-sourced) regular BH

import sympy as sp
import mpmath as mp
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

# Symbols / parameters
theta = 1.0  # set theta=1 for dimensionless analysis; results scale with sqrt(theta)
s = 1.5

# Lower incomplete gamma for s=3/2
def lower_gamma_32(x):
    # mpmath lower incomplete gamma gamma(s, x)
    return mp.gammainc(s, 0, x)

# Metric function pieces for theta=1 (dimensionless r; M in units of sqrt(theta))
def mass_mtheta(r, M):
    return (2.0*M/mp.sqrt(mp.pi)) * lower_gamma_32(r**2/4.0)

def f_theta(r, M):
    return 1.0 - 2.0*mass_mtheta(r, M)/r

def dgamma_dr(r):
    # d/dr lowergamma(3/2, r^2/(4 theta)) with theta=1
    return (r**2)/(4.0* (theta**1.5)) * mp.e**(-r**2/(4.0*theta))

# Horizon relation: f(r_+)=0 -> M = sqrt(pi) r_+ / (4 gamma(3/2, r_+^2/4))
def M_on_horizon(rp):
    g = lower_gamma_32(rp**2/4.0)
    return mp.sqrt(mp.pi)*rp/(4.0*g)

# Hawking temperature T_H = f'(r+)/4pi with M eliminated by horizon condition
def T_H(rp):
    g = lower_gamma_32(rp**2/4.0)
    term = 1.0 - (rp**3 * mp.e**(-rp**2/4.0)) / (4.0 * (theta**1.5) * g)
    return (1.0/(4.0*mp.pi*rp)) * term

# Extremal horizon r_ext: T_H(r_ext)=0 (equivalently bracket term = 0)
def bracket_for_TH0(rp):
    g = lower_gamma_32(rp**2/4.0)
    return 1.0 - (rp**3 * mp.e**(-rp**2/4.0)) / (4.0 * (theta**1.5) * g)

# Solve for extremal radius
# Use a reasonable initial guess; r_ext is typically O(2-4) * sqrt(theta)
root_guess = 3.0
r_ext = mp.findroot(bracket_for_TH0, root_guess)
M_ext = M_on_horizon(r_ext)

# Also find the location of the maximum Hawking temperature (where dT/dr=0)
def dT_dr_numeric(r):
    # central finite difference
    h = 1e-4
    return (T_H(r+h) - T_H(r-h)) / (2*h)

# Scan to bracket the max near where T_H is largest
rs = [1.5 + 0.05*i for i in range(120)]  # r in [1.5, 7.5]
Ts = [T_H(r) for r in rs]
imax = max(range(len(Ts)), key=lambda i: Ts[i])
r_guess_max = rs[imax]

# Refine with root of derivative near r_guess_max
r_Tmax = mp.findroot(dT_dr_numeric, r_guess_max)
Tmax = T_H(r_Tmax)
M_at_rTmax = M_on_horizon(r_Tmax)

# Small-r expansion constants: ell^2 = 3 sqrt(pi) theta^(3/2) / M
def ell2_of_M(M):
    return 3.0*mp.sqrt(mp.pi)*(theta**1.5)/M

# Mapping L <-> sqrt(theta): L = (6 sqrt(pi))^(1/3) * sqrt(theta)
L_coeff = (6.0*mp.sqrt(mp.pi))**(1.0/3.0)
L_over_sqrt_theta = L_coeff

# Package results into a dataframe
data = [
    ["Extremal horizon r_ext / sqrt(theta)", float(r_ext)],
    ["Extremal mass M_ext / sqrt(theta)", float(M_ext)],
    ["T_H maximum at r_Tmax / sqrt(theta)", float(r_Tmax)],
    ["T_H maximum value (units 1/sqrt(theta))", float(Tmax)],  # dimensionful when restoring units
    ["M on horizon at r_Tmax (M/ sqrt(theta))", float(M_at_rTmax)],
    ["ell^2 at M_ext (units theta)", float(ell2_of_M(M_ext))],
    ["Mapping coefficient L / sqrt(theta)", float(L_over_sqrt_theta)]
]
df = pd.DataFrame(data, columns=["Quantity", "Value (dimensionless with theta=1)"])

display_dataframe_to_user("NCBH numerical constants (theta=1 units)", df)

# Print a compact textual summary as well
summary = df.to_string(index=False)
summary
