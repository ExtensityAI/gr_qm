# Numerical verification: GR shadow diameters vs EHT and implied bound on a regular-core scale L
# Plus core de Sitter entropy upper limit vs BH entropy.
#
# Assumptions:
# - Shadow diameter in Schwarzschild/Kerr varies weakly with spin; we use the Schwarzschild formula
#   d_GR = 2*sqrt(27) * GM/(c^2 D)
# - For a toy regular-core profile m(r) = M r^3/(r^3 + L^3), the fractional shift in shadow diameter near
#   r ~ 3M scales as delta_d/d ≈ (8/27) * (L/r_s)^3   (first-order in small L/r_s).
#   Hence, L/r_s ≲ [(27/8)*|delta_d|/d]^{1/3}
# - We compute 2σ (≈95%) bounds using the combined (quadrature) uncertainty of measurement and mass/distance inputs.
#
# References (values plugged externally in the report with citations):
# - M87*: d_obs ≈ 42 ± 3 μas (EHT 2019, paper VI)
#   M = (6.5 ± 0.7) × 10^9 Msun, D = 16.8 (+0.8, -0.7) Mpc
# - Sgr A*: d_obs = 51.8 ± 2.3 μas (EHT 2022 paper I/IV)
#   Independent M, D from GRAVITY: M = (4.30 ± 0.06_stat ± 0.36_sys) × 10^6 Msun; R0 = 8.178 ± 0.025 kpc
#
import numpy as np
import pandas as pd

# Physical constants (SI)
G = 6.67430e-11        # m^3 kg^-1 s^-2
c = 299792458.0        # m/s
hbar = 1.054571817e-34 # J s
k_B = 1.380649e-23     # J/K
# Planck length squared l_P^2 = ħ G / c^3
lP2 = hbar * G / c**3

Msun = 1.98847e30      # kg
pc = 3.085677581491367e16  # m
kpc = 1e3*pc
Mpc = 1e6*pc
uas_to_rad = (1/3600/1e6) * (np.pi/180.0)

def d_shadow_GR(M_kg, D_m):
    """Schwarzschild shadow angular diameter in radians: d = 2*sqrt(27) * GM/(c^2 D)."""
    return 2*np.sqrt(27) * G * M_kg / (c**2 * D_m)

def r_s(M_kg):
    return 2*G*M_kg/c**2

def entropy_from_area(A_m2):
    """Bekenstein-Hawking entropy (dimensionless, in units of k_B) = A/(4 l_P^2)."""
    return A_m2/(4*lP2)

def combine_sigma(x, xsig):
    return x, xsig

# Input values with uncertainties
# M87*
M87_M = 6.5e9 * Msun
M87_M_sig = 0.7e9 * Msun
M87_D = 16.8 * Mpc
M87_D_sig = 0.75 * Mpc  # symmetrized
M87_d_obs_uas = 42.0
M87_d_obs_sig = 3.0

# Sgr A*
SgrA_M = 4.30e6 * Msun
SgrA_M_sig = np.sqrt((0.06e6*Msun)**2 + (0.36e6*Msun)**2)  # combine stat+sys
SgrA_D = 8.178 * kpc
SgrA_D_sig = 0.025 * kpc
SgrA_d_obs_uas = 51.8
SgrA_d_obs_sig = 2.3

# Compute predictions and uncertainties via linear error propagation on M and D
def compute_entry(name, M, M_sig, D, D_sig, d_obs_uas, d_obs_sig):
    d_pred_rad = d_shadow_GR(M, D)
    d_pred_uas = d_pred_rad / uas_to_rad

    # Partial derivatives
    dd_dM = (2*np.sqrt(27) * G)/(c**2 * D) / uas_to_rad
    dd_dD = - (2*np.sqrt(27) * G * M)/(c**2 * D**2) / uas_to_rad

    d_pred_sig = np.sqrt((dd_dM*M_sig)**2 + (dd_dD*D_sig)**2)

    # Residual (measured - predicted) and combined uncertainty (measurement + prediction)
    resid = d_obs_uas - d_pred_uas
    resid_sig = np.sqrt(d_obs_sig**2 + d_pred_sig**2)

    # Fractional residual and 2σ upper bound on L/rs using |δd|/d ≈ (8/27)*(L/rs)^3
    frac = resid / d_pred_uas
    frac_2sig = (abs(resid) + 2*resid_sig) / d_pred_uas
    L_over_rs_bound = ((27/8)*frac_2sig)**(1/3)

    # Entropy estimates at this bound
    rs = r_s(M)
    L_bound = L_over_rs_bound * rs
    # de Sitter core radius ell: ell^2 = L^3/(2M_geom); using geometric M_geom = GM/c^2
    M_geom = G*M/c**2  # in meters
    ell2 = L_bound**3/(2*M_geom)

    # Core de Sitter entropy S_core = A/(4 lP^2) with A = 4π ell^2
    S_core = entropy_from_area(4*np.pi*ell2)
    # BH entropy S_BH with A = 4π r_s^2
    S_BH = entropy_from_area(4*np.pi*rs**2)
    ratio = S_core / S_BH

    return {
        "Object": name,
        "M [Msun]": M/Msun,
        "D [Mpc]": D/Mpc,
        "d_obs [μas]": d_obs_uas,
        "d_GR_pred [μas]": d_pred_uas,
        "pred σ [μas]": d_pred_sig,
        "residual [μas]": resid,
        "resid σ_tot [μas]": resid_sig,
        "frac resid [%]": 100*frac,
        "2σ bound L/rs": L_over_rs_bound,
        "rs [km]": rs/1000,
        "L bound [km]": L_bound/1000,
        "S_core@bound [kB]": S_core,   # dimensionless in units of k_B
        "S_BH [kB]": S_BH,
        "S_core/S_BH": ratio
    }

rows = []
rows.append(compute_entry("M87*", M87_M, M87_M_sig, M87_D, M87_D_sig, M87_d_obs_uas, M87_d_obs_sig))
rows.append(compute_entry("Sgr A*", SgrA_M, SgrA_M_sig, SgrA_D, SgrA_D_sig, SgrA_d_obs_uas, SgrA_d_obs_sig))

df = pd.DataFrame(rows)

# Show results nicely
from caas_jupyter_tools import display_dataframe_to_user
display_dataframe_to_user("EHT shadow test and L/rs bounds", df.round({
    "M [Msun]": 3, "D [Mpc]": 3, "d_obs [μas]": 2, "d_GR_pred [μas]": 2,
    "pred σ [μas]": 2, "residual [μas]": 2, "resid σ_tot [μas]": 2,
    "frac resid [%]": 2, "2σ bound L/rs": 3, "rs [km]": 1, "L bound [km]": 1,
    "S_core@bound [kB]": 3, "S_BH [kB]": 3, "S_core/S_BH": 3
}))
