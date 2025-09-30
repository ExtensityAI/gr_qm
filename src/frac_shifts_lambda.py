# Compute fractional shifts for λ (imaginary-part proxy) and fit linear+cubic
def frac_shift(x):
    return (x - x[0]) / x[0]

dlam_A = frac_shift(Lam_A)
dlam_B3 = frac_shift(Lam_B3)
dlam_B1 = frac_shift(Lam_B1)

ka_A, ca_A = fit_lin_cubic(eps_A, dlam_A)
kb3_A, cb3_A = fit_lin_cubic(eps_B3, dlam_B3)
kb1_A, cb1_A = fit_lin_cubic(eps_B1, dlam_B1)

import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user
lam_fit_df = pd.DataFrame({
    "Case": ["A: F=G", "B: p=3, η=1", "B: p=1, η=0.05"],
    "k_λ (linear)": [ka_A, kb3_A, kb1_A],
    "c_λ (cubic)": [ca_A, cb3_A, cb1_A]
})
display_dataframe_to_user("Eikonal λ (damping) — linear & cubic coefficients", lam_fit_df)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(eps_A, dlam_A, label="A")
plt.scatter(eps_B3, dlam_B3, label="B p=3")
plt.scatter(eps_B1, dlam_B1, label="B p=1")
plt.xlabel("ε")
plt.ylabel("δλ/λ")
plt.title("Fractional shifts of Lyapunov exponent λ")
plt.legend()
plt.show()

print("λ fits:")
print("A:", ka_A, ca_A)
print("B p=3:", kb3_A, cb3_A)
print("B p=1:", kb1_A, cb1_A)
