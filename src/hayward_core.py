
import numpy as np

def m_hayward(r, L, M=1.0):
    return M * r**3 / (r**3 + 2.0*M*L**2)

def f_hayward(r, L, M=1.0):
    return 1.0 - 2.0*m_hayward(r, L, M)/r

def V_ax_hayward(r, L, l=2, M=1.0):
    f = f_hayward(r, L, M); m = m_hayward(r, L, M)
    return f*(l*(l+1)/r**2 - 6.0*m/r**3)
