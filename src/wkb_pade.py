
import numpy as np

def wkb1(V0, V2, n=0):
    return np.sqrt(V0) - 1j*(n+0.5)*np.sqrt(-2.0*V2)/(2.0*np.sqrt(V0))

def iyer_will_L2_L3(V2,V3,V4,V5,V6,n=0):
    alpha = n + 0.5
    R2 = V2
    A = V3/R2; B = V4/R2; C = V5/R2; D = V6/R2
    sqrt_term = np.sqrt(-2.0*R2)
    L2 = (1.0/sqrt_term)*( (1.0/8.0)*B*(alpha**2 + 0.25)
                           - (1.0/288.0)*(A**2)*(7.0 + 60.0*alpha**2) )
    L3 = (1.0/(-2.0*R2))*( (5.0/6912.0)*A**4*(77.0 + 188.0*alpha**2)
                           - (1.0/384.0)*A**2*B*(51.0 + 100.0*alpha**2)
                           + (1.0/2304.0)*B**2*(67.0 + 68.0*alpha**2)
                           + (1.0/288.0)*A*C*(19.0 + 28.0*alpha**2)
                           - (1.0/288.0)*D*(5.0 + 4.0*alpha**2) )
    return L2, L3

def wkb3(V0,V2,V3,V4,V5,V6,n=0, variant=1):
    L2, L3 = iyer_will_L2_L3(V2,V3,V4,V5,V6,n)
    alpha = n + 0.5
    if variant == 1:   alpha_eff = alpha + L2 + L3
    elif variant == 2: alpha_eff = alpha - (L2 + L3)
    elif variant == 3: alpha_eff = alpha + (L2 - L3)
    else:              alpha_eff = alpha - (L2 - L3)
    omega_sq = V0 - 1j*np.sqrt(-2.0*V2)*alpha_eff
    return np.sqrt(omega_sq), L2, L3, alpha_eff

def pade11_fit(x, y):
    A = np.vstack([np.ones_like(x), x, -x*y]).T
    a0, a1, b1 = np.linalg.lstsq(A, y, rcond=None)[0]
    return a0, a1, b1

def pade11_eval(e, a0, a1, b1):
    return (a0 + a1*e)/(1.0 + b1*e)
