
import numpy as np

f1, f2, f3 = 1.5251, -1.1568, 0.1292
q1, q2, q3 = 0.7000,  1.4187, -0.4990

def F_of_a(a):
    return f1 + f2*(1.0-a)**f3

def Q_of_a(a):
    return q1 + q2*(1.0-a)**q3

def scale_slopes(kf0, kt0, a, p_f=1.0, p_tau=1.0):
    Fa = F_of_a(a); F0 = F_of_a(0.0)
    Qa = Q_of_a(a); Q0 = Q_of_a(0.0)
    kf = kf0 * (Fa/F0)**p_f
    kt = kt0 * (Qa/Q0)**p_tau
    return kf, kt
