
import numpy as np
import mpmath as mp

def f_schwarzschild(r, M=1.0):
    return 1.0 - 2.0*M/r

def V_RW_axial(r, l=2, M=1.0):
    f = f_schwarzschild(r, M)
    return f*(l*(l+1)/r**2 - 6.0*M/r**3)

def r_from_rstar(rstar, M=1.0):
    if rstar < 0:
        r = 2.0*M*(1.0 + np.exp(rstar/(2.0*M)))
        r = max(r, 2.0001*M)
    else:
        r = rstar + 2.1*M
    for _ in range(40):
        fr = max(r/(2.0*M) - 1.0, 1e-16)
        g = r + 2.0*M*np.log(fr) - rstar
        dg = 1.0 + 2.0*M/(r - 2.0*M)
        r_new = r - g/dg
        if r_new < 2.0000001*M: r_new = 2.0000001*M
        if abs(r_new - r) < 1e-12:
            r = r_new; break
        r = r_new
    return r

def evolve_td_RW(l=2, M=1.0, rstar_obs=20.0, h=0.5, u_max=600.0, v_max=600.0, t1=80.0, t2=180.0):
    Nu = int(u_max/h)+1; Nv = int(v_max/h)+1
    Psi = np.zeros((Nu, Nv), dtype=float)
    rstar0 = 10.0; w0 = 3.0
    for j in range(Nv):
        rstar = (j*h)/2.0
        Psi[0, j] = np.exp(-((rstar - rstar0)/w0)**2)
    Psi[:,0] = 0.0
    for i in range(Nu-1):
        for j in range(Nv-1):
            u = i*h; v = j*h
            rstar_S = 0.5*(v - u)
            r = r_from_rstar(rstar_S, M)
            f = 1.0 - 2.0*M/r
            Vs = f*(l*(l+1)/r**2 - 6.0*M/r**3)
            Psi[i+1, j+1] = Psi[i+1, j] + Psi[i, j+1] - Psi[i, j] - (h*h/8.0)*Vs*(Psi[i+1, j] + Psi[i, j+1])
    ts=[]; ys=[]
    for i in range(Nu):
        u = i*h; v = u + 2.0*rstar_obs
        j = int(round(v/h))
        if 0 <= j < Nv:
            t = 0.5*(u + j*h)
            ts.append(t); ys.append(Psi[i, j])
    ts = np.array(ts); ys = np.array(ys)
    return ts, ys

def prony_single(ts, ys, t1=80.0, t2=180.0):
    mask = (ts>=t1) & (ts<=t2)
    t = ts[mask]; y = ys[mask]
    if len(y) < 3:
        raise ValueError("Insufficient points in window")
    dt = t[1]-t[0]
    Y0 = y[:-2]; Y1 = y[1:-1]; Y2 = y[2:]
    A = np.vstack([Y1, Y0]).T; b = -Y2
    a1, a0 = np.linalg.lstsq(A, b, rcond=None)[0]
    rts = np.roots([1.0, a1, a0])
    cand = None
    for rr in rts:
        mag = np.abs(rr)
        if 0.0 < mag < 1.0:
            beta = np.angle(rr)/dt
            alpha = -np.log(mag)/dt
            if beta > 0: cand = beta - 1j*alpha
    if cand is None:
        rr = rts[np.argmax(np.where(np.abs(rts)<1.0, np.abs(rts), -np.inf))]
        beta = np.angle(rr)/dt; alpha = -np.log(np.abs(rr))/dt
        cand = beta - 1j*alpha
    return cand
