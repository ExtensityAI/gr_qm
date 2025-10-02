from __future__ import annotations
# blackhole_nebula_bhsize_animated_deterministic.py
"""
Deterministic, resumable GPU renderer for a stylized black hole with lensed stars
and logo-colored nebula. Camera/BH are fixed; background animates (drift, vortex,
nebula flow). Resumes rendering by appending frames in FRAMES_DIR. Reproducible
via SEED (procedural noise & twinkle tied to seed; torch/numpy seeded).

Main knobs (documented):
- Rendering: DURATION, FPS, RES_W, RES_H, SAMPLES_PER_PIXEL, GAMMA
- Seeding   : SEED (determinism), RUN_NAME (output separation)
- BH size   : BH_SIZE_SCALE  (apparent diameter; 1.0 = base)
- Camera    : r_cam, INCL, BASE_FOV, FIT_MARGIN
- Disk      : r_in, r_out, Q_EMISS, DOPPLER_PWR, DISK_ANGVEL
- Lensing   : MAX_STEPS, H_STEP
- Photon ring overlay: ENABLE_PHOTON_RING, PHOTON_RING_SIGMA_COEF, PHOTON_RING_GAIN
- Bloom     : BLOOM_THRESH, BLOOM_SIGMA, BLOOM_GAIN
- Stars     : SKY_DENSITY, STAR_GAIN
- Nebula    : DUST_GAIN, DUST_ALPHA, DUST_SCALE, DUST_OCTAVES, DUST_WARP,
              DUST_THRESH, DUST_SOFT, BAND_STRENGTH, BAND_ROT, BAND_SIGMA
- Artistic  : NEBULA_CLOUD_SIZE (patch size), NEBULA_GRAININESS (specks)
- Motion    : SKY_ROT_RATE, VORTEX_STRENGTH, VORTEX_SIGMA_COEF,
              NEBULA_FLOW_RATE, WARP_TIME_RATE, TWINKLE_RATE
Resuming:
- If you already rendered 6 s (144 frames at 24 fps), bump DURATION to 12.0 and
  rerun with the SAME SEED (and same hyperparameters). The script appends frames
  starting at f0144.png using absolute time t = k/FPS.
"""

import os, math, json, glob, subprocess, random
from typing import Optional, List
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io.wavfile import write
import imageio.v3 as iio
from tqdm import tqdm
import torch

# Exposed colors for nebula (overridable at runtime)
NEBULA_COLOR_CYAN    = [0.10, 0.95, 1.00]
NEBULA_COLOR_MAGENTA = [0.98, 0.20, 0.92]
NEBULA_COLOR_GOLD    = [1.00, 0.85, 0.25]

# ========= Determinism =========
SEED = 1337          # <<< set this and keep it fixed to reproduce exactly
RUN_NAME = "default" # to separate outputs for different experiments (optional)

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA as deterministic as reasonably possible
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

seed_everything(SEED)

# Small, seed-derived jitters used in our hash noise (bakes seed into patterns)
_HASH_A = 12.9898 + 0.000173 * SEED
_HASH_B = 78.233  + 0.000271 * SEED
_HASH_C = 43758.5453
_SEED_F = float(SEED)


# ========= Controls =========
DURATION = 20.0          # seconds of animation to render (used for timing + audio)
FPS      = 24            # frames per second (also drives absolute time t = k/FPS)
RES_W    = 1280          # output width  in pixels
RES_H    = 720           # output height in pixels
SAMPLES_PER_PIXEL = 1    # AA samples per pixel (1 = fastest; 2–4 = cleaner edges)
GAMMA    = 1.0           # output gamma (1.0 = linear; try 2.2 for display gamma)

OUT_ROOT      = Path(f"frames_rt_{RUN_NAME}")  # per-run folder (RUN_NAME groups outputs)
FRAMES_DIR    = OUT_ROOT / "frames"            # where f0000.png … frames go
SILENT_MP4    = OUT_ROOT / "video_silent.mp4"  # encoded video without audio
AUDIO_WAV     = OUT_ROOT / "ringdown.wav"      # generated audio (ringdown growl)
FINAL_MP4     = OUT_ROOT / "video_with_audio.mp4"  # muxed final video
PROGRESS_JSON = OUT_ROOT / "progress.json"     # progress metadata (resumable info)
META_JSON     = OUT_ROOT / "run_meta.json"     # hyperparameters + seed snapshot

# ========= Mass / scale =========
BASE_M = 1.0             # base length unit for the scene (disk radii, camera distance)
BH_SIZE_SCALE = 3.0      # >1 makes BH appear larger (scales gravitational mass only)
BH_M  = BASE_M * BH_SIZE_SCALE  # gravitating mass used for bending/horizon/photons
rs    = 2.0 * BH_M       # Schwarzschild radius (event horizon radius)
r_h   = 1.001 * rs       # small safety margin above horizon for capture test

# ========= Core deformation =========
ENABLE_CORE_DEFORMATION = True    # turn on to use your UV-regular core model

# Choose how to specify L (core size):
CORE_MODEL   = "fractional"       # "fractional" (L = eps * rs) or "absolute" (L = CORE_L0)
# If CORE_MODEL == "fractional" then CORE_EPSILON is used; otherwise CORE_L0.
CORE_EPSILON = 0.05               # eps = L/rs (use e.g. 0.05 .. 0.14 within small-def regime)
# Sensible absolute-length default:
# With BASE_M=1 and BH_SIZE_SCALE=3 ⇒ rs=6; CORE_L0=0.30 gives eps≈0.05 (small & safe).
CORE_L0      =  0.30 * BASE_M     # absolute L in scene units (same units as BASE_M)

# Cubic-response coefficients for QNM/audio coupling (Table I in the paper).
CORE_CF   = 0.248                 # fractional frequency shift coeff (δf/f = CORE_CF * (L/rs)^3).
CORE_CTAU = 0.608                 # fractional damping-time shift coeff (δτ/τ = CORE_CTAU * (L/rs)^3).

# ========= Scene =========
r_cam = 50.0 * BASE_M    # camera distance from origin along +z after tilt
INCL  = math.radians(58) # camera tilt (radians) toward the disk plane
BASE_FOV   = math.radians(45)  # desired FOV; auto-widens only if disk doesn’t fit
FIT_MARGIN = 0.12        # extra padding when auto-fitting the disk (fraction of half-FOV)

# Thin disk (kept in BASE_M so BH_SIZE_SCALE does NOT change disk size)
r_in, r_out = 6.0*BASE_M, 22.0*BASE_M  # inner/outer disk radii in the z=0 plane
Q_EMISS     = 2.5        # radial emissivity falloff ~ (rho/r_in)^(-Q_EMISS)
DOPPLER_PWR = 3.0        # exponent for relativistic beaming (larger = brighter crescent)
DISK_ANGVEL = 2*math.pi/DURATION  # disk angular speed (rad/s). Here: 1 full turn per clip.

# Marcher (geodesic-like ray bending in “optical metric”)
MAX_STEPS = 6000         # max integration steps per ray (higher = more robust, slower)
H_STEP    = 0.03         # integration step size (smaller = more accurate, slower)
EPS       = 1e-9         # numerical epsilon to avoid division by zero

# Photon ring overlay (optional cinematic boost around critical impact parameter)
ENABLE_PHOTON_RING     = True    # True to add a stylized glow near the photon ring
PHOTON_RING_SIGMA_COEF = 0.15    # thickness of that glow: sigma = coef * BH_M
PHOTON_RING_GAIN       = 0.8     # strength of the glow contribution

# --- blending refinements ---
PHOTON_RING_SOFTKNEE = 0.65      # 0..1: raise mask^softknee (smaller = softer)
PHOTON_RING_NOISE    = 0.12      # 0..~0.3: thickness jitter around circumference
PHOTON_RING_BG_MIX   = 0.65      # 0..1: how much to pull color from underlying img
PHOTON_RING_TINT     = [1.00, 0.96, 0.88]  # warmish highlight tint, subtle

# Bloom (highlights glow post-process)
BLOOM_THRESH  = 0.35     # bright-pass threshold in [0,1]
BLOOM_SIGMA   = 1.2      # Gaussian blur radius (in pixels) applied to bright-pass
BLOOM_GAIN    = 0.85     # how much of blurred highlights are added back

# Stars & nebula (company colors)
SKY_DENSITY   = 0.00035  # star probability (higher = more stars)
STAR_GAIN     = 0.55     # star brightness multiplier
DUST_GAIN     = 0.70     # base color intensity of nebula clouds
DUST_ALPHA    = 0.75     # maximum opacity of nebula clouds (0=transparent, 1=opaque)
DUST_SCALE    = 55.0     # base noise frequency for clouds (higher = finer detail)
DUST_OCTAVES  = 5        # number of fBm octaves (more = richer patterns, slower)
DUST_WARP     = 1.25     # domain-warp strength (adds swirly, cellular look)
DUST_THRESH   = 0.47     # density threshold where clouds start appearing
DUST_SOFT     = 0.98     # softness of cloud edges (higher = softer falloff)
BAND_STRENGTH = 0.9      # amplitude of the galactic “band” (Milky Way stripe)
BAND_ROT      = 0.6      # rotation/phase of the band around the BH (radians)
BAND_SIGMA    = 0.28     # width of the band (larger = thicker band)

# Artistic controls
NEBULA_CLOUD_SIZE = 15.0 # *visual* patch size: larger = bigger, slower-varying clouds
                         # (internally, effective frequency ≈ DUST_SCALE / NEBULA_CLOUD_SIZE)
NEBULA_GRAININESS = 0.35 # 0..1: density/sharpness of small bright specks inside clouds
                         # (higher = more specks and slightly tighter spots)

# Motion controls
SKY_ROT_RATE      = 0.12 # rad/s: global rotation of the sky/nebula around +z (subtle drift)
VORTEX_STRENGTH   = 0.20 # rad/s at photon ring: local swirl anchored to ring (falls off away)
VORTEX_SIGMA_COEF = 1.1  # width of that swirl in impact-parameter space (× BH_M)
NEBULA_FLOW_RATE  = 0.06 # cycles/s: UV scroll speed for clouds (gives overall flow)
WARP_TIME_RATE    = 0.25 # rate for time-varying domain-warp fields (adds evolving detail)
TWINKLE_RATE      = 0.50 # Hz: star brightness modulation (tiny twinkle so they feel alive)


# ========= Audio =========
def make_ringdown_wav(path, dur):
    fs = 44100
    t  = np.linspace(0, dur, int(fs*dur), endpoint=False)

    # Baselines
    f0   = 254.6
    tau0 = 0.00451 * 70.0

    # Fractional cubic shifts from the paper: (δf/f, δτ/τ) = (CF, CTAU) * (L/rs)^3
    eps = (core_length_L()/rs) if (ENABLE_CORE_DEFORMATION and rs > 0.0) else 0.0
    f_qnm = f0  * (1.0 + CORE_CF   * (eps**3))
    tau   = tau0 * (1.0 + CORE_CTAU * (eps**3))

    wave  = np.zeros_like(t)
    for tc in np.arange(0.0, dur, 1.25):
        dt  = np.clip(t - tc, 0, None)
        env = np.exp(-dt/tau)
        wave += env * np.cos(2*np.pi*f_qnm*dt)
    wave += 0.16*np.sin(2*np.pi*56.0*t) * np.exp(-t/3.5)
    wave  = np.tanh(1.35*wave)
    wave /= np.max(np.abs(wave)) + 1e-9
    write(path, fs, (0.92*wave*32767).astype(np.int16))
    return f_qnm, tau

# ========= Torch helpers =========
def tnorm(v, dim=-1, keepdim=False):
    return torch.sqrt(torch.clamp((v*v).sum(dim=dim, keepdim=keepdim), min=EPS))
def normalize(v, dim=-1):
    return v / (tnorm(v, dim=dim, keepdim=True) + EPS)
def rotation_x(theta, device):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[1.0,0.0,0.0],[0.0,c,-s],[0.0,s,c]], device=device, dtype=torch.float32)

# weak-field index using variable mass m(r): n(r) ≈ 1 + 2 m(r) / r
def n_of_r(r):
    r_safe = torch.clamp(r, min=1e-6)
    if ENABLE_CORE_DEFORMATION:
        return 1.0 + 2.0 * m_of_r(r_safe) / r_safe
    else:
        return 1.0 + rs / r_safe

def grad_n(x):
    """
    ∇n = (dn/dr) r̂ with n(r)=1+2m(r)/r:
    dn/dr = 2 [ m'(r) r - m(r) ] / r^2.
    """
    r = tnorm(x, dim=-1)
    r_safe = torch.clamp(r, min=1e-6)
    if ENABLE_CORE_DEFORMATION:
        mr   = m_of_r(r_safe)
        mpr  = dm_dr(r_safe)
        dn_dr = 2.0 * (mpr * r_safe - mr) / (r_safe**2 + 1e-12)
    else:
        dn_dr = -rs / (r_safe**2)
    return (dn_dr / r_safe).unsqueeze(-1) * x

# ========= Camera =========
def make_camera(device):
    y_off = -r_cam * math.tan(INCL)
    cam   = torch.tensor([0.0, y_off, r_cam], device=device, dtype=torch.float32)
    R     = rotation_x(INCL, device)
    fwd   = torch.mv(R, torch.tensor([0.0,0.0,-1.0], device=device))
    right = torch.tensor([1.0,0.0,0.0], device=device)
    up    = normalize(torch.cross(right, fwd))
    fwd   = normalize(fwd); right = normalize(right)
    return cam, right, up, fwd

def core_length_L() -> float:
    """Return L in scene units."""
    if not ENABLE_CORE_DEFORMATION:
        return 0.0
    if CORE_MODEL == "absolute":
        return float(CORE_L0)
    return float(CORE_EPSILON) * rs  # fractional model

def m_of_r(r):
    """
    Hayward-type mass profile m(r) = M r^3/(r^3 + L^3).
    r can be a torch tensor or float.
    """
    M = BH_M
    L = core_length_L()
    if L <= 0.0:
        # no deformation
        return (torch.as_tensor(M, dtype=torch.float32, device=r.device)
                if torch.is_tensor(r) else M)
    if torch.is_tensor(r):
        return M * (r**3) / (r**3 + L**3 + 1e-12)
    else:
        return M * (r**3) / (r**3 + L**3 + 1e-12)

def dm_dr(r):
    """Derivative of Hayward mass profile: 3 M L^3 r^2 / (r^3 + L^3)^2."""
    M = BH_M
    L = core_length_L()
    if L <= 0.0:
        return torch.zeros_like(r) if torch.is_tensor(r) else 0.0
    if torch.is_tensor(r):
        return (3.0 * M * (L**3) * (r**2)) / ((r**3 + L**3 + 1e-12)**2)
    else:
        return (3.0 * M * (L**3) * (r**2)) / ((r**3 + L**3 + 1e-12)**2)

def photon_ring_b_critical() -> float:
    """
    b_c ≈ 3√3 M * [1 + (8/27)(L/rs)^3] (cubic response, compact core).
    Falls back to Schwarzschild when L=0.
    """
    base = 3.0 * math.sqrt(3.0) * BH_M
    L = core_length_L()
    if L <= 0.0 or rs <= 0.0:
        return base
    eps = L / rs
    return base * (1.0 + (8.0/27.0) * (eps**3))

def _f_horizon(r: float) -> float:
    """f(r) = 1 - 2 m(r) / r using float arithmetic (for root find)."""
    M = BH_M
    L = core_length_L()
    if L <= 0.0:
        # pure Schwarzschild
        return 1.0 - 2.0*M / max(r, 1e-12)
    # m(r) = M r^3/(r^3 + L^3)  ⇒ 2 m(r)/r = 2M r^2/(r^3 + L^3)
    return 1.0 - (2.0*M * (r*r)) / (r**3 + L**3 + 1e-18)

def horizon_radius_float() -> float:
    """
    Find outer horizon r_+ solving f(r)=0. Robust bracket + bisection.
    If not found (no horizon), fall back to ~2M to preserve visuals.
    """
    M = BH_M
    # Quick exact for L=0
    if core_length_L() <= 0.0:
        return 2.0 * M

    # Scan to detect sign change
    r_min = max(1e-6, 0.02 * M)
    r_max = 6.0 * M
    Nscan = 256
    prev_r = r_min
    prev_f = _f_horizon(prev_r)
    r_lo, r_hi = None, None
    for k in range(1, Nscan + 1):
        r = r_min + (r_max - r_min) * (k / Nscan)
        f = _f_horizon(r)
        if prev_f * f <= 0.0:  # sign change (or exact zero)
            r_lo, r_hi = prev_r, r
            break
        prev_r, prev_f = r, f

    if r_lo is None:
        # No sign change detected — likely no horizon; graceful fallback
        return 2.0 * M

    # Bisection refine
    for _ in range(64):
        mid = 0.5 * (r_lo + r_hi)
        fm  = _f_horizon(mid)
        if abs(fm) < 1e-12:
            return max(mid, 1e-6)
        if _f_horizon(r_lo) * fm <= 0.0:
            r_hi = mid
        else:
            r_lo = mid
    return max(0.5 * (r_lo + r_hi), 1e-6)

def update_scales():
    """Derive BH_M, rs and refined r_h (capture) from current settings."""
    global BH_M, rs, r_h
    BH_M = BASE_M * BH_SIZE_SCALE
    rs   = 2.0 * BH_M
    # refined horizon if deformation is enabled; otherwise Schwarzschild
    if ENABLE_CORE_DEFORMATION:
        r_plus = horizon_radius_float()
        r_h    = 1.001 * r_plus
    else:
        r_h    = 1.001 * rs

# Initialize scales
update_scales()

if os.getenv("BH_DEBUG", "0") == "1":
    eps_eff = (core_length_L()/rs) if rs > 0 else 0.0
    print(
        "[init] "
        f"ENABLE_CORE_DEFORMATION={ENABLE_CORE_DEFORMATION}, CORE_MODEL={CORE_MODEL}, "
        f"L_eff={core_length_L():.6f}, eps_eff={eps_eff:.6f}, "
        f"rs={rs:.6f}, r_h={r_h:.6f}, b_c={photon_ring_b_critical():.6f}"
    )

# ========= Value noise + fBm + domain warp =========
def _hash2(i, j):
    # seed baked into the sines (deterministic across runs with same SEED)
    return torch.frac(torch.sin(i*_HASH_A + j*_HASH_B + _SEED_F*0.001) * _HASH_C)

def noise2(x, y):
    xi = torch.floor(x); yi = torch.floor(y)
    xf = x - xi;         yf = y - yi
    u  = xf*xf*(3-2*xf); v = yf*yf*(3-2*yf)
    n00 = _hash2(xi,     yi    ); n10 = _hash2(xi + 1, yi    )
    n01 = _hash2(xi,     yi + 1); n11 = _hash2(xi + 1, yi + 1)
    nx0 = n00*(1-u) + n10*u;    nx1 = n01*(1-u) + n11*u
    return nx0*(1-v) + nx1*v

def fbm2(x, y, octaves=5, lac=2.0, gain=0.5):
    n = torch.zeros_like(x); freq, amp = 1.0, 1.0
    for _ in range(octaves):
        n += amp * noise2(x*freq, y*freq)
        freq *= lac; amp *= gain
    return n

def smoothstep(edge0, edge1, x):
    t = torch.clamp((x-edge0)/(edge1-edge0+1e-6), 0.0, 1.0)
    return t*t*(3 - 2*t)

# ========= Motion mapping for sky sampling =========
def rotate_z(dir_world, angle):
    c = math.cos(angle); s = math.sin(angle)
    x = c*dir_world[...,0] - s*dir_world[...,1]
    y = s*dir_world[...,0] + c*dir_world[...,1]
    return torch.stack([x, y, dir_world[...,2]], dim=-1)

def sky_sample_dirs(dir_world, cam, time_s):
    d = rotate_z(dir_world, SKY_ROT_RATE * time_s)  # global drift
    if VORTEX_STRENGTH != 0.0:                       # ring-anchored swirl
        cam_rep = cam.view(1,1,3).expand_as(d)
        b  = tnorm(torch.cross(cam_rep, d), dim=-1)
        bc  = photon_ring_b_critical()
        sig = VORTEX_SIGMA_COEF * BH_M
        gain = torch.exp(-0.5*((b - bc)/(sig+1e-6))**2)
        ang  = VORTEX_STRENGTH * time_s * gain
        c = torch.cos(ang); s = torch.sin(ang)
        x = c*d[...,0] - s*d[...,1]
        y = s*d[...,0] + c*d[...,1]
        d = torch.stack([x, y, d[...,2]], dim=-1)
    return normalize(d)

# ========= Stars =========
def stars_from_dir(dir_world, device, time_s, density=SKY_DENSITY):
    z   = torch.clamp(dir_world[...,2], -1.0, 1.0)
    theta = torch.acos(z); phi = torch.atan2(dir_world[...,1], dir_world[...,0])
    u = (phi/(2*math.pi) + 0.5) * 1024.0
    v = (theta/math.pi) * 1024.0

    i = torch.floor(u); j = torch.floor(v)
    base = _hash2(i, j)
    star = (base > (1.0 - density*350.0)).float()

    du = u - i - 0.5; dv = v - j - 0.5
    spot = torch.exp(-(du*du + dv*dv)/(2*0.16*0.16))

    twk  = _hash2(i+7.0 + 0.01*_SEED_F, j+13.0 + 0.02*_SEED_F)
    twk2 = _hash2(i+17.0+ 0.03*_SEED_F, j+29.0 + 0.05*_SEED_F)
    twinkle = 0.9 + 0.1*torch.sin(2*math.pi*(twk2 + TWINKLE_RATE*time_s))
    brightness = (0.6 + 0.4*twk) * twinkle

    return (brightness*spot*star).unsqueeze(-1).expand_as(dir_world)

# ========= Logo-colored Nebula (patchy, alpha, grains) with FLOW =========
def nebula_rgba_from_dir(dir_world, device, time_s):
    z    = torch.clamp(dir_world[...,2], -1.0, 1.0)
    theta= torch.acos(z); phi = torch.atan2(dir_world[...,1], dir_world[...,0])
    u = (phi/(2*math.pi) + 0.5) + NEBULA_FLOW_RATE*time_s
    v = (theta/math.pi)           + 0.5*NEBULA_FLOW_RATE*time_s
    u = torch.frac(u); v = torch.frac(v)

    scale_eff = DUST_SCALE / max(NEBULA_CLOUD_SIZE, 1e-4)

    wx = fbm2(u*scale_eff*0.7 + 11.2 + WARP_TIME_RATE*0.37*time_s + 0.007*_SEED_F,
              v*scale_eff*0.7 +  4.7 + WARP_TIME_RATE*0.21*time_s + 0.011*_SEED_F, octaves=3)
    wy = fbm2(u*scale_eff*0.7 + 23.5 + WARP_TIME_RATE*0.29*time_s + 0.013*_SEED_F,
              v*scale_eff*0.7 + 17.3 + WARP_TIME_RATE*0.41*time_s + 0.017*_SEED_F, octaves=3)
    u2 = u*scale_eff + DUST_WARP*(wx - 0.5)
    v2 = v*scale_eff + DUST_WARP*(wy - 0.5)

    fb = fbm2(u2, v2, octaves=DUST_OCTAVES, lac=2.0, gain=0.52)
    fb = (fb - fb.min())/(fb.max() - fb.min() + 1e-6)

    band = torch.sin((phi + SKY_ROT_RATE*time_s) - BAND_ROT)
    band = torch.exp(-(band*band)/(BAND_SIGMA*BAND_SIGMA))
    density = torch.clamp(fb*(1.0 + BAND_STRENGTH*band), 0, 1)

    alpha = smoothstep(DUST_THRESH, DUST_THRESH + DUST_SOFT, density) * DUST_ALPHA

    csel = fbm2(u*1.7 + 3.1 + 0.019*_SEED_F, v*1.7 + 5.9 + 0.023*_SEED_F, octaves=3)
    csel = (csel - csel.min())/(csel.max()-csel.min()+1e-6)

    LOGO_CYAN    = torch.tensor(NEBULA_COLOR_CYAN, device=device)
    LOGO_MAGENTA = torch.tensor(NEBULA_COLOR_MAGENTA, device=device)
    LOGO_GOLD    = torch.tensor(NEBULA_COLOR_GOLD, device=device)

    w_c = smoothstep(0.00, 0.33, 1.0 - csel)
    w_m = smoothstep(0.33, 0.66, 1.0 - torch.abs(csel-0.5)*2.0)
    w_g = smoothstep(0.66, 1.00, csel)
    denom = (w_c + w_m + w_g + 1e-6)
    w_c, w_m, w_g = w_c/denom, w_m/denom, w_g/denom

    color = (LOGO_CYAN*w_c.unsqueeze(-1) +
             LOGO_MAGENTA*w_m.unsqueeze(-1) +
             LOGO_GOLD*w_g.unsqueeze(-1)) * DUST_GAIN

    # grains flowing with clouds
    GRID   = 2048.0
    uu, vv = u*GRID, v*GRID
    i = torch.floor(uu); j = torch.floor(vv)
    base = _hash2(i, j)
    den   = (0.00008 + 0.00090 * float(NEBULA_GRAININESS))
    sigma = (0.70 - 0.45 * float(NEBULA_GRAININESS))
    mask  = (base > (1.0 - den*400.0)).float()
    du, dv = uu - i - 0.5, vv - j - 0.5
    spot   = torch.exp(-(du*du + dv*dv)/(2*sigma*sigma))
    amp    = (0.5 + 0.5*_hash2(i+7.0 + 0.5*time_s + 0.031*_SEED_F,
                               j+13.0+ 0.35*time_s + 0.043*_SEED_F))
    grains = (mask * spot * amp)
    warm_white = torch.tensor([1.0, 0.97, 0.90], device=device)
    grains_rgb = (0.85*warm_white + 0.15*color) * grains.unsqueeze(-1)
    grains_rgb = grains_rgb * alpha.unsqueeze(-1)

    return color, alpha, grains_rgb

# ========= Shading =========
def warm_colormap(I, device):
    I = torch.clamp(I, 0, 1)
    c1 = torch.tensor([0.95, 0.72, 0.40], device=device)
    c2 = torch.tensor([1.00, 0.92, 0.72], device=device)
    c3 = torch.tensor([1.00, 0.98, 0.94], device=device)
    t1  = torch.clamp(I*1.2, 0, 1); mid = c1*(1-t1).unsqueeze(-1) + c2*t1.unsqueeze(-1)
    t2  = torch.clamp((I-0.6)/0.4, 0, 1)
    return (mid*(1-t2).unsqueeze(-1) + c3*t2.unsqueeze(-1)) * I.unsqueeze(-1)

def doppler_intensity(xi, cam, time_s, device):
    rho = torch.sqrt(xi[...,0]**2 + xi[...,1]**2)
    phi = torch.atan2(xi[...,1], xi[...,0])
    if DISK_ANGVEL != 0.0:
        phi = phi - DISK_ANGVEL * time_s
    v = torch.sqrt(torch.clamp(torch.tensor(BASE_M, device=device), min=EPS) /
                   torch.clamp(rho, min=1e-6))
    v = torch.clamp(v, max=0.7)
    phihat = torch.stack([-torch.sin(phi), torch.cos(phi), torch.zeros_like(phi)], dim=-1)
    los = normalize(cam.view(1,3) - xi)
    gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - v*v, min=1e-6))
    dotp = (phihat * los).sum(dim=-1)
    D = 1.0 / (gamma * (1.0 - v * dotp + 1e-6))
    emiss = (rho / r_in) ** (-Q_EMISS)
    return torch.clamp(emiss * (D ** DOPPLER_PWR), 0.0, 10.0)

# ========= Renderer =========
def render_frame_torch(width, height, time_s, device):
    aspect = width / float(height)
    cam, right, up, fwd = make_camera(device)

    # Fit FOV so the whole disk is visible with a small margin
    t_plane = r_cam / max(math.cos(INCL), 1e-6)
    need    = r_out / max(t_plane, 1e-6)
    FOV_eff = max(BASE_FOV, 2.0*math.atan((1.0 + FIT_MARGIN) * need))

    if time_s == 0.0:
        eps_eff = (core_length_L()/rs) if rs > 0 else 0.0
        bc = photon_ring_b_critical()
        print(
            f"FOV_eff: {math.degrees(FOV_eff):.2f}°, "
            f"cam=({cam[0]:.2f},{cam[1]:.2f},{cam[2]:.2f}), "
            f"BH_SIZE_SCALE={BH_SIZE_SCALE:.2f}, SEED={SEED}, "
            f"model={CORE_MODEL}, eps_eff={eps_eff:.5f}, "
            f"r_h={r_h:.5f}, b_c={bc:.5f}",
            flush=True,
        )

    ys = torch.linspace(-math.tan(FOV_eff/2),        math.tan(FOV_eff/2),        height, device=device)
    xs = torch.linspace(-math.tan(FOV_eff/2)*aspect, math.tan(FOV_eff/2)*aspect, width,  device=device)
    YY, XX = torch.meshgrid(ys, xs, indexing="ij")

    img_accum = torch.zeros((height, width, 3), device=device)
    r_min     = torch.full((height, width), 1e9, device=device)

    for _ in range(max(1, SAMPLES_PER_PIXEL)):
        # Primary rays
        dir_world = normalize(
            fwd.view(1,1,3)
            + right.view(1,1,3)*XX.unsqueeze(-1)
            + up.view(1,1,3)*YY.unsqueeze(-1)
        )

        # Impact parameter & azimuth at the camera (constant along the geodesic)
        b0 = tnorm(torch.cross(cam.view(1,1,3), dir_world), dim=-1)
        phi_cam = torch.atan2(dir_world[...,1], dir_world[...,0])

        # Ray-march setup
        x = cam.view(1,1,3).expand(height, width, 3).clone()
        u = dir_world.clone()

        alive   = torch.ones((height, width), device=device, dtype=torch.bool)
        hitmask = torch.zeros_like(alive)
        bhmask  = torch.zeros_like(alive)
        inten   = torch.zeros((height, width), device=device)
        z_prev  = x[...,2].clone()

        for _step in range(MAX_STEPS):
            z = x[...,2]
            r_here = tnorm(x, dim=-1)

            # Horizon capture
            fell = (r_here < r_h) & alive
            if fell.any():
                bhmask[fell] = True
                alive[fell]  = False

            r_min = torch.minimum(r_min, r_here)

            # Detect crossing of the disk plane z=0 and shade the disk
            cross = (z_prev > 0.0) & (z <= 0.0) & alive & (~hitmask)
            if cross.any():
                t0  = z_prev[cross] / (z_prev[cross] - z[cross] + 1e-12)
                xi  = x[cross] - u[cross] * H_STEP * (1.0 - t0).unsqueeze(-1)
                rho = torch.sqrt(xi[...,0]**2 + xi[...,1]**2)
                in_ring = (rho >= r_in) & (rho <= r_out)
                if in_ring.any():
                    I = doppler_intensity(xi[in_ring], cam, time_s, device)
                    idx = torch.nonzero(cross, as_tuple=False).to(device)
                    idx = idx[in_ring]
                    inten[idx[:,0], idx[:,1]]   = I
                    hitmask[idx[:,0], idx[:,1]] = True
                    alive[idx[:,0], idx[:,1]]   = False

            adv = alive & (~hitmask)
            if not adv.any():
                break

            xa = x[adv]; ua = u[adv]
            g = grad_n(xa)
            n = n_of_r(tnorm(xa, dim=-1))
            du = (g - ua * (ua*g).sum(dim=-1, keepdim=True)) / torch.clamp(n.unsqueeze(-1), min=1e-6)
            ua = normalize(ua + H_STEP * du)
            xa = xa + H_STEP * ua
            u[adv] = ua; x[adv] = xa
            z_prev = z

        # Add disk emission (warm colormap)
        if inten.max() > 0:
            I_norm = inten / (inten.max() + 1e-6)
            img_accum += warm_colormap(I_norm, device)

        # Sample background for all misses (nebula, stars)
        miss = (~hitmask) & (~bhmask)
        if miss.any():
            d_sky = sky_sample_dirs(u, cam, time_s)
            neb_col, neb_a, grains = nebula_rgba_from_dir(d_sky, device, time_s)
            stars = STAR_GAIN * stars_from_dir(d_sky, device, time_s)
            stars *= (1.0 - 0.45*neb_a).unsqueeze(-1)
            bg = neb_a.unsqueeze(-1) * neb_col + grains + stars
            img_accum[miss] += bg[miss]

        # Photon-ring overlay AFTER bg so it blends with the scene
        if ENABLE_PHOTON_RING:
            bc = photon_ring_b_critical()
            base_sigma = PHOTON_RING_SIGMA_COEF * BH_M

            # small thickness jitter around circumference
            nphi = noise2(phi_cam*64.0 + 13.7 + 0.017*_SEED_F,
                          phi_cam*64.0 + 29.3 + 0.019*_SEED_F)
            sigma = base_sigma * (1.0 + PHOTON_RING_NOISE * (2.0*nphi - 1.0))

            # gaussian core in impact-parameter space
            core = torch.exp(-0.5 * ((b0 - bc) / (sigma + 1e-6))**2)

            # soft-knee feather
            alpha = torch.clamp(core, 0.0, 1.0) ** PHOTON_RING_SOFTKNEE

            # subtle warm highlight, but mostly borrow color from what's already there
            warm = warm_colormap(core, device)
            tint = torch.tensor(PHOTON_RING_TINT, device=device).view(1,1,3)
            highlight = warm * tint

            # color that tracks the local background
            ring_rgb = PHOTON_RING_GAIN * alpha.unsqueeze(-1) * (
                PHOTON_RING_BG_MIX * img_accum + (1.0 - PHOTON_RING_BG_MIX) * highlight
            )

            # SCREEN blend: out = 1 - (1 - base) * (1 - ring)
            img_accum = 1.0 - (1.0 - img_accum) * (1.0 - ring_rgb)

    # Force the captured region black
    img_accum[(r_min <= r_h + 1e-3)] = 0.0

    # Post-process: normalize, bloom, gamma
    img_np = img_accum.detach().clamp(0, None).cpu().numpy()
    m = img_np.max()
    if m > 0:
        img_np /= m
    bright = np.clip(img_np - BLOOM_THRESH, 0, 1)
    for c in range(3):
        bright[..., c] = gaussian_filter(bright[..., c], BLOOM_SIGMA)
    img_np = np.clip(img_np + BLOOM_GAIN * bright, 0, 1)
    img_np = np.power(img_np, 1.0/max(GAMMA, 1e-6))
    return (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

# ========= Metadata / Progress =========
def current_params_dict():
    return {
        "SEED": SEED, "RUN_NAME": RUN_NAME,
        "DURATION": DURATION, "FPS": FPS, "RES_W": RES_W, "RES_H": RES_H,
        "BASE_M": BASE_M, "BH_SIZE_SCALE": BH_SIZE_SCALE,
        "r_cam": r_cam, "INCL_deg": math.degrees(INCL),
        "BASE_FOV_deg": math.degrees(BASE_FOV), "FIT_MARGIN": FIT_MARGIN,
        "r_in": r_in, "r_out": r_out,
        "Q_EMISS": Q_EMISS, "DOPPLER_PWR": DOPPLER_PWR, "DISK_ANGVEL": DISK_ANGVEL,
        "MAX_STEPS": MAX_STEPS, "H_STEP": H_STEP,
        "ENABLE_PHOTON_RING": ENABLE_PHOTON_RING,
        "PHOTON_RING_SIGMA_COEF": PHOTON_RING_SIGMA_COEF,
        "PHOTON_RING_GAIN": PHOTON_RING_GAIN,
        "BLOOM_THRESH": BLOOM_THRESH, "BLOOM_SIGMA": BLOOM_SIGMA, "BLOOM_GAIN": BLOOM_GAIN,
        "SKY_DENSITY": SKY_DENSITY, "STAR_GAIN": STAR_GAIN,
        "DUST_GAIN": DUST_GAIN, "DUST_ALPHA": DUST_ALPHA, "DUST_SCALE": DUST_SCALE,
        "DUST_OCTAVES": DUST_OCTAVES, "DUST_WARP": DUST_WARP,
        "DUST_THRESH": DUST_THRESH, "DUST_SOFT": DUST_SOFT,
        "BAND_STRENGTH": BAND_STRENGTH, "BAND_ROT": BAND_ROT, "BAND_SIGMA": BAND_SIGMA,
        "NEBULA_CLOUD_SIZE": NEBULA_CLOUD_SIZE, "NEBULA_GRAININESS": NEBULA_GRAININESS,
        "SKY_ROT_RATE": SKY_ROT_RATE, "VORTEX_STRENGTH": VORTEX_STRENGTH,
        "VORTEX_SIGMA_COEF": VORTEX_SIGMA_COEF, "NEBULA_FLOW_RATE": NEBULA_FLOW_RATE,
        "WARP_TIME_RATE": WARP_TIME_RATE, "TWINKLE_RATE": TWINKLE_RATE,
        "NEBULA_COLOR_CYAN": NEBULA_COLOR_CYAN,
        "NEBULA_COLOR_MAGENTA": NEBULA_COLOR_MAGENTA,
        "NEBULA_COLOR_GOLD": NEBULA_COLOR_GOLD,
        "ENABLE_CORE_DEFORMATION": ENABLE_CORE_DEFORMATION,
        "CORE_MODEL": CORE_MODEL,
        "CORE_EPSILON": CORE_EPSILON,
        "CORE_L0": CORE_L0,
        "CORE_CF": CORE_CF,
        "CORE_CTAU": CORE_CTAU,
        "rs": rs,
        "r_h": r_h,
        "CORE_L_effective": core_length_L(),
        "CORE_eps_effective": (core_length_L()/rs if rs > 0 else 0.0),
        "b_c_critical": photon_ring_b_critical(),
    }

def set_nebula_colors(cyan: Optional[List[float]] = None,
                      magenta: Optional[List[float]] = None,
                      gold: Optional[List[float]] = None) -> None:
    global NEBULA_COLOR_CYAN, NEBULA_COLOR_MAGENTA, NEBULA_COLOR_GOLD
    if cyan is not None:
        NEBULA_COLOR_CYAN = [float(x) for x in cyan]
    if magenta is not None:
        NEBULA_COLOR_MAGENTA = [float(x) for x in magenta]
    if gold is not None:
        NEBULA_COLOR_GOLD = [float(x) for x in gold]

def apply_params(params: dict) -> None:
    global SEED, RUN_NAME, DURATION, FPS, RES_W, RES_H, SAMPLES_PER_PIXEL, GAMMA
    global BH_SIZE_SCALE, BH_M, rs, r_h
    global r_cam, INCL, BASE_FOV, FIT_MARGIN
    global r_in, r_out, Q_EMISS, DOPPLER_PWR, DISK_ANGVEL
    global MAX_STEPS, H_STEP
    global ENABLE_PHOTON_RING, PHOTON_RING_SIGMA_COEF, PHOTON_RING_GAIN
    global BLOOM_THRESH, BLOOM_SIGMA, BLOOM_GAIN
    global SKY_DENSITY, STAR_GAIN
    global DUST_GAIN, DUST_ALPHA, DUST_SCALE, DUST_OCTAVES, DUST_WARP
    global DUST_THRESH, DUST_SOFT, BAND_STRENGTH, BAND_ROT, BAND_SIGMA
    global NEBULA_CLOUD_SIZE, NEBULA_GRAININESS
    global SKY_ROT_RATE, VORTEX_STRENGTH, VORTEX_SIGMA_COEF, NEBULA_FLOW_RATE
    global WARP_TIME_RATE, TWINKLE_RATE
    global ENABLE_CORE_DEFORMATION, CORE_MODEL, CORE_EPSILON, CORE_L0, CORE_CF, CORE_CTAU

    for k, v in (params or {}).items():
        if k == "NEBULA_COLOR_CYAN":
            set_nebula_colors(cyan=v)
        elif k == "NEBULA_COLOR_MAGENTA":
            set_nebula_colors(magenta=v)
        elif k == "NEBULA_COLOR_GOLD":
            set_nebula_colors(gold=v)
        else:
            globals()[k] = v

    # Recompute M, rs, and r_h consistently (respects core deformation & horizon solve)
    update_scales()

    if os.getenv("BH_DEBUG", "0") == "1":
        eps_eff = (core_length_L()/rs) if rs > 0 else 0.0
        print(
            "[apply_params] "
            f"ENABLE_CORE_DEFORMATION={ENABLE_CORE_DEFORMATION}, CORE_MODEL={CORE_MODEL}, "
            f"L_eff={core_length_L():.6f}, eps_eff={eps_eff:.6f}, "
            f"rs={rs:.6f}, r_h={r_h:.6f}, b_c={photon_ring_b_critical():.6f}"
        )

    # keep scalar types sane
    if isinstance(BAND_ROT, (int, float)): BAND_ROT = float(BAND_ROT)
    if isinstance(INCL, (int, float)):     INCL     = float(INCL)
    BASE_FOV = float(BASE_FOV)

def render_image(width: int, height: int, time_s: float, device: Optional[str] = None):
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return render_frame_torch(width, height, time_s, dev)

def write_progress(total_frames, completed_indices):
    data = {"total": total_frames,
            "completed": len(completed_indices),
            "done_indices": sorted(int(i) for i in completed_indices)}
    PROGRESS_JSON.write_text(json.dumps(data, indent=2))

# ========= Orchestration =========
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device.upper())
    W, H = RES_W, RES_H
    total_frames = int(round(DURATION * FPS))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Save / check metadata for reproducibility
    params = current_params_dict()
    if META_JSON.exists():
        try:
            prev = json.loads(META_JSON.read_text())
            if prev.get("SEED") != SEED:
                print(f"[WARN] Seed changed (prev {prev.get('SEED')} -> now {SEED}). "
                      "Exact reproduction requires the same SEED.")
        except Exception as e:
            print("[WARN] Could not read existing meta:", e)
    META_JSON.write_text(json.dumps(params, indent=2))

    # Resume: append from the highest missing frame index
    existing = {int(Path(p).stem[1:]) for p in glob.glob(str(FRAMES_DIR / "f*.png"))}
    todo = [k for k in range(total_frames) if k not in existing]
    completed = set(existing)
    print(f"Frames existing: {len(existing)} / {total_frames}. Rendering {len(todo)} more.")

    for k in tqdm(todo, desc="Rendering (GPU)", unit="frame"):
        # absolute time_s keeps continuity when you extend DURATION
        t = k / float(FPS)
        frame = render_frame_torch(W, H, t, device)
        iio.imwrite((FRAMES_DIR / f"f{k:04d}.png").as_posix(), frame)
        completed.add(k)
        if len(completed) % 10 == 0:
            write_progress(total_frames, completed)

    write_progress(total_frames, completed)

    # Encode MP4 (silent)
    subprocess.run([
        "ffmpeg","-y","-framerate",str(FPS),
        "-i", f"{FRAMES_DIR}/f%04d.png",
        "-c:v","libx264","-pix_fmt","yuv420p",
        SILENT_MP4.as_posix()
    ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Audio + mux
    make_ringdown_wav(AUDIO_WAV.as_posix(), DURATION)
    subprocess.run([
        "ffmpeg","-y",
        "-i", SILENT_MP4.as_posix(),
        "-i", AUDIO_WAV.as_posix(),
        "-c:v","copy","-c:a","aac","-shortest",
        FINAL_MP4.as_posix()
    ], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("Wrote:", FINAL_MP4.as_posix())

if __name__ == "__main__":
    main()
