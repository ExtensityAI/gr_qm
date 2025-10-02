
import numpy as np
from bh_gym_env import BlackHoleParamEnv, EnvConfig, TargetSpec
import simulation_rl as sim

# Example: Fit epsilon from synthetic ringdown data at M=70 Msun
Ms = 70.0
true_eps = 0.06
sim.set_core_params(eps=true_eps, model="fractional")
f_true, tau_true = sim.qnm_220_with_core(Ms)

target = TargetSpec(
    f220_Hz=f_true, sigma_f_Hz=0.5,   # pretend 0.5 Hz uncertainty
    tau220_s=tau_true, sigma_tau_s=1e-3,  # 1 ms uncertainty
    M_solar=Ms
)

cfg = EnvConfig(core_model="fractional", eps_min=0.0, eps_max=0.15,
                step_size=0.01, max_steps=32)

env = BlackHoleParamEnv(target, config=cfg)

obs, _ = env.reset(seed=42)
done = False
total_r = 0.0
while True:
    # Random search as a placeholder for your RL algorithm
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    total_r += r
    if term or trunc:
        break

print("Episode finished. chi2=", info["chi2"], " param(eps or L0)=", info["param"])
# Optional: quick preview image (low-res)
try:
    img = env.render()
    print("Render shape:", None if img is None else img.shape)
except Exception as e:
    print("Render skipped:", e)
