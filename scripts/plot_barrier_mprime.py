#!/usr/bin/env python3
import json
import os
import matplotlib

# Use a non-interactive backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
INP = os.path.join(ROOT, 'data', 'diagnostics', 'barrier_diagnostics.json')
OUT = os.path.join(ROOT, 'data', 'diagnostics', 'mprime_at_barrier.pdf')

with open(INP, 'r') as f:
    data = json.load(f)

x = data.get('L_over_rs', [])
y = data.get('mprime_over_M_at_rpeak', [])

# Figure style consistent with other repo plots
plt.figure(figsize=(6.2, 4.2), dpi=150)
plt.plot(x, y, marker='o', ms=4.0, lw=2.0, color='#1f77b4')
plt.xlabel(r'$\varepsilon = L/r_s$')
# Keep mathtext simple to avoid parser issues on older matplotlib versions
plt.ylabel(r"$m'(r_{peak})/M$")
plt.title("Interior gradient at the barrier vs core size")
plt.grid(True, ls=':', alpha=0.6)
plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT)
print(f'Wrote {OUT}')
