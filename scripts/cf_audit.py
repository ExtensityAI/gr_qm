
#!/usr/bin/env python3
import os, json, importlib, numpy as np
RE_REF, IM_REF = 0.37367168, -0.08896232
report = {"schwarzschild": None, "kerr": [], "notes": []}
try:
    lmod = importlib.import_module("src.leaver_schwarzschild")
    w = lmod.qnm_l2n0_schw()
    report["schwarzschild"] = {"Re": float(np.real(w)), "Im": float(np.imag(w)),
                               "dRe": float(np.real(w)-RE_REF), "dIm": float(np.imag(w)-IM_REF)}
except Exception as e:
    report["notes"].append(f"Schwarzschild CF not available: {e!r}")
try:
    kmod = importlib.import_module("src.teukolsky_cf")
    for a in [0.0, 0.3, 0.6, 0.9]:
        try:
            wk = kmod.qnm_l2m2n0_kerr(a)
            report["kerr"].append({"a": a, "Re": float(np.real(wk)), "Im": float(np.imag(wk))})
        except Exception as ke:
            report["notes"].append(f"Kerr CF at a={a} not available: {ke!r}")
except Exception as e:
    report["notes"].append(f"Kerr CF module not importable: {e!r}")
os.makedirs("data/cf_audit", exist_ok=True)
with open("data/cf_audit/report.json","w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
