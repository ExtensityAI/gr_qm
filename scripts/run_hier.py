
#!/usr/bin/env python3
import os, json
from src.hierarchical import run_hier

if __name__ == "__main__":
    repo = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(repo)  # scripts/ -> repo root
    summary = run_hier(repo_base=repo,
                       outdir_rel="data/hier",
                       td_csv_rel="data/hayward_td_qnms_dense.csv",
                       events_csv_rel="data/hier/events_o3b_tablexiii.csv",
                       slope_systematic=0.2,
                       make_plots=True)
    print(json.dumps(summary, indent=2))
