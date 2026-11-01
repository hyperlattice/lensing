"""Run the authorized development stages after all qualified banks complete.

This worker can wait for an active bank build. It never freezes scientific
choices automatically and never restarts a partially started stage or fit.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from scripts.run_robustness_development_v1 import CONFIG


def main(config_path,wait):
    config=yaml.safe_load(Path(config_path).read_text())
    paths=[Path(config[k]) for k in ("source_train","source_development","target_gaussian","target_representative")]
    while not all(p.exists() and (p.parent/"complete.json").exists() for p in paths):
        if not wait:
            raise ValueError("All qualified bank completion reports are required")
        for p in paths:
            if (p.parent/"failed.json").exists():
                raise ValueError("A bank failed; development cannot proceed")
        time.sleep(15)
    output=Path(config["output"])
    output.mkdir(parents=True,exist_ok=True)
    marker=output/"development_execution_started.json"
    with marker.open("x") as stream:
        json.dump({"created_utc":datetime.now(timezone.utc).isoformat(),"config":str(config_path),
            "stages":["reference","weights","tune"],"final_data_access":False},stream,indent=2)
    for action in ("reference","weights","tune"):
        subprocess.run([sys.executable,"-B","-m","scripts.run_robustness_development_v1",action,
            "--config",str(config_path)],check=True)
    subprocess.run([sys.executable,"-B","-m","scripts.review_robustness_development_v1","summarize",
        "--config",str(config_path)],check=True)
    print("Source-only initial tuning complete; boundary and convergence review required before freezing.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(CONFIG))
    parser.add_argument("--wait-for-banks",action="store_true")
    args=parser.parse_args()
    main(args.config,args.wait_for_banks)
