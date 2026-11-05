"""Fetch pinned public author source for inspection; never execute downloaded code."""
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "17e42acb13f0c3111c8a4e665222e215b10d2e7b"
FILES = [".github/README.md", ".github/LICENSE.txt", "envs/training_exact_env.yml",
         "envs/deeplenstronomy_exact_env.yml", "src/training/MVEUDA/train_utils.py",
         "src/training/MVEonly/train_utils.py", "src/training/MVEUDA/MVE_UDA_RunA.ipynb",
         "src/training/MVEonly/MVE_noDA_RunA.ipynb"]


def main():
    root = Path("data/external_reference_research/mve-source-17e42ac")
    report = Path("results/reports/prior-baseline-intake-v1/source_manifest.json")
    if root.exists() or report.exists():
        raise FileExistsError("Pinned intake already exists; use a new intake ID")
    root.mkdir(parents=True)
    records = []
    for name in FILES:
        url = f"https://raw.githubusercontent.com/deepskies/DomainAdaptiveMVEforLensModeling/{COMMIT}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "lensing-research-reproduction"})
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        records.append({"path": name, "url": url, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
        print(f"Fetched {name} ({len(data)} bytes)", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"commit": COMMIT,
        "repository": "https://github.com/deepskies/DomainAdaptiveMVEforLensModeling",
        "files": records, "downloaded_code_executed": False,
        "paper_reproduction_complete": False,
        "git_intake_note": "Partial clone retained separately; Windows sparse checkout failed. Exact raw commit files fetched instead."}, indent=2)+"\n")


if __name__ == "__main__":
    main()
