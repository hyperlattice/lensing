"""Fetch two pinned author artifacts missing from the sparse research clone."""
import hashlib
from pathlib import Path
import urllib.request

COMMIT = "17e42acb13f0c3111c8a4e665222e215b10d2e7b"
FILES = {
    "src/training/MVEUDA/paper_models/RunA_mve": "7b571b176e274487b2ff493124542b447e444e93",
    "src/training/MVEUDA/ModelVizPaper.ipynb": "a3db43b16968422cf8b9e936591d7d844612d082",
    "src/training/MVEUDA/paper_models/RunA_snll": "ef4c7d5ba9c122adf691980abd715680623ee5d4",
    "src/training/MVEonly/paper_models/RunA_noDA_snll": "86b6f539cca68ded1e39ae45f2432ea88b425148",
}
ROOT = Path("data/external_reference_research/agarwal_author_artifacts_v1")

for relative, expected_git_sha1 in FILES.items():
    url = ("https://raw.githubusercontent.com/deepskies/"
           f"DomainAdaptiveMVEforLensModeling/{COMMIT}/{relative}")
    data = urllib.request.urlopen(url, timeout=60).read()
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != expected_git_sha1:
        raise ValueError(f"Pinned Git blob mismatch for {relative}: {actual}")
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(relative, len(data), actual, flush=True)
