"""Fetch and checksum the published Agarwal et al. benchmark archives."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

FILES = {
    "target": ("mb_paper_target_final.zip", 3804478112, "9839a0ae65593405b1d4ebe6d1076117"),
    "source": ("mb_paper_source_final.zip", 3768480940, "e79d4cef608ed4203c5e2e48dd81cf29"),
}
ROOT = Path("data/external_reference_research/agarwal_zenodo_v1")


def fetch(role):
    name, expected_bytes, expected_md5 = FILES[role]
    ROOT.mkdir(parents=True, exist_ok=True)
    final = ROOT / name
    part = ROOT / (name + ".part")
    if final.exists():
        path = final
    else:
        size = part.stat().st_size if part.exists() else 0
        if size > expected_bytes:
            raise ValueError("Partial download exceeds declared archive size")
        url = f"https://zenodo.org/api/records/13647416/files/{name}/content"
        request = urllib.request.Request(url, headers={"Range": f"bytes={size}-"} if size else {})
        with urllib.request.urlopen(request, timeout=120) as response:
            if size and response.status != 206:
                raise ValueError("Server did not honor resume range; preserved partial file")
            with part.open("ab") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        if part.stat().st_size != expected_bytes:
            raise ValueError("Incomplete archive; partial file retained for resume")
        part.replace(final)
        path = final
    md5 = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            md5.update(chunk)
    if path.stat().st_size != expected_bytes or md5.hexdigest() != expected_md5:
        raise ValueError("Published archive size or MD5 mismatch")
    report = {"role": role, "archive": str(path), "bytes": expected_bytes,
              "md5": expected_md5, "record": "https://zenodo.org/records/13647416"}
    (ROOT / f"{role}_verified.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=FILES)
    fetch(parser.parse_args().role)
