"""Build the three development roles sequentially with a global two-worker cap."""
from pathlib import Path
import subprocess
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    for role in ("source_train","source_development","target_adaptation"):
        python=root/".venv-autolens/Scripts/python.exe" if role=="target_adaptation" else Path(sys.executable)
        subprocess.run([str(python),"-B","-m","scripts.generate_robustness_banks_parallel_v3",role,"--workers","2"],cwd=root,check=True)


if __name__=="__main__":
    main()
