from pathlib import Path
from shutil import rmtree

ROOT = Path(__file__).resolve().parents[2]

def delete_run():
    run_dir = ROOT / "run"
    if run_dir.exists():
        rmtree(run_dir)