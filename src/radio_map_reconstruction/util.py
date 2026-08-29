from pathlib import Path
from shutil import rmtree

from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]

def delete_run():
    run_dir = ROOT / "run"
    if run_dir.exists():
        rmtree(run_dir)


def sample_identity(dataset: Dataset, index: int, *, fallback_prefix: str) -> str:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        return f"{fallback_prefix}-{index}"
    gain_path = samples[index][0]
    return Path(gain_path).stem
