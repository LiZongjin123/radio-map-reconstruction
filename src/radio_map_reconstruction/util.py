from pathlib import Path
from shutil import rmtree

from torch import Generator, randperm
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


def select_example_indices(
    *,
    available_examples: int,
    requested_examples: int | None,
    global_seed: int,
) -> tuple[int, ...]:
    """Select a deterministic, prefix-stable permutation of Examples."""
    if requested_examples is None:
        return tuple(range(available_examples))
    if (
        isinstance(requested_examples, bool)
        or not isinstance(requested_examples, int)
        or requested_examples <= 0
    ):
        raise ValueError("requested Examples must be a positive integer")
    if requested_examples > available_examples:
        raise ValueError(
            "requested Example count exceeds available validation Examples: "
            f"requested {requested_examples}, available {available_examples}"
        )
    generator = Generator().manual_seed(global_seed)
    return tuple(
        randperm(available_examples, generator=generator)[:requested_examples].tolist()
    )
