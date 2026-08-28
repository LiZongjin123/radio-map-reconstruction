from pathlib import Path
from shutil import rmtree

from radio_map_reconstruction.config import (
    reconstructor_run_dir,
    run_root_dir,
    sampler_run_dir,
)


def _delete_isolated_run_dir(
    run_dir: Path,
    other_run_dir: Path,
    *,
    owner: str,
) -> None:
    run_dir = run_dir.resolve()
    run_root = run_root_dir().resolve()
    other_run_dir = other_run_dir.resolve()
    if (
        run_dir == run_root
        or not run_dir.is_relative_to(run_root)
        or other_run_dir.is_relative_to(run_dir)
        or run_dir.is_relative_to(other_run_dir)
    ):
        raise ValueError(
            f"{owner} run directory must be an isolated child of the run root"
        )
    if run_dir.exists():
        rmtree(run_dir)


def delete_reconstructor_run() -> None:
    _delete_isolated_run_dir(
        reconstructor_run_dir(),
        sampler_run_dir(),
        owner="Reconstructor",
    )


def delete_sampler_run() -> None:
    _delete_isolated_run_dir(
        sampler_run_dir(),
        reconstructor_run_dir(),
        owner="Sampler",
    )
