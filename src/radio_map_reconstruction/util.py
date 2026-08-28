from shutil import rmtree

from radio_map_reconstruction.config import (
    reconstructor_run_dir,
    run_root_dir,
    sampler_run_dir,
)


def delete_reconstructor_run() -> None:
    run_dir = reconstructor_run_dir().resolve()
    run_root = run_root_dir().resolve()
    sampler_dir = sampler_run_dir().resolve()
    if (
        run_dir == run_root
        or not run_dir.is_relative_to(run_root)
        or sampler_dir.is_relative_to(run_dir)
        or run_dir.is_relative_to(sampler_dir)
    ):
        raise ValueError(
            "Reconstructor run directory must be an isolated child of the run root"
        )
    if run_dir.exists():
        rmtree(run_dir)
