#!/usr/bin/env python
"""Build optional D2D optimizer Cython extensions in place."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER_SOURCES = (
    REPO_ROOT / "megatron/core/optimizer/d2d_distrib_optimizer.py",
    REPO_ROOT / "megatron/core/optimizer/d2d_optimizer_comm.py",
)


def show_help() -> None:
    """Print the supported command-line interface."""
    print("python tools/cython_setup.py -h | --help")
    print("python tools/cython_setup.py build")


def _remove_generated_sources(sources: list[Path]) -> None:
    for source in sources:
        source.unlink(missing_ok=True)
        source.with_suffix(".c").unlink(missing_ok=True)


def build_extensions() -> int:
    """Build every existing optional optimizer extension and install it in place."""
    sources = [source for source in OPTIMIZER_SOURCES if source.exists()]
    if not sources:
        print("No D2D optimizer Cython sources are present; nothing to build.")
        return 0

    from Cython.Build import cythonize
    from Cython.Compiler import Options
    from setuptools import setup

    Options.error_on_unknown_names = False
    previous_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        extensions = cythonize(
            [str(source.relative_to(REPO_ROOT)) for source in sources],
            compiler_directives={"language_level": 3},
        )
        setup(
            name="megatron_optimizer_cython_extensions",
            script_args=["build_ext", "--inplace"],
            ext_modules=extensions,
        )

        missing_extensions = [
            source
            for source in sources
            if not list(source.parent.glob(f"{source.stem}*.so"))
        ]
        if missing_extensions:
            missing = ", ".join(str(path.relative_to(REPO_ROOT)) for path in missing_extensions)
            raise RuntimeError(f"Cython build completed without an in-place extension for: {missing}")

        _remove_generated_sources(sources)
        return 0
    finally:
        shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
        os.chdir(previous_cwd)


def main(argv: list[str] | None = None) -> int:
    """Run the build tool."""
    argv = sys.argv[1:] if argv is None else argv
    if argv in (["-h"], ["--help"]):
        show_help()
        return 0
    if argv != ["build"]:
        show_help()
        return 0

    try:
        return build_extensions()
    except Exception as error:
        print(f"Cython build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
