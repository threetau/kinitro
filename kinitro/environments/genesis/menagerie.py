"""Auto-download MuJoCo Menagerie robot assets for local development.

In Docker, assets are baked into the image at /opt/menagerie (see
environments/genesis/Dockerfile). For local dev, this module downloads
them on first use to ~/.cache/kinitro/menagerie and reuses the cache
on subsequent runs.

The GENESIS_MENAGERIE_PATH env var always takes precedence — if set,
no download is attempted.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger()

MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Pinned commit — must match environments/genesis/Dockerfile ARG MENAGERIE_COMMIT
MENAGERIE_COMMIT = "a03e87bf13502b0b48ebbf2808928fd96ebf9cf3"
# Only check out the directories we actually need (sparse checkout)
MENAGERIE_SPARSE_DIRS = ["unitree_g1"]

# Default cache location (XDG-style)
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "kinitro" / "menagerie"


def _get_cache_dir() -> Path:
    """Return the cache directory for menagerie assets.

    Respects XDG_CACHE_HOME if set, otherwise uses ~/.cache/kinitro/menagerie.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "kinitro" / "menagerie"
    return _DEFAULT_CACHE_DIR


def _is_menagerie_valid(path: Path) -> bool:
    """Check whether *path* contains the expected menagerie assets.

    Validates that every directory listed in MENAGERIE_SPARSE_DIRS exists
    and is non-empty. This is a lightweight check — we don't verify
    individual XML files because the pinned commit guarantees contents.
    """
    for d in MENAGERIE_SPARSE_DIRS:
        dir_path = path / d
        if not dir_path.is_dir():
            return False
        # Check it's not an empty directory (git sparse-checkout stub)
        if not any(dir_path.iterdir()):
            return False
    return True


def _download_menagerie(dest: Path) -> None:
    """Clone MuJoCo Menagerie via git sparse-checkout into *dest*.

    Mirrors the Dockerfile approach: shallow fetch of a single pinned
    commit with only the needed subdirectories checked out.

    Raises RuntimeError if git is not installed or the clone fails.
    """
    logger.info(
        "downloading MuJoCo Menagerie assets",
        dest=str(dest),
        commit=MENAGERIE_COMMIT[:12],
        sparse_dirs=MENAGERIE_SPARSE_DIRS,
    )

    # Verify git is available
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "git is required to download MuJoCo Menagerie assets but was not found. "
            "Install git or manually clone https://github.com/google-deepmind/mujoco_menagerie "
            "and set GENESIS_MENAGERIE_PATH."
        )

    # Clean up any partial previous download
    if dest.exists():
        shutil.rmtree(dest)

    dest.mkdir(parents=True, exist_ok=True)

    # Run the same sequence as the Dockerfile:
    # git init → add remote → sparse-checkout set → fetch → checkout
    commands: list[list[str]] = [
        ["git", "init"],
        ["git", "remote", "add", "origin", MENAGERIE_REPO],
        ["git", "sparse-checkout", "set", *MENAGERIE_SPARSE_DIRS],
        ["git", "fetch", "--depth", "1", "origin", MENAGERIE_COMMIT],
        ["git", "checkout", "FETCH_HEAD"],
    ]

    for cmd in commands:
        result = subprocess.run(
            cmd,
            cwd=str(dest),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MuJoCo Menagerie download failed at step: {' '.join(cmd)}\n"
                f"stderr: {result.stderr.strip()}\n"
                "You can manually clone https://github.com/google-deepmind/mujoco_menagerie "
                "and set GENESIS_MENAGERIE_PATH."
            )

    logger.info("MuJoCo Menagerie download complete", path=str(dest))


def ensure_menagerie() -> str:
    """Ensure MuJoCo Menagerie assets are available and return the root path.

    Resolution order:
    1. ``GENESIS_MENAGERIE_PATH`` env var — used as-is, no validation or
       download (the operator knows what they're doing).
    2. Default Docker path ``/opt/menagerie`` — used if it exists and is
       valid (we're running inside the Genesis container).
    3. Local cache ``~/.cache/kinitro/menagerie`` — downloaded on first
       use via git sparse-checkout, reused on subsequent runs.

    Returns the absolute path to the menagerie root directory.
    """
    # 1. Explicit env var — trust it unconditionally
    env_path = os.environ.get("GENESIS_MENAGERIE_PATH")
    if env_path:
        logger.debug("using GENESIS_MENAGERIE_PATH from environment", path=env_path)
        return env_path

    # 2. Docker default path
    docker_path = Path("/opt/menagerie")
    if _is_menagerie_valid(docker_path):
        logger.debug("using menagerie from Docker default path", path=str(docker_path))
        return str(docker_path)

    # 3. Local cache — download if missing or invalid
    cache_dir = _get_cache_dir()
    if _is_menagerie_valid(cache_dir):
        logger.debug("using cached menagerie assets", path=str(cache_dir))
        return str(cache_dir)

    # Download to cache
    _download_menagerie(cache_dir)

    # Validate after download
    if not _is_menagerie_valid(cache_dir):
        raise RuntimeError(
            f"MuJoCo Menagerie download completed but assets are missing at {cache_dir}. "
            "This may indicate a network issue or changed repository structure. "
            "Try deleting the cache directory and running again."
        )

    return str(cache_dir)
