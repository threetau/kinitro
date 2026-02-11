"""Tests for MuJoCo Menagerie auto-download utility."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from kinitro.environments.genesis.menagerie import (
    MENAGERIE_COMMIT,
    MENAGERIE_REPO,
    MENAGERIE_SPARSE_DIRS,
    _get_cache_dir,
    _is_menagerie_valid,
    ensure_menagerie,
)


class TestIsMenagerieValid:
    """Tests for _is_menagerie_valid."""

    def test_valid_when_dirs_exist_and_non_empty(self, tmp_path: Path):
        """Should return True when all sparse dirs exist and have content."""
        for d in MENAGERIE_SPARSE_DIRS:
            dir_path = tmp_path / d
            dir_path.mkdir(parents=True)
            (dir_path / "model.xml").write_text("<mujoco/>")
        assert _is_menagerie_valid(tmp_path) is True

    def test_invalid_when_dir_missing(self, tmp_path: Path):
        """Should return False when a sparse dir is missing."""
        assert _is_menagerie_valid(tmp_path) is False

    def test_invalid_when_dir_empty(self, tmp_path: Path):
        """Should return False when a sparse dir exists but is empty."""
        for d in MENAGERIE_SPARSE_DIRS:
            (tmp_path / d).mkdir(parents=True)
        assert _is_menagerie_valid(tmp_path) is False

    def test_invalid_when_path_does_not_exist(self, tmp_path: Path):
        """Should return False for a nonexistent path."""
        assert _is_menagerie_valid(tmp_path / "nonexistent") is False


class TestGetCacheDir:
    """Tests for _get_cache_dir."""

    def test_respects_xdg_cache_home(self, tmp_path: Path):
        """Should use XDG_CACHE_HOME when set."""
        with patch.dict("os.environ", {"XDG_CACHE_HOME": str(tmp_path)}):
            result = _get_cache_dir()
        assert result == tmp_path / "kinitro" / "menagerie"

    def test_default_without_xdg(self):
        """Should use ~/.cache/kinitro/menagerie when XDG_CACHE_HOME is unset."""
        # Copy env and remove XDG_CACHE_HOME if present
        env = os.environ.copy()
        env.pop("XDG_CACHE_HOME", None)
        with patch.dict("os.environ", env, clear=True):
            result = _get_cache_dir()
        assert result == Path.home() / ".cache" / "kinitro" / "menagerie"


class TestEnsureMenagerie:
    """Tests for ensure_menagerie resolution order."""

    def test_env_var_takes_precedence(self, tmp_path: Path):
        """GENESIS_MENAGERIE_PATH should be returned as-is without validation."""
        custom_path = str(tmp_path / "custom")
        with patch.dict("os.environ", {"GENESIS_MENAGERIE_PATH": custom_path}):
            result = ensure_menagerie()
        assert result == custom_path

    def test_docker_path_used_when_valid(self, tmp_path: Path):
        """Should use /opt/menagerie when it exists and is valid."""
        # Create valid menagerie at a mock docker path
        docker_path = tmp_path / "opt" / "menagerie"
        for d in MENAGERIE_SPARSE_DIRS:
            dir_path = docker_path / d
            dir_path.mkdir(parents=True)
            (dir_path / "model.xml").write_text("<mujoco/>")

        # Copy env and remove GENESIS_MENAGERIE_PATH if present
        env = os.environ.copy()
        env.pop("GENESIS_MENAGERIE_PATH", None)

        with (
            patch.dict("os.environ", env, clear=True),
            patch("kinitro.environments.genesis.menagerie.Path") as mock_path_cls,
            patch("kinitro.environments.genesis.menagerie._is_menagerie_valid") as mock_valid,
        ):
            mock_path_cls.return_value = docker_path
            # First call is for docker path (valid)
            mock_valid.side_effect = [True]
            result = ensure_menagerie()
            assert result == str(docker_path)

    def test_cache_used_when_valid(self, tmp_path: Path):
        """Should use cache when it exists and is valid (no download)."""
        cache_dir = tmp_path / "cache"
        for d in MENAGERIE_SPARSE_DIRS:
            dir_path = cache_dir / d
            dir_path.mkdir(parents=True)
            (dir_path / "model.xml").write_text("<mujoco/>")

        # Copy env and remove GENESIS_MENAGERIE_PATH if present
        env = os.environ.copy()
        env.pop("GENESIS_MENAGERIE_PATH", None)

        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "kinitro.environments.genesis.menagerie._get_cache_dir",
                return_value=cache_dir,
            ),
            patch("kinitro.environments.genesis.menagerie._download_menagerie") as mock_dl,
        ):
            result = ensure_menagerie()
            mock_dl.assert_not_called()
            assert result == str(cache_dir)

    def test_downloads_when_cache_missing(self, tmp_path: Path):
        """Should call _download_menagerie when cache is empty."""
        cache_dir = tmp_path / "cache"

        def fake_download(dest: Path) -> None:
            """Simulate a successful download by creating expected dirs."""
            for d in MENAGERIE_SPARSE_DIRS:
                dir_path = dest / d
                dir_path.mkdir(parents=True, exist_ok=True)
                (dir_path / "model.xml").write_text("<mujoco/>")

        # Copy env and remove GENESIS_MENAGERIE_PATH if present
        env = os.environ.copy()
        env.pop("GENESIS_MENAGERIE_PATH", None)

        with (
            patch.dict("os.environ", env, clear=True),
            patch(
                "kinitro.environments.genesis.menagerie._get_cache_dir",
                return_value=cache_dir,
            ),
            patch(
                "kinitro.environments.genesis.menagerie._download_menagerie",
                side_effect=fake_download,
            ) as mock_dl,
        ):
            result = ensure_menagerie()
            mock_dl.assert_called_once_with(cache_dir)
            assert result == str(cache_dir)

    def test_constants_match_dockerfile(self):
        """Pinned commit and repo URL should match the Dockerfile."""
        # These are important invariants — if either changes, both should update
        assert MENAGERIE_COMMIT == "a03e87bf13502b0b48ebbf2808928fd96ebf9cf3"
        assert "mujoco_menagerie" in MENAGERIE_REPO
        assert "unitree_g1" in MENAGERIE_SPARSE_DIRS
