"""
Universal agent loader for miner submissions.

Handles dynamic loading of miner-submitted agent implementations while
maintaining security and providing clear error messages.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .agent_interface import AgentInterface


class AgentLoader:
    """
    Loads miner-submitted agent implementations

    Expected submission structure:
    submissions/
    ├── example_submission/
        ├── agent.py          # Required: Must contain Agent class implementing AgentInterface
        ├── model.*           # Model weights (any format: .safetensors, .pt, .npz, etc.)
        ├── config.json       # Optional: Model configuration
        ├── requirements.txt  # Optional: Additional dependencies (auto-installed if present)
        └── README.md         # Optional: Documentation
    """

    @staticmethod
    def load_agent(submission_path: str | Path, **kwargs) -> AgentInterface:
        """
        Load agent from miner submission directory.

        Args:
            submission_path: Path to directory containing agent.py and model files
            **kwargs: Additional arguments passed to Agent constructor

        Returns:
            AgentInterface: Loaded agent ready for evaluation

        Raises:
            ValueError: If submission structure is invalid or agent doesn't implement interface
            ImportError: If required dependencies are missing
            Exception: Any other errors during loading (passed through for debugging)
        """
        submission_dir = Path(submission_path)

        if not submission_dir.exists():
            raise ValueError(f"Submission directory does not exist: {submission_path}")

        if not submission_dir.is_dir():
            raise ValueError(f"Submission path is not a directory: {submission_path}")

        agent_file = submission_dir / "agent.py"
        if not agent_file.exists():
            raise ValueError(f"Required agent.py not found in {submission_path}")

        # Install requirements if requirements.txt exists
        # AgentLoader._install_requirements(submission_dir)

        try:
            # Load the agent module dynamically
            module_name = f"miner_agent_{hash(str(submission_dir))}"
            spec = importlib.util.spec_from_file_location(module_name, agent_file)

            if spec is None or spec.loader is None:
                raise ValueError(f"Failed to create module spec for {agent_file}")

            agent_module = importlib.util.module_from_spec(spec)

            # Add to sys.modules to support relative imports
            sys.modules[module_name] = agent_module

            # Execute the module
            spec.loader.exec_module(agent_module)

            # Check for required Agent class
            if not hasattr(agent_module, "Agent"):
                raise ValueError("agent.py must contain an 'Agent' class")

            agent_class = getattr(agent_module, "Agent")

            # Instantiate the agent with submission directory
            agent = agent_class(submission_dir, **kwargs)

            # Verify it implements the required interface
            if not isinstance(agent, AgentInterface):
                raise ValueError(
                    f"Agent class must inherit from AgentInterface. "
                    f"Got {type(agent).__name__}, expected subclass of AgentInterface."
                )

            return agent

        except Exception as e:
            # Clean up sys.modules on failure
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise e

    @staticmethod
    def _install_requirements(submission_dir: Path) -> None:
        """Install requirements from requirements.txt if it exists."""
        requirements_file = submission_dir / "requirements.txt"
        if not requirements_file.exists():
            return

        # Skip if already installed recently (cache for 1 hour) - but not in Ray workers

        is_ray_worker = "RAY_ADDRESS" in os.environ or "/ray/session_" in os.getcwd()

        cache_file = submission_dir / ".requirements_installed"

        if cache_file.exists() and not is_ray_worker:
            try:
                last_install = cache_file.stat().st_mtime
                if time.time() - last_install < 3600:  # 1 hour
                    print(
                        f"📦 Requirements recently installed, skipping (cache: {cache_file})",
                        flush=True,
                    )
                    return
            except Exception:
                pass
        elif is_ray_worker:
            print(
                "📦 Ray worker detected, forcing requirements installation", flush=True
            )

        print(f"📦 Installing requirements from {requirements_file}", flush=True)
        print("=" * 60, flush=True)

        # Read requirements to show what we're installing
        with open(requirements_file, "r") as f:
            requirements_content = f.read().strip()
        print(f"Requirements to install:\n{requirements_content}", flush=True)
        print("=" * 60, flush=True)

        # Parse and split requirements so we can install certain packages with special flags
        requirement_lines = [
            line.strip()
            for line in requirements_content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        lerobot_req: Optional[str] = next(
            (ln for ln in requirement_lines if ln.startswith("lerobot")), None
        )
        other_reqs = [ln for ln in requirement_lines if ln != lerobot_req]

        try:
            # Try to use uv pip if available (faster and uses uv environment)
            try:
                print("🚀 Installing with uv pip...", flush=True)

                # In Ray workers, install lerobot with all dependencies to avoid import issues
                if lerobot_req is not None:
                    if is_ray_worker:
                        print(
                            f"Ray worker: Installing with all deps: {lerobot_req}",
                            flush=True,
                        )
                        process = subprocess.Popen(
                            ["uv", "pip", "install", lerobot_req],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                        )
                    else:
                        print(f"Installing without deps: {lerobot_req}", flush=True)
                        process = subprocess.Popen(
                            ["uv", "pip", "install", "--no-deps", lerobot_req],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                        )
                    while True:
                        output = process.stdout.readline()
                        if output == "" and process.poll() is not None:
                            break
                        if output:
                            print(output.strip(), flush=True)
                    return_code = process.poll()
                    if return_code != 0:
                        raise subprocess.CalledProcessError(
                            return_code, "uv pip install --no-deps lerobot"
                        )

                # Next, install the remaining requirements normally
                if other_reqs:
                    tmp_req_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w", delete=False
                        ) as tmp_req:
                            tmp_req.write("\n".join(other_reqs) + "\n")
                            tmp_req_path = tmp_req.name

                        process = subprocess.Popen(
                            [
                                "uv",
                                "pip",
                                "install",
                                "--force-reinstall",
                                "-r",
                                tmp_req_path,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                        )
                        while True:
                            output = process.stdout.readline()
                            if output == "" and process.poll() is not None:
                                break
                            if output:
                                print(output.strip(), flush=True)
                        return_code = process.poll()
                        if return_code != 0:
                            raise subprocess.CalledProcessError(
                                return_code, "uv pip install -r <temp>"
                            )
                    finally:
                        if tmp_req_path and os.path.exists(tmp_req_path):
                            try:
                                os.unlink(tmp_req_path)
                            except Exception:
                                pass

                print("✅ Requirements installed successfully with uv!", flush=True)
                # Create cache file to skip future installs
                try:
                    cache_file = submission_dir / ".requirements_installed"
                    cache_file.touch()
                except Exception:
                    pass

            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                if isinstance(e, FileNotFoundError):
                    print("⚠️  uv not found, falling back to pip...", flush=True)
                else:
                    print(
                        f"⚠️  uv installation failed (exit code {e.returncode}), falling back to pip...",
                        flush=True,
                    )

                # Fallback to regular pip with the same two-step approach
                print("🐍 Installing with pip...", flush=True)
                python_executable = sys.executable

                # Install lerobot without deps first
                if lerobot_req is not None:
                    process = subprocess.Popen(
                        [
                            python_executable,
                            "-m",
                            "pip",
                            "install",
                            "--no-deps",
                            lerobot_req,
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )
                    while True:
                        output = process.stdout.readline()
                        if output == "" and process.poll() is not None:
                            break
                        if output:
                            print(output.strip(), flush=True)
                    return_code = process.poll()
                    if return_code != 0:
                        raise subprocess.CalledProcessError(
                            return_code, "pip install --no-deps lerobot"
                        )

                # Now install remaining requirements (if any)
                if other_reqs:
                    tmp_req_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w", delete=False
                        ) as tmp_req:
                            tmp_req.write("\n".join(other_reqs) + "\n")
                            tmp_req_path = tmp_req.name

                        process = subprocess.Popen(
                            [
                                python_executable,
                                "-m",
                                "pip",
                                "install",
                                "-r",
                                tmp_req_path,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                        )
                        while True:
                            output = process.stdout.readline()
                            if output == "" and process.poll() is not None:
                                break
                            if output:
                                print(output.strip(), flush=True)
                        return_code = process.poll()
                        if return_code != 0:
                            raise subprocess.CalledProcessError(
                                return_code, "pip install -r <temp>"
                            )
                    finally:
                        if tmp_req_path and os.path.exists(tmp_req_path):
                            try:
                                os.unlink(tmp_req_path)
                            except Exception:
                                pass

                print("✅ Requirements installed successfully with pip!", flush=True)
                # Create cache file to skip future installs
                try:
                    cache_file = submission_dir / ".requirements_installed"
                    cache_file.touch()
                except Exception:
                    pass

        except subprocess.CalledProcessError as e:
            print(
                f"❌ Failed to install requirements (exit code {e.returncode})",
                flush=True,
            )
            print(
                "Proceeding anyway - agent may fail if dependencies are missing",
                flush=True,
            )
        except Exception as e:
            print(f"❌ Unexpected error during installation: {e}", flush=True)
            print(
                "Proceeding anyway - agent may fail if dependencies are missing",
                flush=True,
            )

        print("=" * 60, flush=True)

    @staticmethod
    def prefetch_models_if_configured(submission_dir: Path) -> None:
        """Pre-download large model assets referenced in submission config if possible.

        This avoids long, opaque downloads within Ray actors on first use.
        """
        try:
            cfg = AgentLoader.load_config(submission_dir)
            model_path = cfg.get("model_path") if isinstance(cfg, dict) else None
            if not model_path:
                return

            # Resolve local paths relative to the submission directory
            path_str = str(model_path)
            potential_local = Path(path_str)
            if not potential_local.is_absolute():
                potential_local = (submission_dir / potential_local).resolve()

            # If model_path is a local directory or file, nothing to prefetch
            if potential_local.exists():
                return

            # If the string looks like a filesystem path (contains os.sep, starts with '.' or '/')
            # then skip trying to treat it as an HF repo id.
            if (
                path_str.startswith(".")
                or path_str.startswith("/")
                or os.sep in path_str
            ):
                return

            # Prefer hf-transfer for speed, if available
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

            print(f"📥 Prefetching model assets for {model_path}...", flush=True)
            try:
                from huggingface_hub import snapshot_download  # noqa: F401, PLC0415
            except Exception as imp_exc:  # pragma: no cover
                print(
                    f"⚠️  huggingface_hub not available yet ({imp_exc}); skipping prefetch",
                    flush=True,
                )
                return

            try:
                snapshot_download(
                    repo_id=str(model_path),
                    resume_download=True,
                )
                print("✅ Model assets cached in local HF cache", flush=True)
            except Exception as dl_exc:  # pragma: no cover
                print(f"⚠️  Prefetch failed: {dl_exc}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"⚠️  Prefetch step skipped due to error: {e}", flush=True)

    @staticmethod
    def load_config(submission_path: str | Path) -> Dict[str, Any]:
        """
        Load optional configuration from submission directory.

        Args:
            submission_path: Path to submission directory

        Returns:
            Dict containing configuration, empty if config.json doesn't exist
        """
        config_file = Path(submission_path) / "config.json"
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config.json: {e}")
        return {}

    @staticmethod
    def validate_submission(submission_path: str | Path) -> bool:
        """
        Validate that a submission has the minimum required structure.

        Args:
            submission_path: Path to submission directory

        Returns:
            True if submission is valid, False otherwise
        """
        try:
            submission_dir = Path(submission_path)

            # Check basic structure
            if not submission_dir.exists() or not submission_dir.is_dir():
                return False

            # Check for required agent.py
            if not (submission_dir / "agent.py").exists():
                return False

            return True

        except Exception:
            return False
