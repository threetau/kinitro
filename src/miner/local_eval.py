from __future__ import annotations

import json
import math
import socket
import subprocess
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import ray
from ray.util.queue import Queue
from snowflake import SnowflakeGenerator

from core.errors import ConfigurationError, LocalEvaluationError
from core.log import get_logger
from evaluator.rollout import BenchmarkSpec, RolloutCluster
from evaluator.rollout.envs import EnvResult
from evaluator.rpc.rpc_process import RPCProcess

logger = get_logger(__name__)

QUEUE_CAPACITY = 32
DEFAULT_CLUSTER_NAME = "miner-local"


def _launch_agent_process(command: str) -> subprocess.Popen[str]:
    """Launch the agent server if a start command is provided."""
    logger.info("Launching agent process with command: %s", command)
    try:
        process = subprocess.Popen(  # noqa: S603,S607 - user-provided command
            command,
            shell=True,
            text=True,
        )
    except Exception as exc:
        raise LocalEvaluationError(
            f"Failed to launch agent process with command '{command}': {exc}"
        ) from exc
    logger.info("Agent process started with PID %s", process.pid)
    return process


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    """Terminate a subprocess gracefully."""
    if process.poll() is not None:
        return
    logger.info("Stopping agent process PID %s", process.pid)
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Agent process did not exit gracefully, killing (PID %s)", process.pid
            )
            process.kill()
    except Exception as exc:  # pragma: no cover - defensive cleanup
        logger.warning("Failed to terminate agent process PID %s: %s", process.pid, exc)


def _wait_for_port(host: str, port: int, timeout: float) -> None:
    """Wait until the agent server accepts TCP connections."""
    logger.info("Waiting for agent RPC server at %s:%s", host, port)
    deadline = time.time() + timeout
    last_error: Optional[Exception] = None

    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            try:
                sock.connect((host, port))
                logger.info("Agent RPC server is reachable at %s:%s", host, port)
                return
            except OSError as exc:  # pragma: no cover - timing dependent
                last_error = exc
        time.sleep(0.5)

    raise LocalEvaluationError(
        f"Timed out after {timeout:.1f}s waiting for agent RPC server at {host}:{port}"
    ) from last_error


def _ensure_ray_runtime(ray_cpus: float, ray_gpus: float) -> None:
    """Initialize Ray if needed with the requested resources."""
    if ray.is_initialized():
        logger.info("Reusing existing Ray runtime")
        return

    init_kwargs: dict[str, Any] = {
        "include_dashboard": False,
        "ignore_reinit_error": True,
        "log_to_driver": True,
    }
    if ray_cpus is not None:
        init_kwargs["num_cpus"] = max(1, math.ceil(float(ray_cpus)))
    if ray_gpus is not None:
        init_kwargs["num_gpus"] = max(0, float(ray_gpus))

    logger.info(
        "Initializing Ray locally with resources: cpus=%s gpus=%s",
        init_kwargs.get("num_cpus"),
        init_kwargs.get("num_gpus"),
    )
    ray.init(**init_kwargs)


def _load_benchmark_entries_from_file(path: Path) -> list[dict[str, Any]]:
    """Load benchmark definitions from a JSON or TOML file."""
    if not path.exists():
        raise ConfigurationError(f"Benchmark spec file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    try:
        if suffix in {".json"}:
            data: Any = json.loads(text)
        elif suffix in {".toml"}:
            if tomllib is None:
                raise ConfigurationError(
                    "tomllib is unavailable; install Python 3.11+ to use TOML specs"
                )
            data = tomllib.loads(text)
        else:
            raise ConfigurationError(
                f"Unsupported benchmark spec extension '{suffix}'. Use .json or .toml."
            )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ConfigurationError(f"Failed to parse benchmark spec file: {exc}") from exc

    if isinstance(data, dict) and "benchmarks" in data:
        data = data["benchmarks"]

    if not isinstance(data, list):
        raise ConfigurationError(
            "Benchmark spec file must contain a list of benchmark objects"
        )

    return [entry for entry in data if entry is not None]


def _apply_config_defaults(
    config: dict[str, Any],
    _settings,
) -> dict[str, Any]:
    """Return config unchanged; CLI overrides removed for local eval."""
    return dict(config)


def _to_string_tuple(value: Any) -> tuple[str, ...]:
    """Normalise a string or iterable into a tuple of strings."""
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError as exc:
        raise ConfigurationError(
            f"camera_names must be a string or iterable of strings, got: {value!r}"
        ) from exc


def _build_benchmark_spec_from_entry(
    entry: dict[str, Any],
    _settings,
) -> BenchmarkSpec:
    """Convert a dict entry into a BenchmarkSpec."""
    provider = entry.get("provider")
    if not provider:
        raise ConfigurationError(
            "Benchmark provider missing; supply it in the spec file or via --benchmark-provider"
        )

    name = entry.get("benchmark_name")
    if not name:
        raise ConfigurationError(
            "Benchmark name missing; supply it in the spec file or via --benchmark-name"
        )

    base_config = entry.get("config") or {}
    if not isinstance(base_config, dict):
        raise ConfigurationError("Benchmark config must be an object/dict")

    merged_config = _apply_config_defaults(base_config, None)

    spec_kwargs: dict[str, Any] = {}
    if "render_mode" in entry:
        spec_kwargs["render_mode"] = entry.get("render_mode")

    if "camera_names" in entry:
        spec_kwargs["camera_names"] = _to_string_tuple(entry["camera_names"])

    if "camera_attribute" in entry:
        spec_kwargs["camera_attribute"] = entry.get("camera_attribute")

    return BenchmarkSpec(
        provider=str(provider),
        benchmark_name=str(name),
        config=merged_config,
        **spec_kwargs,
    )


def _resolve_benchmark_specs(settings) -> list[BenchmarkSpec]:
    """Resolve benchmark specs from a file"""
    spec_file = settings.get("benchmark_spec_file")
    specs: list[BenchmarkSpec] = []

    if spec_file:
        path = Path(spec_file).expanduser().resolve()
        entries = _load_benchmark_entries_from_file(path)
        specs = [_build_benchmark_spec_from_entry(entry, settings) for entry in entries]
    else:
        raise ConfigurationError(
            "--benchmark-spec-file (or [local_eval].benchmark_spec_file) is required for local-eval"
        )

    if not specs:
        raise ConfigurationError("No benchmark specifications were resolved")

    return specs


def _summarize_env_results(env_results: Iterable[EnvResult]) -> dict[str, Any]:
    """Calculate aggregate metrics for a completed run."""
    env_results = list(env_results)
    if not env_results:
        return {
            "total_envs": 0,
            "total_episodes": 0,
            "mean_success_rate": 0.0,
            "mean_reward": 0.0,
            "mean_steps": 0.0,
        }

    total_envs = len(env_results)
    total_episodes = sum(len(result.episodes) for result in env_results)
    mean_success_rate = sum(result.success_rate for result in env_results) / total_envs
    mean_reward = sum(result.mean_reward for result in env_results) / total_envs
    mean_steps = sum(result.mean_steps for result in env_results) / total_envs

    return {
        "total_envs": total_envs,
        "total_episodes": total_episodes,
        "mean_success_rate": mean_success_rate,
        "mean_reward": mean_reward,
        "mean_steps": mean_steps,
    }


def _serialize_env_results(env_results: Iterable[EnvResult]) -> list[dict[str, Any]]:
    """Convert EnvResult objects into JSON-serialisable dictionaries."""
    serialised: list[dict[str, Any]] = []
    for result in env_results:
        env_spec = result.env_spec
        serialised.append(
            {
                "environment": {
                    "provider": env_spec.provider,
                    "benchmark": env_spec.benchmark_name,
                    "env_name": env_spec.env_name,
                    "episodes_per_task": env_spec.episodes_per_task,
                    "max_episode_steps": env_spec.max_episode_steps,
                },
                "metrics": {
                    "success_rate": result.success_rate,
                    "mean_reward": result.mean_reward,
                    "mean_steps": result.mean_steps,
                    "episodes": len(result.episodes),
                },
            }
        )
    return serialised


def _persist_summary(
    results_dir: Path,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    agent_host: str,
    agent_port: int,
    benchmark_specs: list[BenchmarkSpec],
    aggregate_metrics: dict[str, Any],
    env_results: Iterable[EnvResult],
) -> Path:
    """Write a JSON summary for the run."""
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "agent": {"host": agent_host, "port": agent_port},
        "benchmarks": [
            {
                "provider": spec.provider,
                "name": spec.benchmark_name,
                "config": spec.config,
                "render_mode": spec.render_mode,
                "camera_names": list(spec.camera_names) if spec.camera_names else None,
                "camera_attribute": spec.camera_attribute,
            }
            for spec in benchmark_specs
        ],
        "aggregate": aggregate_metrics,
        "environments": _serialize_env_results(env_results),
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def handle_local_eval_command(config) -> None:
    """Entry point for the miner local evaluation command."""
    settings = config.settings
    agent_host = settings.get("agent_host", "127.0.0.1")
    agent_port = int(settings.get("agent_port", 8000))
    agent_start_cmd = settings.get("agent_start_cmd")
    agent_start_timeout = float(settings.get("agent_start_timeout", 30))
    results_dir = Path(settings.get("local_results_dir", ".kinitro/miner_runs"))
    ray_num_cpus = settings.get("ray_num_cpus", 2)
    ray_num_gpus = settings.get("ray_num_gpus", 0)

    logger.info("Preparing local evaluation harness")

    benchmark_specs = _resolve_benchmark_specs(settings)
    id_generator = SnowflakeGenerator(42)
    rollout_worker_id = next(id_generator)
    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    logger.info(
        "Run ID: %s | Agent: %s:%s | Benchmarks: %d",
        run_id,
        agent_host,
        agent_port,
        len(benchmark_specs),
    )
    for spec in benchmark_specs:
        logger.info(
            "Benchmark spec loaded: %s/%s config=%s",
            spec.provider,
            spec.benchmark_name,
            spec.config,
        )

    agent_process: Optional[subprocess.Popen[Any]] = None
    rpc_thread: Optional[threading.Thread] = None
    worker_to_rpc_queue: Optional[Queue] = None
    rpc_to_worker_queue: Optional[Queue] = None
    cluster: Optional[RolloutCluster] = None

    started_at = datetime.now(timezone.utc)
    try:
        if agent_start_cmd:
            agent_process = _launch_agent_process(agent_start_cmd)

        _wait_for_port(agent_host, agent_port, timeout=agent_start_timeout)
        _ensure_ray_runtime(ray_num_cpus, ray_num_gpus)

        # Queues for RPC interaction
        worker_to_rpc_queue = Queue(maxsize=QUEUE_CAPACITY)
        rpc_to_worker_queue = Queue(maxsize=QUEUE_CAPACITY)

        remote_options: dict[str, Any] = {
            "max_restarts": 0,
            "max_task_retries": 0,
        }
        if ray_num_cpus is not None:
            remote_options["num_cpus"] = float(ray_num_cpus)
        if ray_num_gpus:
            remote_options["num_gpus"] = float(ray_num_gpus)

        cluster = RolloutCluster(
            DEFAULT_CLUSTER_NAME,
            worker_remote_options=remote_options,
        )

        worker = cluster.create_worker(
            rollout_worker_id=rollout_worker_id,
            benchmark_specs=benchmark_specs,
            submission_container_host=agent_host,
            submission_container_port=agent_port,
            submission_id=rollout_worker_id,
            s3_config=None,
            episode_log_interval=int(settings.get("episode_log_interval", 1)),
            step_log_interval=int(settings.get("step_log_interval", 1)),
            database_url=None,
        )

        rpc_thread = threading.Thread(
            target=RPCProcess,
            args=(agent_host, agent_port, rpc_to_worker_queue, worker_to_rpc_queue),
            name="miner-local-rpc",
            daemon=True,
        )
        rpc_thread.start()
        time.sleep(0.5)

        logger.info("Testing RPC connectivity with the agent")
        rpc_test = ray.get(
            worker.test_rpc.remote(worker_to_rpc_queue, rpc_to_worker_queue)
        )

        if getattr(rpc_test, "success", False) is False:
            error_detail = getattr(rpc_test, "error_message", "unknown error")
            raise LocalEvaluationError(f"Agent RPC ping failed: {error_detail}")

        logger.info("Starting benchmark execution")
        env_results: list[EnvResult] = ray.get(
            worker.run_all_benchmark_tasks.remote(
                worker_to_rpc_queue,
                rpc_to_worker_queue,
            )
        )

        finished_at = datetime.now(timezone.utc)
        aggregate_metrics = _summarize_env_results(env_results)
        summary_path = _persist_summary(
            results_dir,
            run_id,
            started_at,
            finished_at,
            agent_host,
            agent_port,
            benchmark_specs,
            aggregate_metrics,
            env_results,
        )

        logger.info(
            "Local evaluation complete: %s environments, %.3f mean success rate",
            aggregate_metrics["total_envs"],
            aggregate_metrics["mean_success_rate"],
        )
        logger.info("Summary written to %s", summary_path)

    except LocalEvaluationError:
        raise
    except ConfigurationError:
        raise
    except Exception as exc:
        raise LocalEvaluationError(f"Local evaluation failed: {exc}") from exc
    finally:
        finished_at = datetime.now(timezone.utc)
        if cluster:
            logger.debug("Cleaning up rollout cluster")
            try:
                cluster.cleanup_all_workers()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to clean up rollout cluster: %s", exc)

        queues: tuple[Optional[Queue], ...] = (
            worker_to_rpc_queue,
            rpc_to_worker_queue,
        )
        for queue in queues:
            if queue is not None:
                try:
                    queue.shutdown(force=True)
                except Exception as exc:  # pragma: no cover - defensive cleanup
                    logger.debug("Queue shutdown failed: %s", exc)

        if rpc_thread:
            rpc_thread.join(timeout=1)

        if agent_process:
            _terminate_process(agent_process)

        if ray.is_initialized():
            try:
                ray.shutdown()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to tear down Ray runtime: %s", exc)

        logger.info(
            "Local evaluation teardown completed (duration %.2fs)",
            (finished_at - started_at).total_seconds(),
        )
