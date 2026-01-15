"""
Job scheduler for Kinitro backend.

Creates and broadcasts evaluation jobs to validators.
Extracted from BackendService for better separation of concerns.
"""

import copy
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.db.models import EvaluationStatus
from core.log import get_logger
from core.messages import EvalJobMessage

from .constants import EVAL_JOB_TIMEOUT
from .events import JobCreatedEvent
from .models import (
    BackendEvaluationJob,
    Competition,
    MinerSubmission,
)
from .realtime import EventType, event_broadcaster

if TYPE_CHECKING:
    from .websocket_hub import WebSocketHub

logger = get_logger(__name__)


class JobConfig:
    """Configuration for job scheduling."""

    def __init__(
        self,
        default_job_timeout_seconds: int = int(EVAL_JOB_TIMEOUT.total_seconds()),
        submission_download_url_ttl: int = 21600,  # 6 hours
    ):
        self.default_job_timeout_seconds = default_job_timeout_seconds
        self.submission_download_url_ttl = submission_download_url_ttl


def _extract_benchmark_spec_payload(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split a stored benchmark configuration into the full benchmark spec and the
    underlying execution config used by the evaluator.
    """
    spec_copy = copy.deepcopy(dict(config))
    try:
        base_config_source = config["config"]
    except KeyError as exc:
        raise ValueError("Benchmark spec is missing 'config' payload") from exc
    try:
        base_config = copy.deepcopy(dict(base_config_source))
    except TypeError as exc:
        raise ValueError("'config' payload must be a mapping") from exc
    return spec_copy, base_config


def _normalize_benchmark_spec_payload(
    provider: str,
    benchmark_name: str,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Ensure a benchmark specification payload includes top-level metadata and a nested config mapping.

    Accepts either the new-style payload (with a `config` key) or a bare config mapping and
    returns a copy that always matches the new-style structure.
    """
    if payload is None:
        base_config: dict[str, Any] = {}
        return {
            "provider": provider,
            "benchmark_name": benchmark_name,
            "config": base_config,
        }

    payload_dict = dict(payload)
    if "config" in payload_dict:
        return copy.deepcopy(payload_dict)

    base_config = copy.deepcopy(payload_dict)
    return {
        "provider": provider,
        "benchmark_name": benchmark_name,
        "config": base_config,
    }


class SubmissionNotFoundError(Exception):
    """Raised when a submission cannot be located."""


class EvaluationJobNotFoundError(Exception):
    """Raised when an evaluation job cannot be located."""


class NoBenchmarksAvailableError(Exception):
    """Raised when no benchmarks are available for an evaluation rerun."""


class JobScheduler:
    """
    Creates and broadcasts evaluation jobs.

    This class is responsible for:
    - Creating evaluation jobs for submissions
    - Broadcasting jobs to connected validators
    - Handling job reruns
    - Monitoring for stale jobs
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        ws_hub: "WebSocketHub",
        config: JobConfig,
        id_generator,
        submission_storage=None,
    ):
        self.session_factory = session_factory
        self.ws_hub = ws_hub
        self.config = config
        self.id_generator = id_generator
        self.submission_storage = submission_storage

    def _job_timeout_seconds(self, competition: Optional[Competition]) -> int:
        """Return the timeout for a competition, falling back to the default."""
        if competition and competition.job_timeout_seconds:
            try:
                value = int(competition.job_timeout_seconds)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid job_timeout_seconds for competition %s: %r",
                    getattr(competition, "id", "unknown"),
                    getattr(competition, "job_timeout_seconds", None),
                )
        return self.config.default_job_timeout_seconds

    async def create_jobs_for_submission(
        self,
        submission: MinerSubmission,
        competition: Competition,
    ) -> List[BackendEvaluationJob]:
        """Create evaluation jobs for a submission based on competition benchmarks."""
        jobs: List[BackendEvaluationJob] = []

        for benchmark in competition.benchmarks:
            if "provider" not in benchmark or "benchmark_name" not in benchmark:
                logger.error(
                    "Benchmark missing provider or benchmark_name: %s", benchmark
                )
                continue

            if isinstance(benchmark, dict):
                benchmark_spec = copy.deepcopy(benchmark)
            else:
                logger.warning(
                    "Benchmark specification for competition %s is not a dict (%r); "
                    "wrapping in config field",
                    competition.id,
                    type(benchmark),
                )
                benchmark_spec = {"config": benchmark}

            job = BackendEvaluationJob(
                id=next(self.id_generator),
                submission_id=submission.id,
                competition_id=competition.id,
                miner_hotkey=submission.miner_hotkey,
                hf_repo_id=submission.hf_repo_id,
                env_provider=benchmark_spec.get("provider", benchmark["provider"]),
                benchmark_name=benchmark_spec.get(
                    "benchmark_name", benchmark["benchmark_name"]
                ),
                config=benchmark_spec,
                timeout_seconds=self._job_timeout_seconds(competition),
                artifact_object_key=submission.artifact_object_key,
                artifact_sha256=submission.artifact_sha256,
                artifact_size_bytes=submission.artifact_size_bytes,
            )
            jobs.append(job)

        return jobs

    async def schedule_submission(
        self,
        submission: MinerSubmission,
        competition: Competition,
        session: AsyncSession,
    ) -> List[BackendEvaluationJob]:
        """Create and persist evaluation jobs for a submission, then broadcast them."""
        jobs = await self.create_jobs_for_submission(submission, competition)

        if not jobs:
            logger.error(
                "No evaluation jobs generated for submission %s", submission.id
            )
            return []

        session.add_all(jobs)
        await session.flush()

        return jobs

    async def publish_jobs(self, jobs: Sequence[BackendEvaluationJob]) -> None:
        """Emit events and broadcasts for newly created jobs."""
        if not jobs:
            return

        connected_validator_hotkeys = tuple(
            dict.fromkeys(self.ws_hub.get_validator_hotkeys())
        )

        for job in jobs:
            _benchmark_spec_payload, base_config_payload = (
                _extract_benchmark_spec_payload(job.config)
            )
            job_event = JobCreatedEvent(
                job_id=str(job.id),
                competition_id=job.competition_id,
                submission_id=job.submission_id,
                miner_hotkey=job.miner_hotkey,
                hf_repo_id=job.hf_repo_id,
                env_provider=job.env_provider,
                benchmark_name=job.benchmark_name,
                config=base_config_payload,
                status=EvaluationStatus.QUEUED,
                validator_statuses={
                    hotkey: EvaluationStatus.QUEUED
                    for hotkey in connected_validator_hotkeys
                },
            )

            try:
                await event_broadcaster.broadcast_event(
                    EventType.JOB_CREATED, job_event
                )
            except Exception as exc:
                logger.error(
                    "Failed to broadcast job created event for job %s: %s",
                    job.id,
                    exc,
                )

            try:
                await self.broadcast_job(job)
            except Exception as exc:
                logger.error("Failed to push job %s to validators: %s", job.id, exc)

    async def broadcast_job(self, job: BackendEvaluationJob) -> int:
        """Broadcast job to connected validators.

        Returns:
            Number of validators that received the job
        """
        if not self.ws_hub.has_connections():
            logger.warning("No validators connected")
            return 0

        artifact_url = None
        artifact_expires_at: Optional[datetime] = None
        if self.submission_storage and job.artifact_object_key:
            try:
                artifact_url, artifact_expires_at = (
                    self.submission_storage.generate_download_url(
                        job.artifact_object_key, self.config.submission_download_url_ttl
                    )
                )
            except Exception as exc:
                logger.error(
                    "Failed to generate artifact URL for job %s: %s", job.id, exc
                )
        else:
            logger.error(
                "Cannot broadcast submission %s: storage unavailable or artifact missing",
                job.submission_id,
            )
            return 0

        benchmark_spec_payload, base_config_payload = _extract_benchmark_spec_payload(
            job.config
        )

        timeout_seconds = job.timeout_seconds or self.config.default_job_timeout_seconds

        job_msg = EvalJobMessage(
            job_id=job.id,
            competition_id=job.competition_id,
            submission_id=job.submission_id,
            miner_hotkey=job.miner_hotkey,
            hf_repo_id=job.hf_repo_id,
            env_provider=job.env_provider,
            benchmark_name=job.benchmark_name,
            config=base_config_payload,
            benchmark_spec=benchmark_spec_payload,
            artifact_url=artifact_url,
            artifact_expires_at=artifact_expires_at,
            artifact_sha256=job.artifact_sha256,
            artifact_size_bytes=job.artifact_size_bytes,
            timeout=timedelta(seconds=timeout_seconds),
        )

        message = job_msg.model_dump_json()
        broadcast_count = await self.ws_hub.broadcast_message(message)

        logger.info(f"Broadcasted job {job.id} to {broadcast_count} validators")
        return broadcast_count

    async def rerun_submission_evaluations(
        self,
        submission_id: int,
        benchmark_names: Optional[List[str]] = None,
        requested_by_api_key_id: Optional[int] = None,
    ) -> List[BackendEvaluationJob]:
        """Re-run evaluations for a submission across its configured benchmarks."""
        benchmark_filter = (
            {name.strip() for name in benchmark_names if name.strip()}
            if benchmark_names
            else None
        )

        new_jobs: List[BackendEvaluationJob] = []

        async with self.session_factory() as session:
            submission = await session.get(MinerSubmission, submission_id)
            if not submission:
                raise SubmissionNotFoundError(f"Submission {submission_id} not found")

            competition = await session.get(Competition, submission.competition_id)
            if not competition:
                raise SubmissionNotFoundError(
                    f"Competition {submission.competition_id} not found for submission {submission_id}"
                )

            benchmarks = competition.benchmarks or []
            for benchmark in benchmarks:
                provider = benchmark.get("provider")
                benchmark_name = benchmark.get("benchmark_name")

                if not provider or not benchmark_name:
                    logger.error(
                        "Submission %s rerun skipped invalid benchmark entry: %s",
                        submission_id,
                        benchmark,
                    )
                    continue

                if benchmark_filter and benchmark_name not in benchmark_filter:
                    continue

                spec_payload = _normalize_benchmark_spec_payload(
                    provider,
                    benchmark_name,
                    benchmark,
                )

                job = BackendEvaluationJob(
                    id=next(self.id_generator),
                    submission_id=submission.id,
                    competition_id=competition.id,
                    miner_hotkey=submission.miner_hotkey,
                    hf_repo_id=submission.hf_repo_id,
                    env_provider=provider,
                    benchmark_name=benchmark_name,
                    config=spec_payload,
                    timeout_seconds=self._job_timeout_seconds(competition),
                    artifact_object_key=submission.artifact_object_key,
                    artifact_sha256=submission.artifact_sha256,
                    artifact_size_bytes=submission.artifact_size_bytes,
                )
                new_jobs.append(job)

            if not new_jobs:
                raise NoBenchmarksAvailableError(
                    "No matching benchmarks available for rerun request"
                )

            session.add_all(new_jobs)
            await session.commit()

            for job in new_jobs:
                await session.refresh(job)

        await self.publish_jobs(new_jobs)

        logger.info(
            "Submission %s rerun triggered by API key %s; queued %s jobs",
            submission_id,
            requested_by_api_key_id,
            len(new_jobs),
        )

        return new_jobs

    async def rerun_job_evaluation(
        self,
        job_id: int,
        requested_by_api_key_id: Optional[int] = None,
    ) -> BackendEvaluationJob:
        """Re-run a specific evaluation job by cloning its configuration."""
        async with self.session_factory() as session:
            existing_job = await session.get(BackendEvaluationJob, job_id)
            if not existing_job:
                raise EvaluationJobNotFoundError(f"Job {job_id} not found")

            spec_payload = _normalize_benchmark_spec_payload(
                existing_job.env_provider,
                existing_job.benchmark_name,
                existing_job.config
                if isinstance(existing_job.config, Mapping)
                else None,
            )

            new_job = BackendEvaluationJob(
                id=next(self.id_generator),
                submission_id=existing_job.submission_id,
                competition_id=existing_job.competition_id,
                miner_hotkey=existing_job.miner_hotkey,
                hf_repo_id=existing_job.hf_repo_id,
                env_provider=existing_job.env_provider,
                benchmark_name=existing_job.benchmark_name,
                config=spec_payload,
                timeout_seconds=existing_job.timeout_seconds,
                artifact_object_key=existing_job.artifact_object_key,
                artifact_sha256=existing_job.artifact_sha256,
                artifact_size_bytes=existing_job.artifact_size_bytes,
            )

            session.add(new_job)
            await session.commit()
            await session.refresh(new_job)

        await self.publish_jobs([new_job])

        logger.info(
            "Job %s rerun created as job %s by API key %s",
            job_id,
            new_job.id,
            requested_by_api_key_id,
        )

        return new_job
