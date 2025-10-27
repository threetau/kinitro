import hashlib
import os
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import dotenv
import requests
from fiber.chain.chain_utils import load_hotkey_keypair

from core.chain import commit_to_substrate
from core.errors import (
    CommitmentError,
    ConfigurationError,
    LocalEvaluationError,
    UploadError,
)
from core.log import get_logger
from core.schemas import ChainCommitment, ModelProvider

from .config import MinerConfig
from .local_eval import handle_local_eval_command

logger = get_logger(__name__)
dotenv.load_dotenv()


def _ensure_directory(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ConfigurationError(f"Submission directory does not exist: {path}")
    return path


def _package_submission(submission_dir: Path) -> tuple[Path, int, str]:
    with tempfile.TemporaryDirectory(prefix="kinitro-submission-") as tmp_dir:
        tar_path = Path(tmp_dir) / "submission.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(str(submission_dir), arcname=".")

        sha256 = hashlib.sha256()
        size = 0
        with tar_path.open("rb") as archive:
            for chunk in iter(lambda: archive.read(1024 * 1024), b""):
                size += len(chunk)
                sha256.update(chunk)

        final_path = submission_dir / "submission.tar.gz"
        if final_path.exists():
            final_path.unlink()
        tar_path.replace(final_path)
        return final_path, size, sha256.hexdigest()


def _build_signature_message(
    hotkey: str,
    competition_id: str,
    version: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    timestamp: int,
) -> bytes:
    parts = [
        hotkey,
        competition_id,
        version,
        artifact_sha256,
        str(artifact_size_bytes),
        str(timestamp),
    ]
    return "|".join(parts).encode("utf-8")


def _post_json(url: str, payload: dict) -> dict:
    try:
        response = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:  # pragma: no cover - network interaction
        raise UploadError(f"Failed to contact backend: {exc}") from exc
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:  # pragma: no cover - network interaction
        raise UploadError(f"Upload request failed: {exc} - {response.text}") from exc
    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - unexpected response
        raise UploadError("Backend returned invalid JSON") from exc


def _upload_artifact(upload_url: str, tar_path: Path, headers: Optional[dict]) -> None:
    request_headers = headers or {}
    with tar_path.open("rb") as archive:
        try:
            response = requests.put(
                upload_url,
                data=archive,
                headers=request_headers,
                timeout=300,
            )
        except (
            requests.RequestException
        ) as exc:  # pragma: no cover - network interaction
            raise UploadError(f"Failed to upload artifact: {exc}") from exc
    if response.status_code // 100 != 2:
        raise UploadError(
            f"Artifact upload failed with status {response.status_code}: {response.text}"
        )


def handle_upload_command(config: MinerConfig) -> None:
    submission_dir = _ensure_directory(config.settings["submission_dir"])
    backend_url = config.settings.get("backend_url") or os.getenv("BACKEND_URL")
    if not backend_url:
        raise ConfigurationError(
            "Backend URL is required. Provide --backend-url or set backend_url in configuration."
        )

    competition_id = config.settings.get("competition_id")
    if not competition_id:
        raise ConfigurationError("competition_id is required in configuration")

    version = config.settings.get("submission_version")
    if not version:
        version = f"v-{int(time.time())}"

    logger.info("Packaging submission from %s", submission_dir)
    artifact_path, artifact_size_bytes, artifact_sha256 = _package_submission(
        submission_dir
    )
    logger.info(
        "Created archive %s (size=%s bytes)", artifact_path, artifact_size_bytes
    )

    keypair = load_hotkey_keypair(
        wallet_name=config.settings["wallet_name"],
        hotkey_name=config.settings["hotkey_name"],
    )
    miner_hotkey = keypair.ss58_address

    timestamp = int(datetime.now(timezone.utc).timestamp())
    message = _build_signature_message(
        miner_hotkey,
        competition_id,
        version,
        artifact_sha256,
        artifact_size_bytes,
        timestamp,
    )
    signature_hex = "0x" + keypair.sign(message).hex()

    payload = {
        "competition_id": competition_id,
        "version": version,
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_size_bytes,
        "timestamp": timestamp,
        "hotkey": miner_hotkey,
        "signature": signature_hex,
    }

    try:
        logger.info("Requesting presigned upload URL from backend...")
        request_url = backend_url.rstrip("/") + "/submissions/request-upload"
        response_data = _post_json(request_url, payload)

        upload_url = response_data.get("upload_url")
        if not upload_url:
            raise UploadError("Backend response missing upload URL")

        logger.info("Upload url: %s", upload_url)
        logger.info("Uploading artifact to vault...")
        _upload_artifact(upload_url, artifact_path, response_data.get("headers"))

        commit_payload = response_data.get("commit_payload", {})
        submission_id = commit_payload.get(
            "submission_id", response_data.get("submission_id")
        )
        if submission_id is None:
            logger.warning("Backend did not return submission ID in commit payload")

        logger.info("Upload completed successfully!")
        logger.info("Submission ID: %s", submission_id)
        logger.info("Artifact SHA256: %s", artifact_sha256)
        logger.info("Artifact size bytes: %s", artifact_size_bytes)
        logger.info(
            "To commit this submission, run:\n  uv run python -m miner commit --submission-id %s",
            submission_id,
        )
    finally:
        try:
            artifact_path.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover - filesystem failure
            logger.debug(
                "Failed to remove temporary archive %s: %s", artifact_path, exc
            )


def handle_commit_command(config: MinerConfig) -> None:
    competition_id = config.settings.get("competition_id")
    if not competition_id:
        raise ConfigurationError("competition_id is required in configuration")

    submission_id = config.settings.get("submission_id")
    if submission_id is None or str(submission_id).strip() == "":
        raise ConfigurationError(
            "Submission ID is required. Provide --submission-id with the value returned from the upload step."
        )
    submission_id = str(submission_id).strip()

    commit_data = ChainCommitment(
        provider=ModelProvider.S3,
        repo_id=str(submission_id),
        comp_id=competition_id,
    )

    commit_to_substrate(config, commit_data)

    logger.info("Substrate commitment completed successfully!")
    logger.info(
        "Committed submission ID %s for competition %s", submission_id, competition_id
    )


def main():
    """Main entry point for the miner CLI."""
    try:
        config = MinerConfig()
        command = config.settings.command

        if command == "upload":
            handle_upload_command(config)
        elif command == "commit":
            handle_commit_command(config)
        elif command == "local-eval":
            handle_local_eval_command(config)
        else:
            raise ConfigurationError(f"Unknown command: {command}")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        raise SystemExit(1) from e
    except UploadError as e:
        logger.error(f"Upload failed: {e}")
        raise SystemExit(1) from e
    except CommitmentError as e:
        logger.error(f"Substrate commitment failed: {e}")
        raise SystemExit(1) from e
    except LocalEvaluationError as e:
        logger.error(f"Local evaluation failed: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
