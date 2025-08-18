"""Hugging Face submission system"""

import os
from pathlib import Path
from typing import Optional

import dotenv
from huggingface_hub import HfApi

from core.errors import UploadError
from core.log import get_logger

logger = get_logger(__name__)

dotenv.load_dotenv()


def upload_submission_to_hf(
    submission_dir: str, repo_id: str, token: str, commit_message: Optional[str] = None
) -> None:
    """
    Upload submission directory to Hugging Face repository.

    Args:
        submission_dir: Path to the submission directory
        repo_id: Hugging Face repository ID (e.g., "username/repo-name")
        token: Hugging Face API token
        commit_message: Optional commit message

    Raises:
        UploadError: If upload fails
        FileNotFoundError: If submission directory doesn't exist
    """
    logger.info(f"Uploading submission from {submission_dir} to {repo_id}")

    # Validate submission directory exists
    submission_path = Path(submission_dir)
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission directory not found: {submission_dir}")

    # Initialize Hugging Face API
    api = HfApi(token=token)

    try:
        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            logger.info(f"Repository {repo_id} ready")
        except Exception as e:
            logger.warning(f"Repository creation failed or already exists: {e}")

        # Upload the entire submission directory
        commit_info = api.upload_folder(
            folder_path=str(submission_path),
            repo_id=repo_id,
            commit_message=commit_message
            or f"Upload submission from {submission_path.name}",
            token=token,
        )

        logger.info(f"Successfully uploaded submission. Commit SHA: {commit_info.oid}")

    except Exception as e:
        raise UploadError(f"Failed to upload submission: {e}") from e


def download_submission_from_hf(repo_id: str, submission_dir: str) -> None:
    """
    Download submission from Hugging Face repository.
    """
    token = os.getenv("HF_TOKEN")

    logger.info(f"Downloading submission from {repo_id}")

    # Initialize Hugging Face API
    api = HfApi(token=token)

    # Download the entire submission directory
    download_path = api.snapshot_download(
        repo_id=repo_id, local_dir=submission_dir, token=token
    )

    logger.info(f"Successfully downloaded submission to {download_path}")
