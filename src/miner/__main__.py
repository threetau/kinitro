import json
import os
from pathlib import Path
from typing import Optional

import dotenv
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.commitments import CommitmentDataFieldType, set_commitment
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from huggingface_hub import HfApi

from core.log import get_logger
from miner.config import MinerConfig

logger = get_logger(__name__)
dotenv.load_dotenv()


class MinerError(Exception):
    """Base exception for miner operations."""

    pass


class ConfigurationError(MinerError):
    """Raised when configuration is invalid."""

    pass


class UploadError(MinerError):
    """Raised when upload operations fail."""

    pass


class CommitmentError(MinerError):
    """Raised when substrate commitment operations fail."""

    pass


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


def commit_to_substrate(config: MinerConfig, commit_data: dict) -> None:
    """
    Commit information to substrate chain using fiber.

    Args:
        config: Miner configuration
        commit_data: Data to commit to the chain

    Raises:
        CommitmentError: If substrate commitment fails
    """
    logger.info("Connecting to substrate chain...")

    try:
        # Get substrate connection
        substrate = get_substrate(
            subtensor_network=config.settings.subtensor_network,
            subtensor_address=config.settings.subtensor_address,
        )

        logger.info("Successfully connected to substrate")

        # Create metagraph to get neuron information (unused but available for future use)
        Metagraph(
            netuid=config.settings.netuid,
            substrate=substrate,
        )

        # Prepare commitment data
        commitment_data = json.dumps(commit_data)

        keypair = load_hotkey_keypair(
            wallet_name=config.settings.wallet_name,
            hotkey_name=config.settings.hotkey_name,
        )

        # Set commitment on chain
        logger.info("Setting commitment on chain...")
        success = set_commitment(
            substrate=substrate,
            keypair=keypair,
            netuid=config.settings.netuid,
            fields=[
                (CommitmentDataFieldType.RAW, commitment_data.encode("utf-8")),
            ],
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        if not success:
            raise CommitmentError("Failed to commit data to substrate chain")

        logger.info("Successfully committed data to substrate chain")

    except Exception as e:
        if isinstance(e, CommitmentError):
            raise
        raise CommitmentError(
            f"Error connecting to substrate or committing data: {e}"
        ) from e


def handle_upload_command(config: MinerConfig) -> None:
    """
    Handle the upload command: upload submission to Hugging Face.

    Args:
        config: Miner configuration

    Raises:
        ConfigurationError: If configuration is invalid
        UploadError: If upload fails
    """
    # Validate required parameters
    if not config.settings.hf_repo_id:
        raise ConfigurationError(
            "Hugging Face repository ID is required. Use --hf-repo-id or set hf_repo_id in config."
        )

    # Get HF token from environment variable
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ConfigurationError(
            "Hugging Face token is required. Set the HF_TOKEN environment variable."
        )

    # Upload submission to Hugging Face
    upload_submission_to_hf(
        submission_dir=config.settings.submission_dir,
        repo_id=config.settings.hf_repo_id,
        token=hf_token,
    )

    logger.info("Upload completed successfully!")
    logger.info(f"Repository: {config.settings.hf_repo_id}")
    logger.info("To commit this to substrate chain, run:")
    logger.info(
        f"  uv run python -m miner commit --hf-repo-id {config.settings.hf_repo_id}"
    )


def handle_commit_command(config: MinerConfig) -> None:
    """
    Handle the commit command: commit information to substrate chain.

    Args:
        config: Miner configuration

    Raises:
        ConfigurationError: If configuration is invalid
        CommitmentError: If substrate commitment fails
    """
    # Validate required parameters
    if not config.settings.hf_repo_id:
        raise ConfigurationError(
            "Hugging Face repository ID is required. Use --hf-repo-id or set hf_repo_id in config."
        )

    # Prepare commitment data
    commit_data = {
        "repo_id": config.settings.hf_repo_id,
    }

    # Commit to substrate chain
    commit_to_substrate(config, commit_data)

    logger.info("Substrate commitment completed successfully!")
    logger.info(f"Committed data: {commit_data}")


def main():
    """Main entry point for the miner CLI."""
    try:
        config = MinerConfig()
        command = config.settings.command

        if command == "upload":
            handle_upload_command(config)
        elif command == "commit":
            handle_commit_command(config)
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
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
