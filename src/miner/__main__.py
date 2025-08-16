import os

import dotenv

from core.chain import commit_to_substrate
from core.errors import CommitmentError, ConfigurationError, UploadError
from core.log import get_logger
from core.submission import upload_submission_to_hf
from miner.config import MinerConfig

logger = get_logger(__name__)
dotenv.load_dotenv()


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
