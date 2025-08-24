#!/usr/bin/env python3
"""
Test the parent-child validator system
"""

import asyncio
import logging
import sys
import os
import time
from unittest.mock import Mock, patch

import testing.postgresql
from alembic import command
from alembic.config import Config

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from core.schemas import ChainCommitmentResponse, ModelChainCommitment, ModelProvider
from core.db.db_manager import DatabaseManager
from validator.rpc.parent_server import ParentValidatorServer
from validator.rpc.child_server import ChildValidatorServer

logger = logging.getLogger(__name__)


class MockValidator:
    """Mock validator for testing"""

    def __init__(self, db_manager=None):
        self.config = Mock()
        self.config.settings = {
            "pg_database": "postgresql://test@localhost/test" if db_manager else None
        }

        self.db_manager = db_manager
        self.keypair = Mock()
        self.keypair.ss58_address = "test_validator_address"


def test_parent_child_communication():
    """Test basic parent-child validator communication with database"""

    with testing.postgresql.Postgresql() as postgresql:
        # Setup test database
        database_url = postgresql.url()
        alembic_config = Config("alembic.ini")
        alembic_config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_config, "head")

        try:
            # Initialize database manager
            db_manager = DatabaseManager(database_url)

            # Create mock validators with real database
            parent_validator = MockValidator(db_manager=db_manager)

            # Create parent server
            parent_server = ParentValidatorServer(parent_validator)

            # Test job addition
            commitment_data = ModelChainCommitment(
                version="1.0", provider=ModelProvider.HUGGING_FACE, repo_id="test/repo"
            )
            commitment = ChainCommitmentResponse(
                hotkey="test_miner", data=commitment_data
            )

            parent_server.add_job_to_queue(commitment)
            assert len(parent_server.pending_jobs) == 1

            # Test getting stats
            stats = parent_server.get_child_stats()
            assert stats["total_children"] == 0
            assert stats["pending_jobs"] == 1

            print("✓ Parent-child communication test passed!")
            return True

        except Exception as e:
            print(f"✗ Parent-child communication test failed: {e}")
            return False
        finally:
            if "db_manager" in locals():
                db_manager.close_connections()


def test_child_server_job_reception():
    """Test child server receiving jobs with database"""

    with testing.postgresql.Postgresql() as postgresql:
        # Setup test database
        database_url = postgresql.url()
        alembic_config = Config("alembic.ini")
        alembic_config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_config, "head")

        try:
            # Initialize database manager
            db_manager = DatabaseManager(database_url)

            child_validator = MockValidator(db_manager=db_manager)
            child_server = ChildValidatorServer(child_validator)

            # Mock job data
            job_data = Mock()
            job_data.jobId = 12345
            job_data.submissionId = 67890
            job_data.minerHotkey = "test_miner"
            job_data.hfRepoId = "test/repo"
            job_data.hfRepoCommit = "abc123"
            job_data.envProvider = "test_provider"
            job_data.envName = "test_env"
            job_data.logsPath = "/tmp/logs"
            job_data.randomSeed = 42
            job_data.maxRetries = 3
            job_data.retryCount = 0
            job_data.createdAt = int(time.time() * 1000)

            # Mock the async queue method to avoid pgqueuer dependency
            async def mock_queue_job(job):
                pass

            with patch.object(
                child_server, "_queue_job_for_processing", side_effect=mock_queue_job
            ):
                # Run the async test
                async def run_test():
                    result = await child_server.receiveJob(job_data)
                    assert result["accepted"] is True
                    assert "accepted and queued" in result["message"]
                    return True

                try:
                    result = asyncio.run(run_test())
                    if result:
                        print("✓ Child server job reception test passed!")
                        return True
                    else:
                        return False
                except Exception as e:
                    print(f"✗ Child server job reception test failed: {e}")
                    return False

        except Exception as e:
            print(f"✗ Child server job reception test failed: {e}")
            return False
        finally:
            if "db_manager" in locals():
                db_manager.close_connections()


def test_parent_server_stats():
    """Test parent server statistics"""

    with testing.postgresql.Postgresql() as postgresql:
        # Setup test database
        database_url = postgresql.url()
        alembic_config = Config("alembic.ini")
        alembic_config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_config, "head")

        try:
            # Initialize database manager
            db_manager = DatabaseManager(database_url)

            parent_validator = MockValidator(db_manager=db_manager)
            parent_server = ParentValidatorServer(parent_validator)

            # Add some mock children
            parent_server.children["child1"] = Mock()
            parent_server.children["child1"].last_seen = time.time()
            parent_server.children["child1"].active_jobs = {1, 2, 3}
            parent_server.children["child1"].total_jobs_sent = 10
            parent_server.children["child1"].total_jobs_completed = 7
            parent_server.children["child1"].endpoint = "localhost:8002"

            parent_server.children["child2"] = Mock()
            parent_server.children["child2"].last_seen = time.time() - 400  # Inactive
            parent_server.children["child2"].active_jobs = set()
            parent_server.children["child2"].total_jobs_sent = 5
            parent_server.children["child2"].total_jobs_completed = 5
            parent_server.children["child2"].endpoint = "localhost:8003"

            stats = parent_server.get_child_stats()

            assert stats["total_children"] == 2
            assert stats["active_children"] == 1  # Only child1 is active
            assert len(stats["children"]) == 2

            print("✓ Parent server stats test passed!")
            return True

        except Exception as e:
            print(f"✗ Parent server stats test failed: {e}")
            return False
        finally:
            if "db_manager" in locals():
                db_manager.close_connections()


def main():
    """Run all parent-child validator system tests."""
    print("Running parent-child validator system tests...\n")

    tests = [
        test_parent_child_communication,
        test_child_server_job_reception,
        test_parent_server_stats,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All parent-child validator tests passed!")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
