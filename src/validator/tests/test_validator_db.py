#!/usr/bin/env python3
"""
Test script for validator database integration.

This script tests the database persistence functionality without requiring a full validator setup.
"""

import os
import sys
from datetime import datetime
import testing.postgresql
from sqlalchemy import create_engine
from alembic import command
from alembic.config import Config

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from validator.db.db_manager import DatabaseManager


def test_validator_state_persistence():
    """Test validator state persistence functionality."""
    print("Testing validator state persistence...")

    with testing.postgresql.Postgresql() as postgresql:
        # Use test database URL
        database_url = postgresql.url()
        alembic_config = Config("alembic.ini")
        alembic_config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_config, "head")

        try:
            # Initialize database manager
            db_manager = DatabaseManager(database_url)

            print("✓ Database manager initialized successfully")

            # Test validator hotkey
            test_validator_hotkey = "test_validator_123"

            # Test creating/getting validator state
            validator_state = db_manager.get_or_create_validator_state(
                test_validator_hotkey
            )
            print(
                f"✓ Initial validator state: last_seen_block={validator_state.last_seen_block}"
            )

            # Test updating last seen block
            new_block = 12345
            updated_state = db_manager.update_validator_last_seen_block(
                test_validator_hotkey, new_block
            )
            print(
                f"✓ Updated validator state: last_seen_block={updated_state.last_seen_block}"
            )

            # Test getting the same state again (should persist)
            retrieved_state = db_manager.get_or_create_validator_state(
                test_validator_hotkey
            )
            assert retrieved_state.last_seen_block == new_block, (
                f"Expected {new_block}, got {retrieved_state.last_seen_block}"
            )
            print(
                f"✓ Retrieved persisted state: last_seen_block={retrieved_state.last_seen_block}"
            )

            print("✓ Validator state persistence test passed!")
            return True

        except Exception as e:
            print(f"✗ Validator state persistence test failed: {e}")
            return False
        finally:
            if "db_manager" in locals():
                db_manager.close_connections()


def test_commitment_fingerprint_persistence():
    """Test commitment fingerprint persistence functionality."""
    print("\nTesting commitment fingerprint persistence...")

    with testing.postgresql.Postgresql() as postgresql:
        # Use test database URL
        database_url = postgresql.url()
        alembic_config = Config("alembic.ini")
        alembic_config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_config, "head")

        try:
            # Initialize database manager
            db_manager = DatabaseManager(database_url)

            print("✓ Database manager initialized successfully")

            # Test data
            test_miner_hotkey = "test_miner_456"
            test_fingerprint = "v1|provider|repo_id_123"

            # Test creating commitment fingerprint
            fingerprint = db_manager.get_or_create_commitment_fingerprint(
                test_miner_hotkey, test_fingerprint
            )
            print(f"✓ Created commitment fingerprint: {fingerprint.fingerprint}")

            # Test retrieving specific fingerprint
            retrieved_fingerprint = db_manager.get_commitment_fingerprint(
                test_miner_hotkey
            )
            assert retrieved_fingerprint is not None, "Fingerprint should exist"
            assert retrieved_fingerprint.fingerprint == test_fingerprint
            print(
                f"✓ Retrieved commitment fingerprint: {retrieved_fingerprint.fingerprint}"
            )

            # Test updating fingerprint
            updated_fingerprint = "v2|provider|repo_id_456"
            updated = db_manager.get_or_create_commitment_fingerprint(
                test_miner_hotkey, updated_fingerprint
            )
            print(f"✓ Updated commitment fingerprint: {updated.fingerprint}")

            # Test retrieving all fingerprints
            all_fingerprints = db_manager.get_all_commitment_fingerprints()
            print(f"✓ Retrieved {len(all_fingerprints)} fingerprints")

            print("✓ Commitment fingerprint persistence test passed!")
            return True

        except Exception as e:
            print(f"✗ Commitment fingerprint persistence test failed: {e}")
            return False
        finally:
            if "db_manager" in locals():
                db_manager.close_connections()


def test_database_schema():
    """Test that database schema exists and is accessible."""
    print("\nTesting database schema...")

    with testing.postgresql.Postgresql() as postgresql:
        # Use test database URL
        database_url = postgresql.url()
        alembic_config = Config("alembic.ini")
        alembic_config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_config, "head")

        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(database_url)

            with engine.connect() as conn:
                # Test that our tables exist
                result = conn.execute(
                    text("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('validator_state', 'commitment_fingerprints')
                """)
                )

                tables = [row[0] for row in result]
                print(f"✓ Found tables: {tables}")

                if "validator_state" not in tables:
                    print(
                        "⚠ Warning: validator_state table not found. Run migrations first."
                    )
                    return False

                if "commitment_fingerprints" not in tables:
                    print(
                        "⚠ Warning: commitment_fingerprints table not found. Run migrations first."
                    )
                    return False

            print("✓ Database schema test passed!")
            return True

        except Exception as e:
            print(f"✗ Database schema test failed: {e}")
            print("  Make sure PostgreSQL is running and the database exists.")
            return False


def main():
    """Run all database tests."""
    print("Running validator database integration tests...\n")

    tests = [
        test_database_schema,
        test_validator_state_persistence,
        test_commitment_fingerprint_persistence,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Database integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
