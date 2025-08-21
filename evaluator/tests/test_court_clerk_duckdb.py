"""
Unit tests for CourtClerk DuckDB integration.

Tests the DuckDB storage of episodes and episode steps while keeping
evaluation jobs and results in PostgreSQL.
"""

from datetime import datetime

import pytest
from kinitro_eval.db.court_clerk import CourtClerk


class TestCourtClerkDuckDB:
    """Test CourtClerk DuckDB integration for episodes and steps."""

    @pytest.fixture
    def clerk(self):
        """Create CourtClerk with in-memory DuckDB for testing."""
        return CourtClerk(duckdb_path=":memory:")

    def test_episode_creation(self, clerk):
        """Test episode creation in DuckDB."""
        episode = clerk.create_episode(
            evaluation_id=123,
            episode_index=0,
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_reward=25.5,
            success=True,
        )

        assert episode.id is not None
        assert episode.evaluation_id == 123
        assert episode.episode_index == 0
        assert episode.total_reward == 25.5
        assert episode.success is True

    def test_episode_retrieval(self, clerk):
        """Test episode retrieval from DuckDB."""
        # Create episode
        created = clerk.create_episode(
            evaluation_id=456, episode_index=1, total_reward=10.0, success=False
        )

        # Retrieve episode
        retrieved = clerk.get_episode(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.evaluation_id == 456
        assert retrieved.episode_index == 1
        assert retrieved.total_reward == 10.0
        assert retrieved.success is False

    def test_episode_listing(self, clerk):
        """Test listing episodes by evaluation ID."""
        eval_id = 789

        # Create multiple episodes
        episodes = []
        for i in range(3):
            episode = clerk.create_episode(
                evaluation_id=eval_id,
                episode_index=i,
                total_reward=float(i * 10),
                success=(i == 2),
            )
            episodes.append(episode)

        # List episodes
        listed = clerk.list_episodes_by_evaluation(eval_id)

        assert len(listed) == 3
        assert [ep.episode_index for ep in listed] == [0, 1, 2]
        assert [ep.total_reward for ep in listed] == [0.0, 10.0, 20.0]

    def test_episode_update(self, clerk):
        """Test episode updates in DuckDB."""
        # Create episode
        episode = clerk.create_episode(
            evaluation_id=101, episode_index=0, total_reward=5.0, success=False
        )

        # Update episode
        updated = clerk.update_episode(episode.id, total_reward=99.9, success=True)

        assert updated is not None
        assert updated.id == episode.id
        assert updated.total_reward == 99.9
        assert updated.success is True

    def test_episode_deletion(self, clerk):
        """Test episode deletion from DuckDB."""
        # Create episode
        episode = clerk.create_episode(
            evaluation_id=202, episode_index=0, total_reward=15.0, success=True
        )

        # Delete episode
        deleted = clerk.delete_episode(episode.id)
        assert deleted is True

        # Verify deletion
        retrieved = clerk.get_episode(episode.id)
        assert retrieved is None

    def test_episode_step_creation(self, clerk):
        """Test episode step creation in DuckDB."""
        # Create episode first
        episode = clerk.create_episode(
            evaluation_id=303, episode_index=0, total_reward=0.0, success=False
        )

        # Create step
        step = clerk.create_episode_step(
            episode_id=episode.id,
            step_index=0,
            observation_path={"arm_pos": [0.1, 0.2, 0.3]},
            reward=1.5,
            action={"move": [0.5, -0.2]},
        )

        assert step.id is not None
        assert step.episode_id == episode.id
        assert step.step_index == 0
        assert step.observation_path == {"arm_pos": [0.1, 0.2, 0.3]}
        assert step.reward == 1.5
        assert step.action == {"move": [0.5, -0.2]}

    def test_episode_step_listing(self, clerk):
        """Test listing episode steps."""
        # Create episode
        episode = clerk.create_episode(
            evaluation_id=404, episode_index=0, total_reward=0.0, success=False
        )

        # Create multiple steps
        steps = []
        for i in range(3):
            step = clerk.create_episode_step(
                episode_id=episode.id,
                step_index=i,
                observation_path={"step": i},
                reward=float(i),
                action={"action": i},
            )
            steps.append(step)

        # List steps
        listed = clerk.list_episode_steps(episode.id)

        assert len(listed) == 3
        assert [s.step_index for s in listed] == [0, 1, 2]
        assert [s.reward for s in listed] == [0.0, 1.0, 2.0]

    def test_episode_step_update(self, clerk):
        """Test episode step updates."""
        # Create episode and step
        episode = clerk.create_episode(
            evaluation_id=505, episode_index=0, total_reward=0.0, success=False
        )

        step = clerk.create_episode_step(
            episode_id=episode.id,
            step_index=0,
            observation_path={"initial": True},
            reward=1.0,
            action={"initial": True},
        )

        # Update step
        updated = clerk.update_episode_step(
            step.id, reward=5.0, action={"updated": True, "value": 42}
        )

        assert updated is not None
        assert updated.id == step.id
        assert updated.reward == 5.0
        assert updated.action == {"updated": True, "value": 42}

    def test_episode_step_deletion(self, clerk):
        """Test episode step deletion."""
        # Create episode and step
        episode = clerk.create_episode(
            evaluation_id=606, episode_index=0, total_reward=0.0, success=False
        )

        step = clerk.create_episode_step(
            episode_id=episode.id,
            step_index=0,
            observation_path={"test": True},
            reward=1.0,
            action={"test": True},
        )

        # Delete step
        deleted = clerk.delete_episode_step(step.id)
        assert deleted is True

        # Verify deletion
        retrieved = clerk.get_episode_step(step.id)
        assert retrieved is None

    def test_episode_deletion_cascades_to_steps(self, clerk):
        """Test that deleting an episode also deletes its steps."""
        # Create episode
        episode = clerk.create_episode(
            evaluation_id=707, episode_index=0, total_reward=0.0, success=False
        )

        # Create steps
        steps = []
        for i in range(2):
            step = clerk.create_episode_step(
                episode_id=episode.id,
                step_index=i,
                observation_path={"step": i},
                reward=float(i),
                action={"step": i},
            )
            steps.append(step)

        # Verify steps exist
        listed_steps = clerk.list_episode_steps(episode.id)
        assert len(listed_steps) == 2

        # Delete episode
        deleted = clerk.delete_episode(episode.id)
        assert deleted is True

        # Verify steps were also deleted (cascade)
        remaining_steps = clerk.list_episode_steps(episode.id)
        assert len(remaining_steps) == 0

        # Verify individual steps are gone
        for step in steps:
            retrieved = clerk.get_episode_step(step.id)
            assert retrieved is None

    def test_json_serialization_in_steps(self, clerk):
        """Test that complex JSON objects are properly stored and retrieved."""
        # Create episode
        episode = clerk.create_episode(
            evaluation_id=808, episode_index=0, total_reward=0.0, success=False
        )

        # Create step with complex JSON data
        complex_obs = {
            "arm_pos": [0.1, 0.2, 0.3],
            "gripper_state": {"closed": True, "force": 0.5},
            "objects": [
                {"id": 1, "pos": [1.0, 2.0, 3.0], "type": "cube"},
                {"id": 2, "pos": [4.0, 5.0, 6.0], "type": "sphere"},
            ],
        }

        complex_action = {
            "arm_target": [0.5, 0.6, 0.7],
            "gripper_cmd": "close",
            "metadata": {"timestamp": "2025-01-01T00:00:00Z"},
        }

        step = clerk.create_episode_step(
            episode_id=episode.id,
            step_index=0,
            observation_path=complex_obs,
            reward=2.5,
            action=complex_action,
        )

        # Retrieve and verify JSON integrity
        retrieved = clerk.get_episode_step(step.id)

        assert retrieved.observation_path == complex_obs
        assert retrieved.action == complex_action
        assert retrieved.observation_path["objects"][0]["type"] == "cube"
        assert retrieved.action["metadata"]["timestamp"] == "2025-01-01T00:00:00Z"
