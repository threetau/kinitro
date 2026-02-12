"""Tests for Genesis physics simulation environments."""

from typing import Any

import numpy as np
import pytest

from kinitro.cli.env.commands import AVAILABLE_ENV_FAMILIES
from kinitro.environments.genesis.envs.g1_humanoid import G1Environment
from kinitro.environments.genesis.robot_config import (
    G1_CONFIG,
    RobotConfig,
)
from kinitro.environments.genesis.scene_generator import (
    LARGE_OBJECT_SIZE,
    SMALL_OBJECT_SIZE,
    SceneGenerator,
)
from kinitro.environments.genesis.task_generator import TaskGenerator
from kinitro.environments.genesis.task_types import (
    OBJECT_COLORS,
    OBJECT_TYPES,
    TASK_REQUIRED_PROPERTIES,
    SceneObject,
    TaskSpec,
    TaskType,
    check_task_feasibility,
)
from kinitro.environments.registry import ENVIRONMENTS
from kinitro.types import ObjectType

# =============================================================================
# Helpers
# =============================================================================


def _make_scene_object(
    object_id: str = "obj_00_red_box",
    object_type: ObjectType = ObjectType.BOX,
    position: list[float] | None = None,
    color: str = "red",
    size: float = 0.05,
    pickupable: bool = True,
    is_picked_up: bool = False,
) -> SceneObject:
    """Convenience builder for SceneObject."""
    return SceneObject(
        object_id=object_id,
        object_type=object_type,
        position=position or [1.0, 1.0, 0.05],
        color=color,
        color_rgb=OBJECT_COLORS.get(color, (0.5, 0.5, 0.5)),
        size=size,
        pickupable=pickupable,
        is_picked_up=is_picked_up,
    )


def _make_task_spec(
    task_type: TaskType = TaskType.NAVIGATE,
    target_object_id: str = "obj_00_red_box",
    target_object_type: ObjectType = ObjectType.BOX,
    target_position: list[float] | None = None,
    destination_object_id: str | None = None,
    destination_position: list[float] | None = None,
    initial_state: dict[str, Any] | None = None,
) -> TaskSpec:
    """Convenience builder for TaskSpec."""
    return TaskSpec(
        task_type=task_type,
        task_prompt=f"Do something with {target_object_type.value}.",
        target_object_id=target_object_id,
        target_object_type=target_object_type,
        target_position=target_position or [2.0, 2.0, 0.05],
        destination_object_id=destination_object_id,
        destination_position=destination_position,
        initial_state=initial_state or {},
    )


def _make_g1_env() -> G1Environment:
    """Create a G1Environment without full __init__ (for unit-testing reward/success).

    Invariant: _compute_reward and _check_success must only use their explicit
    arguments and never access self.* attributes.  If that changes, these tests
    will need a proper mock or fixture with instance state.
    """
    return object.__new__(G1Environment)


@pytest.fixture()
def g1_env() -> G1Environment:
    return _make_g1_env()


def _make_test_objects() -> list[SceneObject]:
    """Create a mixed set of pickupable + landmark objects for task generation tests."""
    return [
        _make_scene_object(
            object_id="obj_00_red_box",
            object_type=ObjectType.BOX,
            color="red",
            pickupable=True,
            position=[1.5, 0.0, 0.05],
        ),
        _make_scene_object(
            object_id="obj_01_green_sphere",
            object_type=ObjectType.SPHERE,
            color="green",
            pickupable=True,
            position=[0.0, 1.5, 0.05],
        ),
        _make_scene_object(
            object_id="obj_02_blue_cylinder",
            object_type=ObjectType.CYLINDER,
            color="blue",
            size=0.15,
            pickupable=False,
            position=[2.0, 2.0, 0.05],
        ),
        _make_scene_object(
            object_id="obj_03_yellow_box",
            object_type=ObjectType.BOX,
            color="yellow",
            size=0.2,
            pickupable=False,
            position=[-2.0, 1.0, 0.05],
        ),
    ]


# =============================================================================
# TestTaskType
# =============================================================================


class TestTaskType:
    """Tests for TaskType enum and related constants."""

    def test_enum_values(self) -> None:
        """Enum values should match expected strings."""
        assert TaskType.NAVIGATE.value == "navigate"
        assert TaskType.PICKUP.value == "pickup"
        assert TaskType.PLACE.value == "place"
        assert TaskType.PUSH.value == "push"

    def test_all_task_types_in_required_properties(self) -> None:
        """Every TaskType should have an entry in TASK_REQUIRED_PROPERTIES."""
        for task_type in TaskType:
            assert task_type in TASK_REQUIRED_PROPERTIES

    def test_required_properties_are_lists(self) -> None:
        """All required property entries should be lists of strings."""
        for task_type, props in TASK_REQUIRED_PROPERTIES.items():
            assert isinstance(props, list), f"{task_type} properties not a list"
            for p in props:
                assert isinstance(p, str)

    def test_object_types_are_valid_strings(self) -> None:
        """OBJECT_TYPES should contain known primitive types."""
        assert len(OBJECT_TYPES) > 0
        for t in OBJECT_TYPES:
            assert isinstance(t, str)
        assert "box" in OBJECT_TYPES
        assert "sphere" in OBJECT_TYPES
        assert "cylinder" in OBJECT_TYPES

    def test_enum_has_exactly_four_members(self) -> None:
        """Should have exactly NAVIGATE, PICKUP, PLACE, PUSH."""
        assert len(TaskType) == 4


# =============================================================================
# TestTaskSpec
# =============================================================================


class TestTaskSpec:
    """Tests for TaskSpec serialization."""

    def test_roundtrip_all_fields(self) -> None:
        """to_dict → from_dict should preserve all fields."""
        spec = _make_task_spec(
            task_type=TaskType.PLACE,
            destination_object_id="obj_02_blue_cylinder",
            destination_position=[3.0, 3.0, 0.1],
            initial_state={"initial_target_pos": [1.0, 1.0, 0.05]},
        )
        restored = TaskSpec.from_dict(spec.to_dict())

        assert restored.task_type == spec.task_type
        assert restored.task_prompt == spec.task_prompt
        assert restored.target_object_id == spec.target_object_id
        assert restored.target_object_type == spec.target_object_type
        assert restored.target_position == spec.target_position
        assert restored.destination_object_id == spec.destination_object_id
        assert restored.destination_position == spec.destination_position
        assert restored.initial_state == spec.initial_state

    def test_roundtrip_optional_fields_none(self) -> None:
        """Optional destination fields should survive roundtrip as None."""
        spec = _make_task_spec(task_type=TaskType.NAVIGATE)
        restored = TaskSpec.from_dict(spec.to_dict())

        assert restored.destination_object_id is None
        assert restored.destination_position is None

    def test_roundtrip_with_initial_state(self) -> None:
        """initial_state dict should survive roundtrip."""
        spec = _make_task_spec(
            task_type=TaskType.PICKUP,
            initial_state={"initial_height": 0.05, "extra": [1, 2, 3]},
        )
        restored = TaskSpec.from_dict(spec.to_dict())
        assert restored.initial_state == {"initial_height": 0.05, "extra": [1, 2, 3]}

    def test_from_dict_invalid_task_type(self) -> None:
        """from_dict should raise ValueError on invalid task_type string."""
        data = _make_task_spec().to_dict()
        data["task_type"] = "fly_to_moon"
        with pytest.raises(ValueError):
            TaskSpec.from_dict(data)

    def test_to_dict_keys(self) -> None:
        """to_dict should contain all expected keys."""
        data = _make_task_spec().to_dict()
        expected_keys = {
            "task_type",
            "task_prompt",
            "target_object_id",
            "target_object_type",
            "target_position",
            "destination_object_id",
            "destination_position",
            "initial_state",
        }
        assert set(data.keys()) == expected_keys


# =============================================================================
# TestSceneObject
# =============================================================================


class TestSceneObject:
    """Tests for SceneObject data class."""

    def test_construction(self) -> None:
        """Should construct with all fields."""
        obj = _make_scene_object()
        assert obj.object_id == "obj_00_red_box"
        assert obj.object_type == ObjectType.BOX
        assert obj.color == "red"
        assert obj.color_rgb == (0.9, 0.2, 0.2)
        assert obj.pickupable is True

    def test_default_is_picked_up(self) -> None:
        """is_picked_up should default to False."""
        obj = _make_scene_object()
        assert obj.is_picked_up is False

    def test_color_palette_distinct(self) -> None:
        """All colors in the palette should have distinct RGB values."""
        rgb_values = list(OBJECT_COLORS.values())
        assert len(rgb_values) == len(set(rgb_values)), "Duplicate RGB values in palette"
        assert len(OBJECT_COLORS) >= 6, "Palette should have at least 6 colors"


# =============================================================================
# TestCheckTaskFeasibility
# =============================================================================


class TestCheckTaskFeasibility:
    """Tests for check_task_feasibility function."""

    @pytest.mark.parametrize(
        "task_type, obj_kwargs, dest_factory, expected",
        [
            pytest.param(TaskType.NAVIGATE, {"pickupable": False}, None, True, id="navigate_any"),
            pytest.param(TaskType.PICKUP, {"pickupable": True}, None, True, id="pickup_ok"),
            pytest.param(
                TaskType.PLACE,
                {"pickupable": True},
                lambda: _make_scene_object(object_id="dest", pickupable=False),
                True,
                id="place_ok",
            ),
            pytest.param(
                TaskType.PUSH,
                {"object_id": "a"},
                lambda: _make_scene_object(object_id="b"),
                True,
                id="push_ok",
            ),
        ],
    )
    def test_feasible_cases(self, task_type, obj_kwargs, dest_factory, expected) -> None:
        obj = _make_scene_object(**obj_kwargs)
        dest = dest_factory() if dest_factory else None
        feasible, _ = check_task_feasibility(task_type, obj, destination=dest)
        assert feasible is expected

    @pytest.mark.parametrize(
        "task_type, obj_kwargs, dest_factory, reason_substr",
        [
            pytest.param(
                TaskType.PICKUP,
                {"pickupable": False},
                None,
                "not pickupable",
                id="pickup_not_pickupable",
            ),
            pytest.param(
                TaskType.PICKUP,
                {"pickupable": True, "is_picked_up": True},
                None,
                "already picked up",
                id="pickup_already_picked",
            ),
            pytest.param(
                TaskType.PLACE, {"pickupable": True}, None, "destination", id="place_no_dest"
            ),
            pytest.param(
                TaskType.PUSH,
                {"object_id": "same"},
                lambda: _make_scene_object(object_id="same"),
                "itself",
                id="push_same_object",
            ),
            pytest.param(TaskType.PUSH, {}, None, "destination", id="push_no_dest"),
        ],
    )
    def test_infeasible_cases(self, task_type, obj_kwargs, dest_factory, reason_substr) -> None:
        obj = _make_scene_object(**obj_kwargs)
        dest = dest_factory() if dest_factory else None
        feasible, reason = check_task_feasibility(task_type, obj, destination=dest)
        assert feasible is False
        assert reason_substr in reason.lower()

    def test_robot_capability_filtering_unsupported(self) -> None:
        """Task type not in robot_supported_tasks should be infeasible."""
        obj = _make_scene_object()
        feasible, reason = check_task_feasibility(
            TaskType.PICKUP, obj, robot_supported_tasks=[TaskType.NAVIGATE]
        )
        assert feasible is False
        assert "does not support" in reason.lower()

    def test_robot_capability_filtering_none_allows_all(self) -> None:
        """robot_supported_tasks=None should allow all task types."""
        obj = _make_scene_object(pickupable=True)
        feasible, _ = check_task_feasibility(TaskType.PICKUP, obj, robot_supported_tasks=None)
        assert feasible is True


# =============================================================================
# TestSceneGenerator
# =============================================================================


class TestSceneGenerator:
    """Tests for procedural scene generation."""

    def test_deterministic(self) -> None:
        """Same seed should produce identical scene config."""
        gen = SceneGenerator()
        s1 = gen.generate_scene(42)
        s2 = gen.generate_scene(42)

        assert s1.terrain_type == s2.terrain_type
        assert len(s1.objects) == len(s2.objects)
        for a, b in zip(s1.objects, s2.objects):
            assert a.object_id == b.object_id
            assert a.position == b.position

    def test_object_count_in_range(self) -> None:
        """Object count should be within the configured range."""
        gen = SceneGenerator(num_objects=(3, 6))
        for seed in range(20):
            scene = gen.generate_scene(seed)
            assert 3 <= len(scene.objects) <= 6, f"seed={seed}: {len(scene.objects)} objects"

    def test_at_least_one_pickupable_and_one_landmark(self) -> None:
        """Each scene should have at least 1 pickupable and 1 non-pickupable object."""
        gen = SceneGenerator(num_objects=(3, 6))
        for seed in range(20):
            scene = gen.generate_scene(seed)
            pickupable = [o for o in scene.objects if o.pickupable]
            landmarks = [o for o in scene.objects if not o.pickupable]
            assert len(pickupable) >= 1, f"seed={seed}: no pickupable objects"
            assert len(landmarks) >= 1, f"seed={seed}: no landmark objects"

    def test_objects_avoid_center(self) -> None:
        """All objects should be placed away from the robot spawn area."""
        gen = SceneGenerator(num_objects=(3, 6))
        for seed in range(20):
            scene = gen.generate_scene(seed)
            for obj in scene.objects:
                dist = np.sqrt(obj.position[0] ** 2 + obj.position[1] ** 2)
                assert dist >= 0.8, f"seed={seed}: {obj.object_id} too close to center ({dist:.2f})"

    def test_pickupable_within_reachable_range(self) -> None:
        """Pickupable objects should be within 70% of arena half-size."""
        gen = SceneGenerator(num_objects=(3, 6), arena_size=5.0)
        max_dist = (5.0 / 2.0) * 0.7
        for seed in range(20):
            scene = gen.generate_scene(seed)
            for obj in scene.objects:
                if obj.pickupable:
                    dist = np.sqrt(obj.position[0] ** 2 + obj.position[1] ** 2)
                    assert dist <= max_dist + 0.01, (
                        f"seed={seed}: pickupable {obj.object_id} too far ({dist:.2f} > {max_dist:.2f})"
                    )

    def test_always_flat_terrain(self) -> None:
        """All seeds should produce flat terrain with empty params."""
        gen = SceneGenerator()
        for seed in range(20):
            scene = gen.generate_scene(seed)
            assert scene.terrain_type == "flat", f"seed={seed}: got {scene.terrain_type}"
            assert scene.terrain_params == {}, f"seed={seed}: non-empty terrain_params"

    def test_get_scene_objects_conversion(self) -> None:
        """get_scene_objects() should produce SceneObject instances matching config."""
        gen = SceneGenerator()
        scene = gen.generate_scene(42)
        objects = scene.get_scene_objects()

        assert len(objects) == len(scene.objects)
        for obj, cfg in zip(objects, scene.objects):
            assert isinstance(obj, SceneObject)
            assert obj.object_id == cfg.object_id
            assert obj.object_type == cfg.object_type
            assert obj.position == cfg.position
            assert obj.pickupable == cfg.pickupable
            assert obj.is_picked_up is False

    def test_object_sizes(self) -> None:
        """Pickupable objects should be small, landmarks should be large."""
        gen = SceneGenerator()
        for seed in range(20):
            scene = gen.generate_scene(seed)
            for obj in scene.objects:
                if obj.pickupable:
                    assert SMALL_OBJECT_SIZE[0] <= obj.size <= SMALL_OBJECT_SIZE[1], (
                        f"seed={seed}: pickupable {obj.object_id} size {obj.size} out of range"
                    )
                else:
                    assert LARGE_OBJECT_SIZE[0] <= obj.size <= LARGE_OBJECT_SIZE[1], (
                        f"seed={seed}: landmark {obj.object_id} size {obj.size} out of range"
                    )


# =============================================================================
# TestTaskGenerator
# =============================================================================


class TestTaskGenerator:
    """Tests for scene-grounded task prompt generation."""

    def test_deterministic(self) -> None:
        """Same seed + same objects should produce same task."""
        gen = TaskGenerator()
        objects = _make_test_objects()

        rng1 = np.random.default_rng(42)
        task1 = gen.generate_task(objects, rng1)

        rng2 = np.random.default_rng(42)
        task2 = gen.generate_task(objects, rng2)

        assert task1 is not None and task2 is not None
        assert task1.task_type == task2.task_type
        assert task1.target_object_id == task2.target_object_id
        assert task1.task_prompt == task2.task_prompt

    def test_returns_none_for_empty_objects(self) -> None:
        """Should return None when no objects are available."""
        gen = TaskGenerator()
        rng = np.random.default_rng(42)
        task = gen.generate_task([], rng)
        assert task is None

    def test_navigate_selects_any_object(self) -> None:
        """NAVIGATE should select from any object in the scene."""
        gen = TaskGenerator(task_types=[TaskType.NAVIGATE])
        objects = _make_test_objects()
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)

        assert task is not None
        assert task.task_type == TaskType.NAVIGATE
        assert task.target_object_id in [o.object_id for o in objects]

    def test_pickup_only_pickupable(self) -> None:
        """PICKUP should only select pickupable objects."""
        gen = TaskGenerator(task_types=[TaskType.PICKUP])
        objects = _make_test_objects()
        pickupable_ids = {o.object_id for o in objects if o.pickupable}
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)

        assert task is not None
        assert task.task_type == TaskType.PICKUP
        assert task.target_object_id in pickupable_ids

    def test_place_requires_pickupable_and_landmark(self) -> None:
        """PLACE should require both pickupable and landmark objects."""
        gen = TaskGenerator(task_types=[TaskType.PLACE])
        objects = _make_test_objects()
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)

        assert task is not None
        assert task.task_type == TaskType.PLACE
        assert task.destination_object_id is not None
        # Target should be pickupable
        target = next(o for o in objects if o.object_id == task.target_object_id)
        assert target.pickupable is True

    def test_place_returns_none_no_landmarks(self) -> None:
        """PLACE should return None when there are no landmark objects."""
        gen = TaskGenerator(task_types=[TaskType.PLACE])
        # All pickupable, no landmarks
        objects = [
            _make_scene_object(object_id="a", pickupable=True),
            _make_scene_object(object_id="b", pickupable=True),
        ]
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)
        assert task is None

    def test_push_requires_two_objects(self) -> None:
        """PUSH should require at least 2 objects (target != destination)."""
        gen = TaskGenerator(task_types=[TaskType.PUSH])
        objects = _make_test_objects()
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)

        assert task is not None
        assert task.task_type == TaskType.PUSH
        assert task.target_object_id != task.destination_object_id

    def test_push_returns_none_single_object(self) -> None:
        """PUSH should return None with only 1 object."""
        gen = TaskGenerator(task_types=[TaskType.PUSH])
        objects = [_make_scene_object(object_id="only_one")]
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)
        assert task is None

    def test_prompt_contains_target_info(self) -> None:
        """Generated prompt should contain target color and object type."""
        gen = TaskGenerator(task_types=[TaskType.NAVIGATE])
        objects = _make_test_objects()
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)

        assert task is not None
        target = next(o for o in objects if o.object_id == task.target_object_id)
        assert target.color in task.task_prompt.lower()
        assert target.object_type.value in task.task_prompt.lower()

    def test_place_prompt_contains_destination_info(self) -> None:
        """PLACE prompt should contain destination color and object type."""
        gen = TaskGenerator(task_types=[TaskType.PLACE])
        objects = _make_test_objects()
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng)

        assert task is not None
        dest = next(o for o in objects if o.object_id == task.destination_object_id)
        assert dest.color in task.task_prompt.lower()
        assert dest.object_type.value in task.task_prompt.lower()

    def test_robot_capability_filtering(self) -> None:
        """Unsupported task types should be excluded by robot config."""
        navigate_only_config = RobotConfig(
            name="test",
            mjcf_path="test.xml",
            morph_type="mjcf",
            init_pos=(0.0, 0.0, 0.5),
            init_quat=(1.0, 0.0, 0.0, 0.0),
            num_actuated_dofs=1,
            joint_names=["j1"],
            default_dof_pos=[0.0],
            action_scale=[1.0],
            fall_height_threshold=0.2,
            ego_camera_link="link",
            ego_camera_pos_offset=(0.0, 0.0, 0.0),
            supported_task_types=[TaskType.NAVIGATE],
        )
        gen = TaskGenerator()
        objects = _make_test_objects()
        rng = np.random.default_rng(42)

        # With navigate-only robot, should only generate navigate tasks
        for _ in range(20):
            task = gen.generate_task(objects, rng, robot_config=navigate_only_config)
            if task is not None:
                assert task.task_type == TaskType.NAVIGATE

    def test_generate_task_with_specific_type(self) -> None:
        """generate_task with specific task_type parameter should honour it."""
        gen = TaskGenerator()
        objects = _make_test_objects()
        rng = np.random.default_rng(42)

        task = gen.generate_task(objects, rng, task_type=TaskType.NAVIGATE)

        assert task is not None
        assert task.task_type == TaskType.NAVIGATE


# =============================================================================
# TestG1Reward
# =============================================================================


class TestG1Reward:
    """Tests for G1Environment._compute_reward using object.__new__() bypass."""

    def test_navigate_alive_bonus(self, g1_env) -> None:
        """NAVIGATE reward should always include alive bonus."""
        robot_state = {"base_pos": np.array([0.0, 0.0, 0.75])}
        spec = _make_task_spec(task_type=TaskType.NAVIGATE, target_position=[5.0, 5.0, 0.0])

        reward = g1_env._compute_reward(robot_state, {}, spec)

        assert reward >= 0.01  # alive bonus

    def test_navigate_reward_increases_closer(self, g1_env) -> None:
        """NAVIGATE reward should increase as distance decreases."""
        spec = _make_task_spec(task_type=TaskType.NAVIGATE, target_position=[3.0, 0.0, 0.0])

        far_state = {"base_pos": np.array([0.0, 0.0, 0.75])}
        near_state = {"base_pos": np.array([2.5, 0.0, 0.75])}

        r_far = g1_env._compute_reward(far_state, {}, spec)
        r_near = g1_env._compute_reward(near_state, {}, spec)

        assert r_near > r_far

    def test_navigate_high_reward_near_target(self, g1_env) -> None:
        """NAVIGATE reward should be higher when very close to target."""
        spec = _make_task_spec(task_type=TaskType.NAVIGATE, target_position=[0.1, 0.0, 0.0])
        robot_state = {"base_pos": np.array([0.0, 0.0, 0.75])}

        reward = g1_env._compute_reward(robot_state, {}, spec)
        assert reward > 0.01  # more than just alive bonus

    def test_pickup_approach_reward(self, g1_env) -> None:
        """PICKUP should give approach reward when near the object."""
        spec = _make_task_spec(
            task_type=TaskType.PICKUP,
            target_object_id="obj_00",
            target_position=[1.0, 0.0, 0.05],
            initial_state={"initial_height": 0.05},
        )
        robot_state = {"base_pos": np.array([0.8, 0.0, 0.75])}
        obj_states = {"obj_00": np.array([1.0, 0.0, 0.05])}

        reward = g1_env._compute_reward(robot_state, obj_states, spec)
        assert reward > 0.01  # alive + approach

    def test_pickup_lift_bonus(self, g1_env) -> None:
        """PICKUP should give large bonus when object is lifted above threshold."""
        spec = _make_task_spec(
            task_type=TaskType.PICKUP,
            target_object_id="obj_00",
            target_position=[1.0, 0.0, 0.05],
            initial_state={"initial_height": 0.05},
        )
        robot_state = {"base_pos": np.array([1.0, 0.0, 0.75])}
        obj_states = {"obj_00": np.array([1.0, 0.0, 0.5])}  # lifted well above 0.05 + 0.15

        reward = g1_env._compute_reward(robot_state, obj_states, spec)
        assert reward >= 1.0  # lift bonus of 1.0

    def test_pickup_no_lift_bonus_below_threshold(self, g1_env) -> None:
        """PICKUP should not give lift bonus when height below threshold."""
        spec = _make_task_spec(
            task_type=TaskType.PICKUP,
            target_object_id="obj_00",
            target_position=[1.0, 0.0, 0.05],
            initial_state={"initial_height": 0.05},
        )
        robot_state = {"base_pos": np.array([1.0, 0.0, 0.75])}
        obj_states = {"obj_00": np.array([1.0, 0.0, 0.1])}  # only 0.05 above, < 0.15

        reward = g1_env._compute_reward(robot_state, obj_states, spec)
        assert reward < 1.0  # no lift bonus

    def test_pickup_missing_object_only_alive_bonus(self, g1_env) -> None:
        """PICKUP should return only alive bonus when object not in states."""
        spec = _make_task_spec(
            task_type=TaskType.PICKUP,
            target_object_id="obj_missing",
            initial_state={"initial_height": 0.05},
        )
        robot_state = {"base_pos": np.array([0.0, 0.0, 0.75])}

        reward = g1_env._compute_reward(robot_state, {}, spec)
        assert abs(reward - 0.01) < 1e-6

    def test_place_reward_based_on_distance(self, g1_env) -> None:
        """PLACE reward should depend on object-to-destination distance."""
        spec = _make_task_spec(
            task_type=TaskType.PLACE,
            target_object_id="obj_00",
            destination_position=[3.0, 3.0, 0.1],
        )
        robot_state = {"base_pos": np.array([0.0, 0.0, 0.75])}

        # Object far from destination
        far = {"obj_00": np.array([0.0, 0.0, 0.05])}
        r_far = g1_env._compute_reward(robot_state, far, spec)

        # Object close to destination
        close = {"obj_00": np.array([2.9, 2.9, 0.1])}
        r_close = g1_env._compute_reward(robot_state, close, spec)

        assert r_close > r_far

    def test_push_reward_based_on_xy_distance(self, g1_env) -> None:
        """PUSH reward should depend on XY distance to destination."""
        spec = _make_task_spec(
            task_type=TaskType.PUSH,
            target_object_id="obj_00",
            destination_position=[3.0, 0.0, 0.05],
        )
        robot_state = {"base_pos": np.array([0.0, 0.0, 0.75])}

        far = {"obj_00": np.array([0.0, 0.0, 0.05])}
        close = {"obj_00": np.array([2.8, 0.0, 0.05])}

        r_far = g1_env._compute_reward(robot_state, far, spec)
        r_close = g1_env._compute_reward(robot_state, close, spec)

        assert r_close > r_far

    def test_all_rewards_non_negative(self, g1_env) -> None:
        """All task types should produce non-negative rewards (alive bonus)."""
        robot_state = {"base_pos": np.array([0.0, 0.0, 0.75])}
        obj_states = {"obj_00": np.array([2.0, 2.0, 0.05])}

        for task_type in TaskType:
            spec = _make_task_spec(
                task_type=task_type,
                target_object_id="obj_00",
                destination_position=[3.0, 3.0, 0.1],
                initial_state={"initial_height": 0.05},
            )
            reward = g1_env._compute_reward(robot_state, obj_states, spec)
            assert reward >= 0.0, f"{task_type} produced negative reward: {reward}"


# =============================================================================
# TestG1Success
# =============================================================================


class TestG1Success:
    """Tests for G1Environment._check_success using object.__new__() bypass."""

    @pytest.mark.parametrize(
        "task_type, spec_kwargs, robot_pos, obj_states, expected",
        [
            pytest.param(
                TaskType.NAVIGATE,
                {"target_position": [3.0, 0.0, 0.0]},
                [2.8, 0.0, 0.75],
                {},
                True,
                id="navigate_success_close",
            ),
            pytest.param(
                TaskType.NAVIGATE,
                {"target_position": [3.0, 0.0, 0.0]},
                [0.0, 0.0, 0.75],
                {},
                False,
                id="navigate_failure_far",
            ),
            pytest.param(
                TaskType.PICKUP,
                {"target_object_id": "obj_00", "initial_state": {"initial_height": 0.05}},
                [0.0, 0.0, 0.75],
                {"obj_00": np.array([1.0, 0.0, 0.25])},
                True,
                id="pickup_success_lifted",
            ),
            pytest.param(
                TaskType.PICKUP,
                {"target_object_id": "obj_00", "initial_state": {"initial_height": 0.05}},
                [0.0, 0.0, 0.75],
                {"obj_00": np.array([1.0, 0.0, 0.1])},
                False,
                id="pickup_failure_not_lifted",
            ),
            pytest.param(
                TaskType.PICKUP,
                {"target_object_id": "obj_missing", "initial_state": {"initial_height": 0.05}},
                [0.0, 0.0, 0.75],
                {},
                False,
                id="pickup_failure_missing",
            ),
            pytest.param(
                TaskType.PLACE,
                {"target_object_id": "obj_00", "destination_position": [3.0, 3.0, 0.1]},
                [0.0, 0.0, 0.75],
                {"obj_00": np.array([3.1, 3.1, 0.1])},
                True,
                id="place_success_within_threshold",
            ),
            pytest.param(
                TaskType.PLACE,
                {"target_object_id": "obj_00", "destination_position": [3.0, 3.0, 0.1]},
                [0.0, 0.0, 0.75],
                {"obj_00": np.array([0.0, 0.0, 0.05])},
                False,
                id="place_failure_too_far",
            ),
            pytest.param(
                TaskType.PLACE,
                {"target_object_id": "obj_00", "destination_position": None},
                [0.0, 0.0, 0.75],
                {"obj_00": np.array([0.0, 0.0, 0.05])},
                False,
                id="place_failure_dest_none",
            ),
            pytest.param(
                TaskType.PUSH,
                {"target_object_id": "obj_00", "destination_position": [3.0, 0.0, 0.05]},
                [0.0, 0.0, 0.75],
                {"obj_00": np.array([2.8, 0.0, 0.05])},
                True,
                id="push_success_close",
            ),
            pytest.param(
                TaskType.PUSH,
                {"target_object_id": "obj_00", "destination_position": [3.0, 0.0, 0.05]},
                [0.0, 0.0, 0.75],
                {"obj_00": np.array([0.0, 0.0, 0.05])},
                False,
                id="push_failure_too_far",
            ),
        ],
    )
    def test_check_success(
        self, g1_env, task_type, spec_kwargs, robot_pos, obj_states, expected
    ) -> None:
        robot_state = {"base_pos": np.array(robot_pos)}
        spec = _make_task_spec(task_type=task_type, **spec_kwargs)
        assert g1_env._check_success(robot_state, obj_states, spec) is expected


# =============================================================================
# TestRobotConfig
# =============================================================================


class TestRobotConfig:
    """Tests for G1_CONFIG robot configuration consistency."""

    def test_g1_dof_consistency(self) -> None:
        """joint_names, default_dof_pos, and action_scale should all match num_actuated_dofs."""
        assert len(G1_CONFIG.joint_names) == G1_CONFIG.num_actuated_dofs
        assert len(G1_CONFIG.default_dof_pos) == G1_CONFIG.num_actuated_dofs
        assert len(G1_CONFIG.action_scale) == G1_CONFIG.num_actuated_dofs

    def test_g1_init_pos_and_quat(self) -> None:
        """init_pos should have 3 elements, init_quat should have 4."""
        assert len(G1_CONFIG.init_pos) == 3
        assert len(G1_CONFIG.init_quat) == 4

    def test_g1_supported_task_types_non_empty(self) -> None:
        """supported_task_types should be non-empty."""
        assert len(G1_CONFIG.supported_task_types) > 0

    def test_g1_ego_camera_link_non_empty(self) -> None:
        """ego_camera_link should be a non-empty string."""
        assert isinstance(G1_CONFIG.ego_camera_link, str)
        assert len(G1_CONFIG.ego_camera_link) > 0

    def test_g1_ego_camera_pos_offset(self) -> None:
        """ego_camera_pos_offset should have 3 elements."""
        assert len(G1_CONFIG.ego_camera_pos_offset) == 3


# =============================================================================
# TestGenesisRegistry
# =============================================================================


class TestGenesisRegistry:
    """Tests for genesis environment registration."""

    def test_g1_in_environments(self) -> None:
        """genesis/g1-v0 should be in the ENVIRONMENTS registry."""
        assert "genesis/g1-v0" in ENVIRONMENTS

    def test_factory_is_callable(self) -> None:
        """The factory for genesis/g1-v0 should be callable."""
        assert callable(ENVIRONMENTS["genesis/g1-v0"])

    def test_cli_families_include_genesis(self) -> None:
        """AVAILABLE_ENV_FAMILIES should include 'genesis'."""
        assert "genesis" in AVAILABLE_ENV_FAMILIES
