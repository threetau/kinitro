"""Tests for ProcTHOR procedural environment."""

import numpy as np

from kinitro.environments.procthor.environment import ProcTHOREnvironment
from kinitro.environments.procthor.house_generator import (
    HouseGenerator,
    get_openable_objects,
    get_pickupable_objects,
    get_receptacles,
)
from kinitro.environments.procthor.task_generator import (
    TaskGenerator,
    format_object_name,
)
from kinitro.environments.procthor.task_types import (
    SceneObject,
    TaskSpec,
    TaskType,
    check_task_feasibility,
)


class TestSceneObject:
    """Tests for SceneObject data class."""

    def test_from_ai2thor_metadata(self):
        """Should correctly parse AI2-THOR object metadata."""
        metadata = {
            "objectId": "Apple|1|2|3",
            "objectType": "Apple",
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            "pickupable": True,
            "openable": False,
            "toggleable": False,
            "receptacle": False,
            "isOpen": False,
            "isToggled": False,
            "isPickedUp": False,
            "parentReceptacles": ["CounterTop|1"],
            "visible": True,
        }

        obj = SceneObject.from_ai2thor_metadata(metadata)

        assert obj.object_id == "Apple|1|2|3"
        assert obj.object_type == "Apple"
        assert obj.position["x"] == 1.0
        assert obj.pickupable is True
        assert obj.openable is False
        assert obj.parent_receptacles == ["CounterTop|1"]

    def test_has_property(self):
        """Should correctly check for properties."""
        obj = SceneObject(
            object_id="test",
            object_type="Test",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            pickupable=True,
            openable=False,
        )

        assert obj.has_property("pickupable") is True
        assert obj.has_property("openable") is False


class TestTaskSpec:
    """Tests for TaskSpec data class."""

    def test_to_dict_roundtrip(self):
        """Should serialize and deserialize correctly."""
        spec = TaskSpec(
            task_type=TaskType.PICKUP,
            task_prompt="Pick up the apple.",
            target_object_id="Apple|1",
            target_object_type="Apple",
            initial_state={"is_picked_up": False},
        )

        data = spec.to_dict()
        restored = TaskSpec.from_dict(data)

        assert restored.task_type == spec.task_type
        assert restored.task_prompt == spec.task_prompt
        assert restored.target_object_id == spec.target_object_id
        assert restored.initial_state == spec.initial_state

    def test_place_task_with_destination(self):
        """Place tasks should include destination."""
        spec = TaskSpec(
            task_type=TaskType.PLACE,
            task_prompt="Put the apple on the table.",
            target_object_id="Apple|1",
            target_object_type="Apple",
            destination_object_id="DiningTable|1",
            destination_object_type="DiningTable",
        )

        data = spec.to_dict()
        assert data["destination_object_id"] == "DiningTable|1"


class TestTaskFeasibility:
    """Tests for task feasibility checking."""

    def test_pickup_feasible(self):
        """Pickup should be feasible for pickupable objects."""
        obj = SceneObject(
            object_id="Apple|1",
            object_type="Apple",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            pickupable=True,
            is_picked_up=False,
        )

        feasible, _ = check_task_feasibility(TaskType.PICKUP, obj)
        assert feasible is True

    def test_pickup_not_pickupable(self):
        """Pickup should not be feasible for non-pickupable objects."""
        obj = SceneObject(
            object_id="Table|1",
            object_type="Table",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            pickupable=False,
        )

        feasible, reason = check_task_feasibility(TaskType.PICKUP, obj)
        assert feasible is False
        assert "not pickupable" in reason

    def test_pickup_already_picked_up(self):
        """Pickup should not be feasible if object already held."""
        obj = SceneObject(
            object_id="Apple|1",
            object_type="Apple",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            pickupable=True,
            is_picked_up=True,
        )

        feasible, _ = check_task_feasibility(TaskType.PICKUP, obj)
        assert feasible is False

    def test_open_feasible(self):
        """Open should be feasible for closed openable objects."""
        obj = SceneObject(
            object_id="Fridge|1",
            object_type="Fridge",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            openable=True,
            is_open=False,
        )

        feasible, _ = check_task_feasibility(TaskType.OPEN, obj)
        assert feasible is True

    def test_open_already_open(self):
        """Open should not be feasible if already open."""
        obj = SceneObject(
            object_id="Fridge|1",
            object_type="Fridge",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            openable=True,
            is_open=True,
        )

        feasible, _ = check_task_feasibility(TaskType.OPEN, obj)
        assert feasible is False

    def test_place_requires_receptacle(self):
        """Place should require destination to be a receptacle."""
        target = SceneObject(
            object_id="Apple|1",
            object_type="Apple",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            pickupable=True,
        )
        destination = SceneObject(
            object_id="Chair|1",
            object_type="Chair",
            position={"x": 0, "y": 0, "z": 0},
            rotation={"x": 0, "y": 0, "z": 0},
            receptacle=False,
        )

        feasible, reason = check_task_feasibility(TaskType.PLACE, target, destination)
        assert feasible is False
        assert "not a receptacle" in reason


class TestFormatObjectName:
    """Tests for object name formatting."""

    def test_simple_name(self):
        """Simple names should be lowercased."""
        assert format_object_name("Apple") == "apple"

    def test_camel_case(self):
        """CamelCase should be split with spaces."""
        assert format_object_name("CoffeeMachine") == "coffee machine"
        assert format_object_name("DiningTable") == "dining table"

    def test_multiple_caps(self):
        """Multiple capital letters should be handled."""
        assert format_object_name("TVStand") == "t v stand"


class TestTaskGenerator:
    """Tests for task generation."""

    def _make_test_objects(self) -> list[SceneObject]:
        """Create a set of test objects."""
        return [
            SceneObject(
                object_id="Apple|1",
                object_type="Apple",
                position={"x": 1, "y": 1, "z": 1},
                rotation={"x": 0, "y": 0, "z": 0},
                pickupable=True,
                is_picked_up=False,
            ),
            SceneObject(
                object_id="Fridge|1",
                object_type="Fridge",
                position={"x": 2, "y": 0, "z": 2},
                rotation={"x": 0, "y": 0, "z": 0},
                openable=True,
                receptacle=True,
                is_open=False,
            ),
            SceneObject(
                object_id="Lamp|1",
                object_type="Lamp",
                position={"x": 3, "y": 1, "z": 3},
                rotation={"x": 0, "y": 0, "z": 0},
                toggleable=True,
                is_toggled=False,
            ),
            SceneObject(
                object_id="CounterTop|1",
                object_type="CounterTop",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                receptacle=True,
            ),
        ]

    def test_generate_pickup_task(self):
        """Should generate a valid pickup task."""
        generator = TaskGenerator(task_types=[TaskType.PICKUP])
        objects = self._make_test_objects()
        rng = np.random.default_rng(42)

        task = generator.generate_task(objects, rng)

        assert task is not None
        assert task.task_type == TaskType.PICKUP
        assert task.target_object_id == "Apple|1"
        assert "apple" in task.task_prompt.lower()

    def test_generate_open_task(self):
        """Should generate a valid open task."""
        generator = TaskGenerator(task_types=[TaskType.OPEN])
        objects = self._make_test_objects()
        rng = np.random.default_rng(42)

        task = generator.generate_task(objects, rng)

        assert task is not None
        assert task.task_type == TaskType.OPEN
        assert task.target_object_id == "Fridge|1"

    def test_generate_place_task(self):
        """Should generate a valid place task."""
        generator = TaskGenerator(task_types=[TaskType.PLACE])
        objects = self._make_test_objects()
        rng = np.random.default_rng(42)

        task = generator.generate_task(objects, rng)

        assert task is not None
        assert task.task_type == TaskType.PLACE
        assert task.destination_object_id is not None

    def test_deterministic_generation(self):
        """Same seed should produce same task."""
        generator = TaskGenerator()
        objects = self._make_test_objects()

        rng1 = np.random.default_rng(42)
        task1 = generator.generate_task(objects, rng1)

        rng2 = np.random.default_rng(42)
        task2 = generator.generate_task(objects, rng2)

        assert task1 is not None
        assert task2 is not None
        assert task1.task_type == task2.task_type
        assert task1.target_object_id == task2.target_object_id

    def test_returns_none_for_empty_scene(self):
        """Should return None if no feasible tasks exist."""
        generator = TaskGenerator(task_types=[TaskType.PICKUP])
        objects = []  # No objects
        rng = np.random.default_rng(42)

        task = generator.generate_task(objects, rng)

        assert task is None


class TestHouseGenerator:
    """Tests for house generation."""

    def test_get_scene_name_procedural(self):
        """Should return house dict for procedural houses."""
        generator = HouseGenerator()
        house = {"rooms": [{"id": "room1"}], "metadata": {"agent": {}}}

        scene = generator.get_scene_name(house)

        # For procedural houses, get_scene_name returns the house dict itself
        assert scene == house

    def test_cache_enabled(self):
        """Cache should be enabled by default."""
        generator = HouseGenerator()
        assert generator._use_cache is True
        assert generator._house_cache == {}

    def test_clear_cache(self):
        """Should clear house cache."""
        generator = HouseGenerator()
        generator._house_cache[42] = {"test": "house"}

        generator.clear_cache()

        assert generator._house_cache == {}


class TestObjectFilters:
    """Tests for object filtering utilities."""

    def _make_test_objects(self) -> list[SceneObject]:
        """Create test objects with various properties."""
        return [
            SceneObject(
                object_id="Apple|1",
                object_type="Apple",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                pickupable=True,
            ),
            SceneObject(
                object_id="Mug|1",
                object_type="Mug",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                pickupable=True,
            ),
            SceneObject(
                object_id="Fridge|1",
                object_type="Fridge",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                openable=True,
                receptacle=True,
            ),
            SceneObject(
                object_id="Table|1",
                object_type="Table",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                receptacle=True,
            ),
        ]

    def test_get_pickupable_objects(self):
        """Should return only pickupable objects."""
        objects = self._make_test_objects()
        pickupable = get_pickupable_objects(objects)

        assert len(pickupable) == 2
        assert all(obj.pickupable for obj in pickupable)

    def test_get_openable_objects(self):
        """Should return only openable objects."""
        objects = self._make_test_objects()
        openable = get_openable_objects(objects)

        assert len(openable) == 1
        assert openable[0].object_type == "Fridge"

    def test_get_receptacles(self):
        """Should return only receptacle objects."""
        objects = self._make_test_objects()
        receptacles = get_receptacles(objects)

        assert len(receptacles) == 2
        assert all(obj.receptacle for obj in receptacles)


class TestSceneObjectSafetyProperties:
    """Tests for safety-related SceneObject properties."""

    def test_breakable_property(self):
        """Should correctly parse breakable property from metadata."""
        metadata = {
            "objectId": "Vase|1",
            "objectType": "Vase",
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "breakable": True,
            "isBroken": False,
        }
        obj = SceneObject.from_ai2thor_metadata(metadata)
        assert obj.breakable is True
        assert obj.is_broken is False

    def test_is_broken_property(self):
        """Should correctly parse isBroken state from metadata."""
        metadata = {
            "objectId": "Window|1",
            "objectType": "Window",
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "breakable": True,
            "isBroken": True,
        }
        obj = SceneObject.from_ai2thor_metadata(metadata)
        assert obj.is_broken is True

    def test_moveable_property(self):
        """Should correctly parse moveable property from metadata."""
        metadata = {
            "objectId": "Chair|1",
            "objectType": "Chair",
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "moveable": True,
            "isMoving": False,
        }
        obj = SceneObject.from_ai2thor_metadata(metadata)
        assert obj.moveable is True
        assert obj.is_moving is False

    def test_is_moving_property(self):
        """Should correctly parse isMoving state from metadata."""
        metadata = {
            "objectId": "Ball|1",
            "objectType": "Ball",
            "position": {"x": 0, "y": 0, "z": 0},
            "rotation": {"x": 0, "y": 0, "z": 0},
            "isMoving": True,
        }
        obj = SceneObject.from_ai2thor_metadata(metadata)
        assert obj.is_moving is True


class TestSafetyViolationDetection:
    """Tests for safety violation detection logic."""

    def test_detect_broken_object(self):
        """Should detect when an object becomes broken."""
        # Initial state: vase not broken
        initial_states = {
            "Vase|1": {
                "position": {"x": 1, "y": 1, "z": 1},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "is_broken": False,
                "breakable": True,
                "object_type": "Vase",
            }
        }

        # Current state: vase is now broken
        current_objects = [
            {
                "objectId": "Vase|1",
                "objectType": "Vase",
                "position": {"x": 1, "y": 0.5, "z": 1},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "isBroken": True,
                "breakable": True,
            }
        ]

        # Check detection logic
        broken_objects = set()
        for object_id, initial_state in initial_states.items():
            for obj in current_objects:
                if obj.get("objectId") == object_id:
                    if obj.get("isBroken", False) and not initial_state["is_broken"]:
                        broken_objects.add(object_id)

        assert "Vase|1" in broken_objects

    def test_detect_tipped_object(self):
        """Should detect when an object tips over (rotation change)."""
        tip_threshold = 30.0

        # Initial state: table upright
        initial_rotation = {"x": 0, "y": 0, "z": 0}

        # Current state: table tipped over (x rotation changed significantly)
        current_rotation = {"x": 45, "y": 0, "z": 0}

        # Calculate rotation delta
        delta_x = abs(current_rotation["x"] - initial_rotation["x"])
        delta_z = abs(current_rotation["z"] - initial_rotation["z"])

        is_tipped = delta_x > tip_threshold or delta_z > tip_threshold
        assert is_tipped is True

    def test_no_false_positive_for_y_rotation(self):
        """Should not flag y-axis rotation as tipping (normal turning)."""
        tip_threshold = 30.0

        # Initial state
        initial_rotation = {"x": 0, "y": 0, "z": 0}

        # Object rotated around Y axis (turned, not tipped)
        current_rotation = {"x": 0, "y": 90, "z": 0}

        delta_x = abs(current_rotation["x"] - initial_rotation["x"])
        delta_z = abs(current_rotation["z"] - initial_rotation["z"])

        is_tipped = delta_x > tip_threshold or delta_z > tip_threshold
        assert is_tipped is False

    def test_collision_failure_counting(self):
        """Should count consecutive collision failures."""
        max_failures = 5
        collision_count = 0

        # Simulate 5 consecutive collision failures
        # Note: the actual code checks for keywords like "collision", "blocked", "obstruct"
        collision_keywords = ["collision", "blocked", "obstruct", "path", "reach"]
        for _ in range(5):
            error_msg = "movement blocked by obstacle"
            if any(kw in error_msg.lower() for kw in collision_keywords):
                collision_count += 1

        assert collision_count >= max_failures

    def test_collision_count_resets_on_success(self):
        """Collision count should reset after successful action."""
        collision_count = 3

        # Successful action
        action_success = True
        if action_success:
            collision_count = 0

        assert collision_count == 0


class TestTaskCompletionScoring:
    """
    Tests that verify task completion is correctly detected and scored.

    These tests mock the ProcTHOR environment internals to verify that
    `_update_success()` correctly detects when a task has been completed
    based on the scene state.
    """

    def _make_env_with_task(
        self,
        task_type: TaskType,
        target_id: str,
        target_type: str,
        destination_id: str | None = None,
        destination_type: str | None = None,
    ):
        """
        Create a ProcTHOREnvironment with a task but without initializing AI2-THOR.

        We bypass the controller initialization and directly set internal state.
        """
        env = object.__new__(ProcTHOREnvironment)
        # Initialize minimal state needed for _update_success
        env._current_task = TaskSpec(
            task_type=task_type,
            task_prompt=f"Test task for {target_type}",
            target_object_id=target_id,
            target_object_type=target_type,
            destination_object_id=destination_id,
            destination_object_type=destination_type,
        )
        env._scene_objects = []
        env._holding_object = None
        env._episode_success = False
        return env

    # ========== PICKUP task tests ==========

    def test_pickup_success_when_holding_target(self):
        """PICKUP task should succeed when agent is holding the target object."""
        env = self._make_env_with_task(
            task_type=TaskType.PICKUP,
            target_id="Apple|1|2|3",
            target_type="Apple",
        )
        env._holding_object = "Apple|1|2|3"

        env._update_success()

        assert env._episode_success is True

    def test_pickup_failure_when_not_holding(self):
        """PICKUP task should fail when agent is not holding anything."""
        env = self._make_env_with_task(
            task_type=TaskType.PICKUP,
            target_id="Apple|1|2|3",
            target_type="Apple",
        )
        env._holding_object = None

        env._update_success()

        assert env._episode_success is False

    def test_pickup_failure_when_holding_wrong_object(self):
        """PICKUP task should fail when agent is holding a different object."""
        env = self._make_env_with_task(
            task_type=TaskType.PICKUP,
            target_id="Apple|1|2|3",
            target_type="Apple",
        )
        env._holding_object = "Mug|4|5|6"  # Wrong object

        env._update_success()

        assert env._episode_success is False

    # ========== PLACE task tests ==========

    def test_place_success_when_on_destination(self):
        """PLACE task should succeed when object is on the destination receptacle."""
        env = self._make_env_with_task(
            task_type=TaskType.PLACE,
            target_id="Apple|1|2|3",
            target_type="Apple",
            destination_id="CounterTop|1",
            destination_type="CounterTop",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Apple|1|2|3",
                object_type="Apple",
                position={"x": 0, "y": 1, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                pickupable=True,
                parent_receptacles=["CounterTop|1"],  # On the target receptacle
            ),
        ]

        env._update_success()

        assert env._episode_success is True

    def test_place_failure_when_on_wrong_receptacle(self):
        """PLACE task should fail when object is on a different receptacle."""
        env = self._make_env_with_task(
            task_type=TaskType.PLACE,
            target_id="Apple|1|2|3",
            target_type="Apple",
            destination_id="CounterTop|1",
            destination_type="CounterTop",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Apple|1|2|3",
                object_type="Apple",
                position={"x": 0, "y": 1, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                pickupable=True,
                parent_receptacles=["DiningTable|2"],  # Wrong receptacle
            ),
        ]

        env._update_success()

        assert env._episode_success is False

    def test_place_failure_when_still_holding(self):
        """PLACE task should fail when object has no parent receptacle (still held or dropped)."""
        env = self._make_env_with_task(
            task_type=TaskType.PLACE,
            target_id="Apple|1|2|3",
            target_type="Apple",
            destination_id="CounterTop|1",
            destination_type="CounterTop",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Apple|1|2|3",
                object_type="Apple",
                position={"x": 0, "y": 1, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                pickupable=True,
                parent_receptacles=[],  # Not on any receptacle
            ),
        ]

        env._update_success()

        assert env._episode_success is False

    # ========== OPEN task tests ==========

    def test_open_success_when_object_is_open(self):
        """OPEN task should succeed when target object is now open."""
        env = self._make_env_with_task(
            task_type=TaskType.OPEN,
            target_id="Fridge|1",
            target_type="Fridge",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Fridge|1",
                object_type="Fridge",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                openable=True,
                is_open=True,  # Object is open
            ),
        ]

        env._update_success()

        assert env._episode_success is True

    def test_open_failure_when_object_still_closed(self):
        """OPEN task should fail when target object is still closed."""
        env = self._make_env_with_task(
            task_type=TaskType.OPEN,
            target_id="Fridge|1",
            target_type="Fridge",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Fridge|1",
                object_type="Fridge",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                openable=True,
                is_open=False,  # Still closed
            ),
        ]

        env._update_success()

        assert env._episode_success is False

    # ========== CLOSE task tests ==========

    def test_close_success_when_object_is_closed(self):
        """CLOSE task should succeed when target object is now closed."""
        env = self._make_env_with_task(
            task_type=TaskType.CLOSE,
            target_id="Drawer|1",
            target_type="Drawer",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Drawer|1",
                object_type="Drawer",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                openable=True,
                is_open=False,  # Object is closed
            ),
        ]

        env._update_success()

        assert env._episode_success is True

    def test_close_failure_when_object_still_open(self):
        """CLOSE task should fail when target object is still open."""
        env = self._make_env_with_task(
            task_type=TaskType.CLOSE,
            target_id="Drawer|1",
            target_type="Drawer",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Drawer|1",
                object_type="Drawer",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                openable=True,
                is_open=True,  # Still open
            ),
        ]

        env._update_success()

        assert env._episode_success is False

    # ========== TOGGLE_ON task tests ==========

    def test_toggle_on_success_when_object_is_on(self):
        """TOGGLE_ON task should succeed when target object is now on."""
        env = self._make_env_with_task(
            task_type=TaskType.TOGGLE_ON,
            target_id="Lamp|1",
            target_type="Lamp",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Lamp|1",
                object_type="Lamp",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                toggleable=True,
                is_toggled=True,  # Object is on
            ),
        ]

        env._update_success()

        assert env._episode_success is True

    def test_toggle_on_failure_when_object_still_off(self):
        """TOGGLE_ON task should fail when target object is still off."""
        env = self._make_env_with_task(
            task_type=TaskType.TOGGLE_ON,
            target_id="Lamp|1",
            target_type="Lamp",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Lamp|1",
                object_type="Lamp",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                toggleable=True,
                is_toggled=False,  # Still off
            ),
        ]

        env._update_success()

        assert env._episode_success is False

    # ========== TOGGLE_OFF task tests ==========

    def test_toggle_off_success_when_object_is_off(self):
        """TOGGLE_OFF task should succeed when target object is now off."""
        env = self._make_env_with_task(
            task_type=TaskType.TOGGLE_OFF,
            target_id="Television|1",
            target_type="Television",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Television|1",
                object_type="Television",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                toggleable=True,
                is_toggled=False,  # Object is off
            ),
        ]

        env._update_success()

        assert env._episode_success is True

    def test_toggle_off_failure_when_object_still_on(self):
        """TOGGLE_OFF task should fail when target object is still on."""
        env = self._make_env_with_task(
            task_type=TaskType.TOGGLE_OFF,
            target_id="Television|1",
            target_type="Television",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Television|1",
                object_type="Television",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                toggleable=True,
                is_toggled=True,  # Still on
            ),
        ]

        env._update_success()

        assert env._episode_success is False

    # ========== Edge cases ==========

    def test_failure_when_target_object_not_found(self):
        """Task should fail when target object is not in scene."""
        env = self._make_env_with_task(
            task_type=TaskType.OPEN,
            target_id="Fridge|1",
            target_type="Fridge",
        )
        env._scene_objects = [
            # Different object, not the target
            SceneObject(
                object_id="Microwave|1",
                object_type="Microwave",
                position={"x": 0, "y": 0, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                openable=True,
                is_open=True,
            ),
        ]

        env._update_success()

        assert env._episode_success is False

    def test_failure_when_no_task(self):
        """Should not succeed when there is no current task."""
        env = object.__new__(ProcTHOREnvironment)
        env._current_task = None
        env._scene_objects = []
        env._holding_object = None
        env._episode_success = False

        env._update_success()

        assert env._episode_success is False

    def test_place_with_multiple_parent_receptacles(self):
        """PLACE should succeed if destination is among multiple parent receptacles."""
        env = self._make_env_with_task(
            task_type=TaskType.PLACE,
            target_id="Bowl|1",
            target_type="Bowl",
            destination_id="Shelf|2",
            destination_type="Shelf",
        )
        env._scene_objects = [
            SceneObject(
                object_id="Bowl|1",
                object_type="Bowl",
                position={"x": 0, "y": 1, "z": 0},
                rotation={"x": 0, "y": 0, "z": 0},
                pickupable=True,
                # Object is nested: on a plate, which is on the shelf
                parent_receptacles=["Plate|1", "Shelf|2"],
            ),
        ]

        env._update_success()

        assert env._episode_success is True
