"""Tests for ProcTHOR procedural environment."""

import numpy as np

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

        feasible, reason = check_task_feasibility(TaskType.PICKUP, obj)
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

        feasible, reason = check_task_feasibility(TaskType.PICKUP, obj)
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

        feasible, reason = check_task_feasibility(TaskType.OPEN, obj)
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

        feasible, reason = check_task_feasibility(TaskType.OPEN, obj)
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
