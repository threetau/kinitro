import random
import time

import pytest
from kinitro_eval.containers import Containers

# Use a real HuggingFace repo for testing
SUBMISSION_REPO = "https://huggingface.co/rishiad/default_submission"


@pytest.fixture(scope="module")
def containers():
    return Containers()


@pytest.fixture(scope="module")
def submission_id():
    # Use a random 64-bit integer for testing
    return random.getrandbits(64)


def test_create_and_destroy_container(containers, submission_id):
    # Test pod creation
    pod_name = containers.create_container(SUBMISSION_REPO, submission_id)
    assert pod_name.startswith("submission-")
    print(f"Pod created: {pod_name}")

    # Optionally, wait a bit for pod to be fully up
    time.sleep(10)

    # Test pod deletion
    containers.destroy_container(submission_id)
    print(f"Pod deleted: {pod_name}")
