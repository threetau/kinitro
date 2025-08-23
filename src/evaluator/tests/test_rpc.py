#!/usr/bin/env python3
"""
Simple RPC test script for testing the agent RPC server and client.
"""

import asyncio
import logging
import multiprocessing as mp
import time

import capnp
import numpy as np
import torch

# Import our modules
from kinitro_eval.rpc.client import AgentClient
from kinitro_eval.rpc.server import run_server_in_process

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test port
TEST_PORT = 45678


# Mock AgentInterface for testing
class AgentInterface:
    def __init__(self, **kwargs):
        # Mock action space
        self.action_space = type("MockSpace", (), {"shape": (4,)})()

    def act(self, obs, **kwargs):
        raise NotImplementedError

    def reset(self):
        pass


class TestAgent(AgentInterface):
    """Test agent implementation that echoes back tensors or returns default values"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("TestAgent initialized with action_space: %s", self.action_space)
        self.calls = 0

    def act(self, obs, **kwargs):
        self.calls += 1
        logger.info(f"Agent.act called {self.calls} times")

        if isinstance(obs, dict) and "echo" in obs:
            logger.info(f"Echo request received with shape: {obs['echo'].shape}")
            return obs["echo"]
        else:
            logger.info("Standard observation received, returning default action")
            shape = getattr(self.action_space, "shape", (4,))
            return torch.zeros(shape, dtype=torch.float32)

    def reset(self):
        logger.info("Agent reset called")
        self.calls = 0


def run_server(port):
    """Run the RPC server with our test agent"""
    agent = TestAgent()
    logger.info(f"Starting agent server on port {port}")
    run_server_in_process(agent, "127.0.0.1", port)


async def run_client_tests():
    """Run a series of tests on the client side"""
    client = AgentClient("127.0.0.1", TEST_PORT)

    try:
        logger.info("Connecting to server...")
        await client.connect()
        logger.info("Connected successfully")

        # Test 1: Basic action
        logger.info("\n----- Test 1: Basic action -----")
        obs = {"state": np.random.random(10)}
        await client.reset()
        action = await client.act(obs)
        logger.info(f"Received action: {action}")
        assert isinstance(action, torch.Tensor)
        assert action.shape == (4,)
        assert torch.all(action == 0.0)

        # Test 2: Echo test with small tensor
        logger.info("\n----- Test 2: Echo small tensor -----")
        echo_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        obs = {"echo": echo_tensor}
        action = await client.act(obs)
        logger.info(f"Echo result: {action}")
        assert torch.allclose(action, echo_tensor)

        # Test 3: Echo test with larger tensor
        logger.info("\n----- Test 3: Echo larger tensor -----")
        echo_tensor = torch.randn(10, 10)
        obs = {"echo": echo_tensor}
        action = await client.act(obs)
        logger.info(f"Large echo result shape: {action.shape}")
        assert torch.allclose(action, echo_tensor)

        # Test 4: Reset
        logger.info("\n----- Test 4: Reset -----")
        await client.reset()
        logger.info("Reset completed")

        # Test 5: Multiple sequential calls
        logger.info("\n----- Test 5: Multiple calls -----")
        for i in range(3):
            action = await client.act({"iteration": i})
            logger.info(f"Call {i + 1} result: {action}")

        logger.info("\nAll tests passed successfully!")
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False
    finally:
        logger.info("Closing client connection")
        await client.close()


def main():
    """Main function to run the tests"""
    # Start server in a separate process
    logger.info("Starting server process")
    server_process = mp.Process(target=run_server, args=(TEST_PORT,), daemon=True)
    server_process.start()

    try:
        logger.info("Waiting for server to start...")
        time.sleep(3.0)  # Give server time to start

        # Run client tests inside KJ loop
        async def run_with_kj():
            async with capnp.kj_loop():
                return await run_client_tests()

        success = asyncio.run(run_with_kj())

        if success:
            logger.info("✅ All tests completed successfully!")
        else:
            logger.error("❌ Tests failed")

    finally:
        # Clean up server process
        logger.info("Cleaning up server process")
        server_process.terminate()
        server_process.join(timeout=2)
        if server_process.is_alive():
            logger.warning("Server process did not terminate, forcing kill")
            server_process.kill()


if __name__ == "__main__":
    main()
