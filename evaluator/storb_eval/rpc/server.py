# The agent server runs on the miner container. The host calls these functions

import asyncio
import logging
import pickle

import agent_capnp
import capnp
import numpy as np
import torch

from ..agent_interface import AgentInterface


class AgentServer(agent_capnp.Agent.Server):
    def __init__(self, agent: AgentInterface):
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.logger.info("AgentServer initialized with agent: %s", type(agent).__name__)

    async def act(self, obs, **kwargs):
        try:
            # Deserialize observation from bytes
            observation = pickle.loads(obs)

            # Call the agent's act method
            action_tensor = self.agent.act(observation)

            # Convert to numpy if it's a torch tensor
            if isinstance(action_tensor, torch.Tensor):
                action_numpy = action_tensor.detach().cpu().numpy()
            else:
                action_numpy = np.array(action_tensor)

            # Prepare tensor response
            response = agent_capnp.Agent.Tensor.new_message()
            response.data = action_numpy.tobytes()
            response.shape = list(action_numpy.shape)
            response.dtype = str(action_numpy.dtype)

            return response
        except Exception as e:
            self.logger.error(f"Error in act: {e}", exc_info=True)
            raise

    async def reset(self, **kwargs):
        try:
            self.agent.reset()
        except Exception as e:
            self.logger.error(f"Error in reset: {e}", exc_info=True)
            raise


async def serve(agent: AgentInterface, address="*", port=8000):
    server = capnp.TwoPartyServer(address, port, bootstrap=AgentServer(agent))
    logging.info(f"Agent RPC server listening on {address}:{port}")

    # Keep the server running
    try:
        await server.run_forever()
    finally:
        server.close()


def start_server(agent: AgentInterface, address="*", port=8000):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(serve(agent, address, port))
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
