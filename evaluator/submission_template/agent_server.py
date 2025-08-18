#!/usr/bin/env python3
"""
Cap'n Proto RPC Server for Agent Interface
"""

import asyncio
import logging
import os
import pickle

import capnp
import numpy as np
import torch

# Load the schema
schema_file = os.path.join(os.path.dirname(__file__), "agent.capnp")
agent_capnp = capnp.load(schema_file)

logger = logging.getLogger(__name__)


class AgentServer(agent_capnp.Agent.Server):
    """Cap'n Proto server implementation for AgentInterface"""

    def __init__(self, agent):
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.logger.info("AgentServer initialized with agent: %s", type(agent).__name__)

    async def act(self, obs, **kwargs):
        """Handle act RPC call"""
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
        """Handle reset RPC call"""
        try:
            self.agent.reset()
        except Exception as e:
            self.logger.error(f"Error in reset: {e}", exc_info=True)
            raise


async def serve(agent, address="127.0.0.1", port=8000):
    """Serve the agent using asyncio approach"""

    async def new_connection(stream):
        """Handler for each new client connection"""
        try:
            # Create TwoPartyServer for this connection
            server = capnp.TwoPartyServer(stream, bootstrap=AgentServer(agent))

            # Wait for the connection to disconnect
            await server.on_disconnect()

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)

    # Create the server
    server = await capnp.AsyncIoStream.create_server(new_connection, address, port)

    logger.info(f"Agent RPC server listening on {address}:{port}")

    try:
        # Keep the server running
        async with server:
            await server.serve_forever()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        logger.info("Server shutting down")


def start_server(agent, address="127.0.0.1", port=8000):
    """Start server with proper asyncio event loop handling"""

    async def run_server_with_kj():
        async with capnp.kj_loop():
            await serve(agent, address, port)

    try:
        asyncio.run(run_server_with_kj())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


def run_server_in_process(agent, address="127.0.0.1", port=8000):
    """Entry point for running server in a separate process"""

    async def run_with_kj():
        async with capnp.kj_loop():
            await serve(agent, address, port)

    asyncio.run(run_with_kj())
