#!/usr/bin/env python3
"""
Cap'n Proto RPC Client for Agent Interface
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


class AgentClient:
    """Client for connecting to Agent RPC server"""

    def __init__(self, host="localhost", port=8000):
        self.address = host
        self.port = port
        self.client = None
        self.agent = None
        self.stream = None

    async def connect(self):
        """Connect asynchronously using AsyncIoStream"""
        try:
            # Create the AsyncIoStream connection
            self.stream = await capnp.AsyncIoStream.create_connection(
                host=self.address, port=self.port
            )

            # Create TwoPartyClient with the stream
            self.client = capnp.TwoPartyClient(self.stream)

            # Get the bootstrap capability
            self.agent = self.client.bootstrap().cast_as(agent_capnp.Agent)

            logger.info(f"Connected to {self.address}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def act(self, obs) -> torch.Tensor:
        """Send observation to agent and get action back"""
        if self.agent is None:
            await self.connect()

        try:
            # Serialize observation
            serialized_obs = pickle.dumps(obs)

            # Make RPC call
            result = await self.agent.act(serialized_obs)

            # Extract tensor data
            action_data = result.action.data
            shape = result.action.shape
            dtype_str = result.action.dtype

            # Convert back to numpy and then torch
            action_np = np.frombuffer(action_data, dtype=np.dtype(dtype_str)).reshape(
                shape
            )
            # Copy the array to make it writable (fixes PyTorch warning)
            action_np = action_np.copy()
            return torch.from_numpy(action_np)
        except Exception as e:
            logger.error(f"Error in act: {e}")
            raise

    async def reset(self):
        """Reset the agent"""
        if self.agent is None:
            await self.connect()

        try:
            await self.agent.reset()
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            raise

    async def close(self):
        """Close the connection"""
        try:
            if self.client:
                # Note: client.close() might not be awaitable in all versions
                if hasattr(self.client, "close"):
                    close_method = self.client.close
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
                self.client = None
                self.agent = None
                self.stream = None
        except Exception as e:
            logger.warning(f"Error during close: {e}")
            # Still reset the references
            self.client = None
            self.agent = None
            self.stream = None
