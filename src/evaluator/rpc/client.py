#!/usr/bin/env python3
"""
Cap'n Proto RPC Client for Agent Interface
Sends observations as Agent.Tensor (no pickle).
Handles promise vs coroutine returns, timeouts, and traversal limit.
"""

import asyncio
import logging
import os
from typing import Tuple

import capnp
import numpy as np
import torch

# Load the schema
schema_file = os.path.join(os.path.dirname(__file__), "agent.capnp")
agent_capnp = capnp.load(schema_file)

logger = logging.getLogger(__name__)


class AgentClient:
    """Client for connecting to Agent RPC server. Sends observations as Tensor struct."""

    def __init__(self, host="localhost", port=8000):
        self.address = host
        self.port = port
        self.client = None
        self.agent = None
        self.stream = None

    async def connect(self, timeout: float = 30.0):
        """Connect asynchronously using AsyncIoStream and create a TwoPartyClient with a raised traversal limit."""
        print(f"Connecting to {self.address}:{self.port}")
        if self.agent is not None:
            print(f"Already connected to {self.address}:{self.port}")
            return

        try:
            print(f"Connecting to {self.address}:{self.port}")
            self.stream = await asyncio.wait_for(
                capnp.AsyncIoStream.create_connection(host=self.address, port=self.port),
                timeout=timeout,
            )

            # Create TwoPartyClient with increased traversal limit (workaround for large messages)
            self.client = capnp.TwoPartyClient(self.stream)
            print(f"TwoPartyClient established to {self.address}:{self.port}")

            self.agent = self.client.bootstrap().cast_as(agent_capnp.Agent)
            print(f"Bootstrapped Agent capability")
        except Exception:
            logger.exception("Failed to connect to agent")
            raise

    async def _await_capnp_result(self, maybe_promise, timeout: float):
        """Handle either promise-like (has a_wait) or coroutine-like awaitable from pycapnp."""
        if hasattr(maybe_promise, "a_wait"):
            coro = maybe_promise.a_wait()
        else:
            coro = maybe_promise
        return await asyncio.wait_for(coro, timeout=timeout)

    def _obs_to_tensor_struct(self, obs: np.ndarray):
        """Convert a numpy array observation to a Capnp Tensor struct (client-side builder)."""
        if not hasattr(obs, "tobytes") or not hasattr(obs, "shape") or not hasattr(obs, "dtype"):
            raise TypeError("Observation must be a numpy ndarray-like object")
        msg = agent_capnp.Tensor.new_message()
        msg.data = obs.tobytes()
        msg.shape = [int(s) for s in obs.shape]
        msg.dtype = str(obs.dtype)
        return msg

    async def act(self, obs: np.ndarray, timeout: float = 5.0) -> torch.Tensor:
        """
        Send observation (numpy ndarray) to agent and receive action as torch.Tensor.
        Uses Tensor struct for obs to avoid pickle.
        """
        if self.agent is None:
            await self.connect()

        # Build Tensor message for observation
        try:
            obs_msg = self._obs_to_tensor_struct(obs)
            logger.debug("AgentClient.act: sending obs shape=%s dtype=%s bytes=%d",
                         obs.shape, obs.dtype, obs.tobytes().__len__())
        except Exception:
            logger.exception("Failed to prepare observation tensor")
            raise

        try:
            maybe_promise = self.agent.act(obs_msg)
            logger.debug("AgentClient.act: call returned type=%s has_a_wait=%s",
                         type(maybe_promise), hasattr(maybe_promise, "a_wait"))

            result = await self._await_capnp_result(maybe_promise, timeout=timeout)

            # Expect result to be a Tensor struct
            if not hasattr(result, "action"):
                logger.error("AgentClient.act: unexpected result: %r", result)
                raise RuntimeError("unexpected RPC result (no .action)")

            action_data = result.action.data
            shape = tuple(result.action.shape)
            dtype_str = result.action.dtype

            action_np = np.frombuffer(action_data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
            return torch.from_numpy(action_np)
        except asyncio.TimeoutError:
            logger.error("Agent call timed out after %.1fs", timeout)
            raise
        except Exception:
            logger.exception("Error in AgentClient.act")
            raise

    async def reset(self):
        """Reset the agent (async)."""
        if self.agent is None:
            await self.connect()
        try:
            maybe_promise = self.agent.reset()
            await self._await_capnp_result(maybe_promise, timeout=5.0)
        except Exception:
            logger.exception("Error in reset")
            raise

    async def ping(self, message: str, timeout: float = 5.0) -> str:
        if self.agent is None:
            await self.connect()
        try:
            logger.info(f"Sending ping: {message}")
            maybe = self.agent.ping(message)
            res = await self._await_capnp_result(maybe, timeout=timeout)
            logger.info(f"Ping response received: {res.response}")
            return res.response
        except Exception:
            logger.exception("Ping failed")
            raise

    async def close(self):
        """Close/cleanup client and stream."""
        try:
            if self.client:
                if hasattr(self.client, "close"):
                    close_method = self.client.close
                    if asyncio.iscoroutinefunction(close_method):
                        await close_method()
                    else:
                        close_method()
            # clear refs
            self.client = None
            self.agent = None
            self.stream = None
            logger.info("AgentClient closed")
        except Exception:
            logger.exception("Error during close")
            # clear refs anyway
            self.client = None
            self.agent = None
            self.stream = None
