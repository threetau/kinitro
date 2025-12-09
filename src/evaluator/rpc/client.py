#!/usr/bin/env python3
"""
Cap'n Proto RPC Client for Agent Interface.

Sends observations as a structured Observation message comprised of Tensor entries
so we can transmit both state vectors and image data without relying on pickle.
Handles promise vs coroutine returns, timeouts, and traversal limit.
"""

import asyncio
import os
from datetime import timedelta
from typing import Any

import capnp
import numpy as np
import torch

from core.log import get_logger

# Load the schema
schema_file = os.path.join(os.path.dirname(__file__), "agent.capnp")
agent_capnp = capnp.load(schema_file)

logger = get_logger(__name__)


class AgentClient:
    """Client for connecting to Agent RPC server using Observation/Tensor structs."""

    def __init__(self, host="localhost", port=8000):
        self.address = host
        self.port = port
        self.client = None
        self.agent = None
        self.stream = None

    async def connect(
        self,
        timeout: timedelta = timedelta(seconds=30),
        *,
        max_attempts: int = 5,
        base_retry_delay: timedelta = timedelta(seconds=1),
    ):
        """Connect asynchronously using AsyncIoStream and create a TwoPartyClient with a raised traversal limit."""
        logger.debug("Connecting to %s:%d", self.address, self.port)
        if self.agent is not None:
            logger.debug("Already connected to %s:%d", self.address, self.port)
            return

        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                self.stream = await asyncio.wait_for(
                    capnp.AsyncIoStream.create_connection(
                        host=self.address, port=self.port
                    ),
                    timeout=timeout.total_seconds(),
                )

                # Create TwoPartyClient with increased traversal limit (workaround for large messages)
                self.client = capnp.TwoPartyClient(self.stream)
                logger.debug("TwoPartyClient established to %s:%d", self.address, self.port)

                self.agent = self.client.bootstrap().cast_as(agent_capnp.Agent)
                logger.debug("Bootstrapped Agent capability")
                return
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    logger.exception(
                        "Failed to connect to agent at %s:%s after %d attempts",
                        self.address,
                        self.port,
                        max_attempts,
                    )
                    raise

                retry_delay = base_retry_delay * attempt
                retry_seconds = retry_delay.total_seconds()
                logger.warning(
                    "Attempt %d/%d to connect to agent at %s:%s failed: %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    self.address,
                    self.port,
                    exc,
                    retry_seconds,
                )
                await asyncio.sleep(retry_seconds)

        if last_exc:
            raise last_exc

    async def _await_capnp_result(self, maybe_promise, timeout: timedelta):
        """Handle either promise-like (has a_wait) or coroutine-like awaitable from pycapnp."""
        if hasattr(maybe_promise, "a_wait"):
            coro = maybe_promise.a_wait()
        else:
            coro = maybe_promise
        return await asyncio.wait_for(coro, timeout=timeout.total_seconds())

    def _to_numpy(self, value):
        """Convert supported tensor-like values to contiguous numpy arrays."""
        if isinstance(value, np.ndarray):
            arr = value
        elif isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)

        if arr.dtype == np.object_:
            raise TypeError("Cannot serialize object dtype observations")

        return np.ascontiguousarray(arr)

    def _encode_observation(self, obs):
        """Convert an observation (array or dict of arrays) into capnp Observation struct."""
        observation_msg = agent_capnp.Observation.new_message()

        if isinstance(obs, dict):
            entries = observation_msg.init("entries", len(obs))
            for idx, (key, value) in enumerate(obs.items()):
                tensor_builder = entries[idx].tensor
                entries[idx].key = str(key)
                array = self._to_numpy(value)
                tensor_builder.data = array.tobytes()
                tensor_builder.shape = [int(dim) for dim in array.shape]
                tensor_builder.dtype = str(array.dtype)
        else:
            entries = observation_msg.init("entries", 1)
            entries[0].key = "__value__"
            array = self._to_numpy(obs)
            tensor_builder = entries[0].tensor
            tensor_builder.data = array.tobytes()
            tensor_builder.shape = [int(dim) for dim in array.shape]
            tensor_builder.dtype = str(array.dtype)

        return observation_msg

    async def act(
        self, obs: Any, timeout: timedelta = timedelta(seconds=5)
    ) -> torch.Tensor:
        """
        Send observation (array or dict of arrays) to agent and receive action as torch.Tensor.
        Uses Observation and Tensor structs to avoid pickle.
        """
        if self.agent is None:
            await self.connect()

        # Build Observation message from provided payload
        try:
            obs_msg = self._encode_observation(obs)
            logger.debug(
                "AgentClient.act: sending %d observation entries",
                len(obs_msg.entries),
            )
        except Exception:
            logger.exception("Failed to prepare observation payload")
            raise

        try:
            maybe_promise = self.agent.act(obs_msg)
            logger.debug(
                "AgentClient.act: call returned type=%s has_a_wait=%s",
                type(maybe_promise),
                hasattr(maybe_promise, "a_wait"),
            )

            result = await self._await_capnp_result(maybe_promise, timeout=timeout)

            # Expect result to be a Tensor struct
            if not hasattr(result, "action"):
                logger.error("AgentClient.act: unexpected result: %r", result)
                raise RuntimeError("unexpected RPC result (no .action)")

            action_data = result.action.data
            shape = tuple(result.action.shape)
            dtype_str = result.action.dtype

            action_np = (
                np.frombuffer(action_data, dtype=np.dtype(dtype_str))
                .reshape(shape)
                .copy()
            )
            return torch.from_numpy(action_np)
        except asyncio.TimeoutError:
            logger.error("Agent call timed out after %.1fs", timeout.total_seconds())
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
            await self._await_capnp_result(maybe_promise, timeout=timedelta(seconds=5))
        except Exception:
            logger.exception("Error in reset")
            raise

    async def ping(
        self, message: str, timeout: timedelta = timedelta(seconds=5)
    ) -> str:
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
