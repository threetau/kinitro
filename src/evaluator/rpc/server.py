#!/usr/bin/env python3
"""
Cap'n Proto RPC Server for Agent Interface
Receives observation as Agent.Tensor (no pickle).
"""

import asyncio
import logging
import os

import capnp
import numpy as np
import torch

# Load the schema
schema_file = os.path.join(os.path.dirname(__file__), "agent.capnp")
agent_capnp = capnp.load(schema_file)

logger = logging.getLogger(__name__)

_TRAVERSAL_WORDS = 100 * 1024 * 1024  # match client; tune appropriately


class AgentServer(agent_capnp.Agent.Server):
    """Cap'n Proto server implementation for AgentInterface"""

    def __init__(self, agent):
        self.agent = agent
        self.logger = logging.getLogger(__name__)
        self.logger.info("AgentServer initialized with agent: %s", type(agent).__name__)

    async def act(self, obs, **kwargs):
        """Handle act RPC call. 'obs' is expected to be an Agent.Tensor struct."""
        try:
            # obs is a struct with .data, .shape, .dtype
            byte_len = len(obs.data) if obs and obs.data is not None else 0
            self.logger.debug("Server.act invoked; incoming obs bytes=%d shape=%s dtype=%s",
                              byte_len, list(obs.shape) if obs else None, obs.dtype if obs else None)

            # reconstruct numpy observation
            obs_np = np.frombuffer(obs.data, dtype=np.dtype(obs.dtype)).reshape(tuple(obs.shape))

            # call the underlying agent synchronously (user's agent.act should accept ndarray)
            action_tensor = self.agent.act(obs_np)

            # convert to numpy
            if isinstance(action_tensor, torch.Tensor):
                action_np = action_tensor.detach().cpu().numpy()
            else:
                action_np = np.array(action_tensor)

            # Build response Tensor
            response = agent_capnp.Tensor.new_message()
            response.data = action_np.tobytes()
            response.shape = [int(s) for s in action_np.shape]
            response.dtype = str(action_np.dtype)
            return response
        except Exception:
            self.logger.exception("Exception in AgentServer.act")
            raise

    async def reset(self, **kwargs):
        try:
            self.agent.reset()
        except Exception:
            self.logger.exception("Error in reset")
            raise

    async def ping(self, message, **kwargs):
        self.logger.info(f"Ping received: {message}")
        return "pong"


async def serve(agent, address="127.0.0.1", port=8000):
    """Serve the agent using asyncio approach"""

    async def new_connection(stream):
        try:
            server = capnp.TwoPartyServer(stream, bootstrap=AgentServer(agent),
                                          traversal_limit_in_words=_TRAVERSAL_WORDS)
            await server.on_disconnect()
        except Exception:
            logger.exception("Error handling connection")

    server = await capnp.AsyncIoStream.create_server(new_connection, address, port)
    logger.info("Agent RPC server listening on %s:%d", address, port)

    try:
        async with server:
            await server.serve_forever()
    except Exception:
        logger.exception("Server error")
    finally:
        logger.info("Server shutting down")


def start_server(agent, address="127.0.0.1", port=8000):
    async def run_server_with_kj():
        async with capnp.kj_loop():
            await serve(agent, address, port)

    try:
        asyncio.run(run_server_with_kj())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


def run_server_in_process(agent, address="127.0.0.1", port=8000):
    async def run_with_kj():
        async with capnp.kj_loop():
            await serve(agent, address, port)

    asyncio.run(run_with_kj())
