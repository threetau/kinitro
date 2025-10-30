import asyncio
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Any, Optional

import capnp
import torch
from ray.util.queue import Queue
from snowflake import SnowflakeGenerator

from evaluator.rpc.client import AgentClient

# RPC timeout constant
DEFAULT_RPC_TIMEOUT = 5.0
# Ray PG queue timeout to avoid indefinite blocking on slow consumers.
PGQ_TIMEOUT = 5.0


class RPCMethod(Enum):
    """Supported RPC methods"""

    PING = "ping"
    ACT = "act"
    RESET = "reset"


@dataclass
class RPCRequest:
    """
    RPC request message from Ray Worker to RPC Process

    Args:
        method: The RPC method to call
        request_id: Unique identifier for this request
        params: Method-specific parameters
        timeout: Optional timeout for this request
    """

    method: RPCMethod
    request_id: str
    params: dict[str, Any]
    timeout: timedelta = timedelta(seconds=DEFAULT_RPC_TIMEOUT)

    @classmethod
    def create_ping(
        cls, message: str, timeout: timedelta = timedelta(seconds=DEFAULT_RPC_TIMEOUT)
    ) -> "RPCRequest":
        """Create a ping request"""
        return cls(
            method=RPCMethod.PING,
            request_id=str(next(SnowflakeGenerator(42))),
            params={"message": message},
            timeout=timeout,
        )

    @classmethod
    def create_act(
        cls, obs: Any, timeout: timedelta = timedelta(seconds=DEFAULT_RPC_TIMEOUT)
    ) -> "RPCRequest":
        """Create an act request with observation payload (array or dict)."""
        return cls(
            method=RPCMethod.ACT,
            request_id=str(next(SnowflakeGenerator(42))),
            params={"obs": obs},
            timeout=timeout,
        )

    @classmethod
    def create_reset(
        cls, timeout: timedelta = timedelta(seconds=DEFAULT_RPC_TIMEOUT)
    ) -> "RPCRequest":
        """Create a reset request"""
        return cls(
            method=RPCMethod.RESET,
            request_id=str(next(SnowflakeGenerator(42))),
            params={},
            timeout=timeout,
        )


@dataclass
class ServerResponse:
    """
    Response data from the actual RPC server (AgentServer)
    This wraps the actual Cap'n Proto response data

    Args:
        method: Which method this response is for
        success: Whether the server call succeeded
        data: The actual response data from the server
        error: Error message if the server call failed
    """

    method: RPCMethod
    success: bool
    data: Any = None
    error: Optional[str] = None

    @classmethod
    def from_ping_response(cls, response_text: str) -> "ServerResponse":
        """Create ServerResponse from ping method result"""
        return cls(method=RPCMethod.PING, success=True, data=response_text)

    @classmethod
    def from_act_response(cls, action_tensor: torch.Tensor) -> "ServerResponse":
        """Create ServerResponse from act method result"""
        return cls(method=RPCMethod.ACT, success=True, data=action_tensor)

    @classmethod
    def from_reset_response(cls) -> "ServerResponse":
        """Create ServerResponse from reset method (no return value)"""
        return cls(method=RPCMethod.RESET, success=True, data=None)

    @classmethod
    def from_error(cls, method: RPCMethod, error_msg: str) -> "ServerResponse":
        """Create ServerResponse for server errors"""
        return cls(method=method, success=False, error=error_msg)


@dataclass
class RPCResponse:
    """
    Final response message from RPC Process back to Ray Worker
    Contains both request correlation and server response data

    Args:
        request_id: ID of the original request from Ray Worker
        server_response: The wrapped response from the RPC server
        processing_error: Any error that occurred in the RPC process itself
    """

    request_id: str
    server_response: Optional[ServerResponse] = None
    processing_error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Overall success: no processing error and server succeeded"""
        return (
            self.processing_error is None
            and self.server_response is not None
            and self.server_response.success
        )

    @property
    def result(self) -> Any:
        """Get the actual result data"""
        if self.server_response and self.server_response.success:
            return self.server_response.data
        return None

    @property
    def error_message(self) -> Optional[str]:
        """Get the combined error message"""
        if self.processing_error:
            return f"Processing error: {self.processing_error}"
        elif self.server_response and not self.server_response.success:
            return f"Server error: {self.server_response.error}"
        return None

    @classmethod
    def from_server_response(
        cls, request_id: str, server_response: ServerResponse
    ) -> "RPCResponse":
        """Create RPCResponse from successful server interaction"""
        return cls(request_id=request_id, server_response=server_response)

    @classmethod
    def from_processing_error(cls, request_id: str, error: str) -> "RPCResponse":
        """Create RPCResponse for RPC process errors"""
        return cls(request_id=request_id, processing_error=error)


class RPCProcess:
    """Handles RPC message processing and server interaction"""

    def __init__(self, host: str, port: int, send_queue: Queue, recv_queue: Queue):
        print(f"Initializing RPCProcess for {host}:{port}")
        self.agent = AgentClient(host, port)
        self.send_queue = send_queue
        self.recv_queue = recv_queue
        self.host = host
        self.port = port
        print(f"Starting RPCProcess for {host}:{port}")
        asyncio.run(capnp.run(self.process()))

    async def handle_request(self, request: RPCRequest) -> RPCResponse:
        try:
            # Call the appropriate server method and wrap response
            match request.method:
                case RPCMethod.PING:
                    server_result = await self.agent.ping(
                        request.params["message"], timeout=request.timeout
                    )
                    server_response = ServerResponse.from_ping_response(server_result)

                case RPCMethod.ACT:
                    obs = request.params["obs"]
                    server_result = await self.agent.act(obs, timeout=request.timeout)
                    server_response = ServerResponse.from_act_response(server_result)

                case RPCMethod.RESET:
                    await self.agent.reset()
                    server_response = ServerResponse.from_reset_response()

                case _:
                    server_response = ServerResponse.from_error(
                        request.method, f"Unknown RPC method: {request.method}"
                    )

            return RPCResponse.from_server_response(request.request_id, server_response)

        except asyncio.TimeoutError:
            error_msg = f"Server call timed out after {request.timeout}s"
            return RPCResponse.from_processing_error(request.request_id, error_msg)

        except Exception as e:
            error_msg = f"Failed to call server: {str(e)}"
            return RPCResponse.from_processing_error(request.request_id, error_msg)

    async def process(self):
        print(f"Starting RPC process for {self.host}:{self.port}")
        try:
            await self.agent.connect()
            # Test connection with structured messages
            startup_request = RPCRequest.create_ping("rpc-process-startup")
            startup_response = await self.handle_request(startup_request)

            if startup_response.success:
                print(
                    f"RPC process for {self.host}:{self.port} connected successfully, server says: {startup_response.result}"
                )
            else:
                print(
                    f"RPC process for {self.host}:{self.port} connection test failed: {startup_response.error_message}"
                )
                return

            while True:
                try:
                    # Check for incoming requests from Ray Worker
                    if self.recv_queue.empty():
                        await asyncio.sleep(0.05)
                        continue

                    # Get request message from Ray Worker
                    request: RPCRequest = await self.recv_queue.get_async(
                        timeout=PGQ_TIMEOUT
                    )
                    print(
                        f"RPC[{self.host}:{self.port}] received: method={request.method.value}, id={request.request_id[:8]}"
                    )

                    # Process the request by calling the RPC server
                    response: RPCResponse = await self.handle_request(request)

                    # Log the result
                    if response.success:
                        result_preview = (
                            str(response.result)[:50]
                            if response.result is not None
                            else "None"
                        )
                        print(
                            f"RPC[{self.host}:{self.port}] success: id={request.request_id[:8]}, result={result_preview}"
                        )
                    else:
                        print(
                            f"RPC[{self.host}:{self.port}] error: id={request.request_id[:8]}, error={response.error_message}"
                        )

                    # Send response back to Ray Worker
                    await self.send_queue.put_async(response, timeout=PGQ_TIMEOUT)

                except Exception as e:
                    print(f"Error processing request in RPC process: {e}")
                    # Send error response if we have request context
                    if "request" in locals():
                        error_response = RPCResponse.from_processing_error(
                            request.request_id, f"Request processing failed: {str(e)}"
                        )
                        await self.send_queue.put_async(
                            error_response, timeout=PGQ_TIMEOUT
                        )
                    break

        except Exception as e:
            print(f"Fatal error in RPC process for {self.host}:{self.port}: {e}")
        finally:
            await self.agent.close()
