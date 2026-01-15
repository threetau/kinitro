"""
WebSocket hub for Kinitro backend.

Manages validator WebSocket connections.
Extracted from BackendService for better separation of concerns.
"""

from typing import Dict, List, Optional

from fastapi import WebSocket
from sqlmodel import SQLModel

from core.log import get_logger
from core.messages import SetWeightsMessage

from .models import SS58Address

logger = get_logger(__name__)

ConnectionId = str  # Unique ID for each WebSocket connection


class WebSocketHub:
    """
    Manages validator WebSocket connections.

    This class is responsible for:
    - Registering and unregistering validator connections
    - Broadcasting messages to all connected validators
    - Sending messages to specific validators
    - Tracking connection state
    """

    def __init__(self):
        # WebSocket connections
        self.active_connections: Dict[ConnectionId, WebSocket] = {}
        self.validator_connections: Dict[ConnectionId, SS58Address] = {}

    async def register_validator(
        self, ws: WebSocket, hotkey: SS58Address, connection_id: ConnectionId
    ) -> ConnectionId:
        """Register a new validator WebSocket connection.

        Args:
            ws: The WebSocket connection
            hotkey: The validator's hotkey
            connection_id: Unique identifier for this connection

        Returns:
            The connection ID
        """
        self.active_connections[connection_id] = ws
        self.validator_connections[connection_id] = hotkey
        logger.info(f"Registered validator {hotkey} with connection ID {connection_id}")
        return connection_id

    async def unregister_validator(
        self, connection_id: ConnectionId
    ) -> Optional[SS58Address]:
        """Unregister a validator connection.

        Args:
            connection_id: The connection ID to unregister

        Returns:
            The hotkey of the unregistered validator, or None if not found
        """
        hotkey = self.validator_connections.pop(connection_id, None)
        ws = self.active_connections.pop(connection_id, None)

        if ws:
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket for {connection_id}: {e}")

        if hotkey:
            logger.info(f"Unregistered validator {hotkey} (connection {connection_id})")

        return hotkey

    async def broadcast_message(self, message: str) -> int:
        """Broadcast a message to all connected validators.

        Args:
            message: The message to broadcast (JSON string)

        Returns:
            Number of validators that received the message
        """
        broadcast_count = 0
        failed_connections: List[ConnectionId] = []

        for conn_id, ws in list(self.active_connections.items()):
            try:
                await ws.send_text(message)
                broadcast_count += 1
            except Exception as e:
                logger.error(f"Failed to send to {conn_id}: {e}")
                failed_connections.append(conn_id)

        # Clean up failed connections
        for conn_id in failed_connections:
            self.active_connections.pop(conn_id, None)
            self.validator_connections.pop(conn_id, None)

        return broadcast_count

    async def broadcast_to_validators(self, message: SQLModel) -> int:
        """Broadcast a message object to all connected validators.

        Args:
            message: The message object to broadcast

        Returns:
            Number of validators that received the message
        """
        return await self.broadcast_message(message.model_dump_json())

    async def send_to_validator(self, hotkey: SS58Address, message: SQLModel) -> bool:
        """Send a message to a specific validator by hotkey.

        Args:
            hotkey: The validator's hotkey
            message: The message to send

        Returns:
            True if the message was sent successfully
        """
        # Find connection ID for this hotkey
        conn_id = None
        for cid, hk in self.validator_connections.items():
            if hk == hotkey:
                conn_id = cid
                break

        if not conn_id:
            logger.warning(f"No connection found for validator {hotkey}")
            return False

        ws = self.active_connections.get(conn_id)
        if not ws:
            logger.warning(f"WebSocket not found for connection {conn_id}")
            return False

        try:
            await ws.send_text(message.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send to validator {hotkey}: {e}")
            # Clean up failed connection
            self.active_connections.pop(conn_id, None)
            self.validator_connections.pop(conn_id, None)
            return False

    async def broadcast_weights(self, weights_dict: Dict[int, float]) -> int:
        """Broadcast weight updates to all connected validators.

        Args:
            weights_dict: Mapping of UIDs to weights

        Returns:
            Number of validators that received the weights
        """
        weight_msg = SetWeightsMessage(weights=weights_dict)
        return await self.broadcast_to_validators(weight_msg)

    def get_validator_hotkeys(self) -> List[SS58Address]:
        """Get list of all connected validator hotkeys."""
        return list(dict.fromkeys(self.validator_connections.values()))

    def get_connection_for_hotkey(self, hotkey: SS58Address) -> Optional[ConnectionId]:
        """Get the connection ID for a validator hotkey."""
        for conn_id, hk in self.validator_connections.items():
            if hk == hotkey:
                return conn_id
        return None

    def has_connections(self) -> bool:
        """Check if there are any active connections."""
        return bool(self.active_connections)

    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_hotkey_for_connection(
        self, connection_id: ConnectionId
    ) -> Optional[SS58Address]:
        """Get the hotkey for a connection ID."""
        return self.validator_connections.get(connection_id)

    async def close_all(self) -> None:
        """Close all WebSocket connections."""
        for ws in self.active_connections.values():
            try:
                await ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")

        self.active_connections.clear()
        self.validator_connections.clear()
        logger.info("All WebSocket connections closed")
