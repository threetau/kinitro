"""Tests for WebSocketHub component."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.websocket_hub import WebSocketHub


class TestWebSocketHub:
    """Tests for WebSocketHub."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hub = WebSocketHub()

    @pytest.mark.asyncio
    async def test_register_validator(self):
        """Test registering a validator connection."""
        ws = AsyncMock()
        hotkey = "test_hotkey_12345"
        conn_id = "conn_1"

        result = await self.hub.register_validator(ws, hotkey, conn_id)

        assert result == conn_id
        assert conn_id in self.hub.active_connections
        assert self.hub.active_connections[conn_id] == ws
        assert conn_id in self.hub.validator_connections
        assert self.hub.validator_connections[conn_id] == hotkey

    @pytest.mark.asyncio
    async def test_unregister_validator(self):
        """Test unregistering a validator connection."""
        ws = AsyncMock()
        hotkey = "test_hotkey_12345"
        conn_id = "conn_1"

        await self.hub.register_validator(ws, hotkey, conn_id)
        result = await self.hub.unregister_validator(conn_id)

        assert result == hotkey
        assert conn_id not in self.hub.active_connections
        assert conn_id not in self.hub.validator_connections
        ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_connection(self):
        """Test unregistering a connection that doesn't exist."""
        result = await self.hub.unregister_validator("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting a message to all validators."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await self.hub.register_validator(ws1, "hotkey1", "conn_1")
        await self.hub.register_validator(ws2, "hotkey2", "conn_2")

        count = await self.hub.broadcast_message('{"test": "message"}')

        assert count == 2
        ws1.send_text.assert_called_once_with('{"test": "message"}')
        ws2.send_text.assert_called_once_with('{"test": "message"}')

    @pytest.mark.asyncio
    async def test_broadcast_message_handles_failures(self):
        """Test that broadcast handles send failures gracefully."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws1.send_text.side_effect = Exception("Connection closed")

        await self.hub.register_validator(ws1, "hotkey1", "conn_1")
        await self.hub.register_validator(ws2, "hotkey2", "conn_2")

        count = await self.hub.broadcast_message('{"test": "message"}')

        # Only ws2 should have succeeded
        assert count == 1
        # Failed connection should be removed
        assert "conn_1" not in self.hub.active_connections
        assert "conn_1" not in self.hub.validator_connections

    @pytest.mark.asyncio
    async def test_send_to_validator(self):
        """Test sending a message to a specific validator."""
        ws = AsyncMock()
        hotkey = "test_hotkey"
        conn_id = "conn_1"

        await self.hub.register_validator(ws, hotkey, conn_id)

        message = MagicMock()
        message.model_dump_json.return_value = '{"test": "message"}'

        result = await self.hub.send_to_validator(hotkey, message)

        assert result is True
        ws.send_text.assert_called_once_with('{"test": "message"}')

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_validator(self):
        """Test sending to a validator that doesn't exist."""
        message = MagicMock()
        message.model_dump_json.return_value = '{"test": "message"}'

        result = await self.hub.send_to_validator("nonexistent", message)

        assert result is False

    def test_get_validator_hotkeys(self):
        """Test getting list of connected validator hotkeys."""
        # Manually add connections for sync test
        self.hub.validator_connections = {
            "conn_1": "hotkey1",
            "conn_2": "hotkey2",
            "conn_3": "hotkey1",  # Duplicate hotkey
        }

        hotkeys = self.hub.get_validator_hotkeys()

        # Should deduplicate hotkeys
        assert len(hotkeys) == 2
        assert "hotkey1" in hotkeys
        assert "hotkey2" in hotkeys

    def test_has_connections(self):
        """Test checking if there are active connections."""
        assert self.hub.has_connections() is False

        self.hub.active_connections["conn_1"] = MagicMock()
        assert self.hub.has_connections() is True

    def test_connection_count(self):
        """Test getting connection count."""
        assert self.hub.connection_count() == 0

        self.hub.active_connections["conn_1"] = MagicMock()
        self.hub.active_connections["conn_2"] = MagicMock()
        assert self.hub.connection_count() == 2

    def test_get_hotkey_for_connection(self):
        """Test getting hotkey for a connection ID."""
        self.hub.validator_connections["conn_1"] = "hotkey1"

        assert self.hub.get_hotkey_for_connection("conn_1") == "hotkey1"
        assert self.hub.get_hotkey_for_connection("nonexistent") is None

    def test_get_connection_for_hotkey(self):
        """Test getting connection ID for a hotkey."""
        self.hub.validator_connections["conn_1"] = "hotkey1"
        self.hub.validator_connections["conn_2"] = "hotkey2"

        assert self.hub.get_connection_for_hotkey("hotkey1") == "conn_1"
        assert self.hub.get_connection_for_hotkey("nonexistent") is None

    @pytest.mark.asyncio
    async def test_close_all(self):
        """Test closing all connections."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await self.hub.register_validator(ws1, "hotkey1", "conn_1")
        await self.hub.register_validator(ws2, "hotkey2", "conn_2")

        await self.hub.close_all()

        assert len(self.hub.active_connections) == 0
        assert len(self.hub.validator_connections) == 0
        ws1.close.assert_called_once()
        ws2.close.assert_called_once()
