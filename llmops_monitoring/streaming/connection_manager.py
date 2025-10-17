"""
WebSocket connection manager.

Manages active WebSocket connections and broadcasting.
"""

import asyncio
import json
from typing import Dict, Set, Optional, Any
from datetime import datetime
from uuid import uuid4

from llmops_monitoring.schema.events import MetricEvent
from llmops_monitoring.utils.logging_config import get_logger


logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections and event broadcasting.

    Features:
    - Track active connections
    - Broadcast events to all or filtered connections
    - Handle connection lifecycle
    - Support subscription filters
    """

    def __init__(self):
        """Initialize connection manager."""
        # Store connections: {connection_id: WebSocket}
        self.active_connections: Dict[str, Any] = {}

        # Store filters: {connection_id: filter_dict}
        self.connection_filters: Dict[str, Dict[str, Any]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: Any,
        filters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a new WebSocket connection.

        Args:
            websocket: WebSocket instance
            filters: Optional filters for this connection

        Returns:
            Connection ID
        """
        connection_id = str(uuid4())

        async with self._lock:
            await websocket.accept()
            self.active_connections[connection_id] = websocket

            if filters:
                self.connection_filters[connection_id] = filters

            logger.info(f"WebSocket connected: {connection_id} (filters: {filters})")

        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connection",
                "connection_id": connection_id,
                "message": "Connected to LLMOps monitoring stream",
                "filters": filters or {},
                "timestamp": datetime.utcnow().isoformat()
            },
            websocket
        )

        return connection_id

    async def disconnect(self, connection_id: str):
        """
        Unregister a WebSocket connection.

        Args:
            connection_id: Connection ID to remove
        """
        async with self._lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            if connection_id in self.connection_filters:
                del self.connection_filters[connection_id]

            logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: Any):
        """
        Send message to specific WebSocket.

        Args:
            message: Message to send
            websocket: WebSocket instance
        """
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connections.

        Args:
            message: Message to broadcast
        """
        disconnected = []

        for connection_id, websocket in list(self.active_connections.items()):
            try:
                # Check if connection has filters
                if connection_id in self.connection_filters:
                    if not self._matches_filter(message, self.connection_filters[connection_id]):
                        continue

                await websocket.send_text(json.dumps(message, default=str))

            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)

        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)

    async def broadcast_event(self, event: MetricEvent):
        """
        Broadcast a metric event to all matching connections.

        Args:
            event: MetricEvent to broadcast
        """
        message = {
            "type": "event",
            "data": {
                "event_id": str(event.event_id),
                "session_id": event.session_id,
                "trace_id": event.trace_id,
                "span_id": event.span_id,
                "parent_span_id": event.parent_span_id,
                "operation_name": event.operation_name,
                "operation_type": event.operation_type,
                "timestamp": event.timestamp.isoformat() if event.timestamp else None,
                "duration_ms": event.duration_ms,
                "error": event.error,
                "error_type": event.error_type,
                "text_metrics": event.text_metrics.model_dump() if event.text_metrics else None,
                "image_metrics": event.image_metrics.model_dump() if event.image_metrics else None,
                "custom_attributes": event.custom_attributes or {}
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.broadcast(message)

    def _matches_filter(self, message: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if message matches connection filters.

        Args:
            message: Message to check
            filters: Filter criteria

        Returns:
            True if message matches filters
        """
        if message.get("type") != "event":
            return True  # Always send non-event messages

        event_data = message.get("data", {})

        # Check session_id filter
        if "session_id" in filters:
            if event_data.get("session_id") != filters["session_id"]:
                return False

        # Check operation_name filter
        if "operation_name" in filters:
            if event_data.get("operation_name") != filters["operation_name"]:
                return False

        # Check operation_type filter
        if "operation_type" in filters:
            if event_data.get("operation_type") != filters["operation_type"]:
                return False

        # Check error_only filter
        if filters.get("error_only"):
            if not event_data.get("error"):
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "active_connections": len(self.active_connections),
            "filtered_connections": len(self.connection_filters),
            "connection_ids": list(self.active_connections.keys())
        }
