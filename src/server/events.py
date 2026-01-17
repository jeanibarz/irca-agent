import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# FM-06: Maximum queue size per subscriber to prevent memory exhaustion
MAX_SUBSCRIBER_QUEUE_SIZE = 100


class EventBroadcaster:
    """
    Simple in-memory event broadcaster using asyncio.
    Supports Server-Sent Events (SSE) by managing subscriber queues.
    """

    _instance = None
    _instance_lock = threading.Lock()  # FM-04: Thread-safe singleton

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()
        self._subscribers_lock = threading.Lock()  # FM-33: Protect set mutations

    @classmethod
    def get_instance(cls) -> "EventBroadcaster":
        """Get singleton instance with thread-safe initialization (FM-04)."""
        if cls._instance is None:
            with cls._instance_lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = EventBroadcaster()
        return cls._instance

    async def subscribe(self) -> Any:
        """
        Yields generator allowing a client to subscribe to the event stream.
        """
        # FM-06: Use bounded queue to prevent memory exhaustion from slow consumers
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_SUBSCRIBER_QUEUE_SIZE)
        with self._subscribers_lock:  # FM-33: Thread-safe add
            self._subscribers.add(q)
        try:
            while True:
                # Get message from queue
                # This will block until a message is published
                msg = await q.get()
                # Yield in SSE format
                yield msg
        except asyncio.CancelledError:
            logger.info("Subscriber disconnected")
            raise
        finally:
            with self._subscribers_lock:  # FM-33: Thread-safe remove
                self._subscribers.discard(q)  # Use discard to avoid KeyError if already removed

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Publish an event to all active subscribers.
        """
        # FM-33: Take snapshot of subscribers to avoid iteration issues
        with self._subscribers_lock:
            subscribers_snapshot = list(self._subscribers)

        if not subscribers_snapshot:
            return

        import json

        # SSE format: "event: type\ndata: json_payload\n\n"
        payload = json.dumps(data)
        message = f"event: {event_type}\ndata: {payload}\n\n"

        for q in subscribers_snapshot:
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                # FM-06: Drop message for slow consumer rather than blocking/crashing
                logger.warning("Subscriber queue full, dropping message")
