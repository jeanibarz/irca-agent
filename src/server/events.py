import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventBroadcaster:
    """
    Simple in-memory event broadcaster using asyncio.
    Supports Server-Sent Events (SSE) by managing subscriber queues.
    """

    _instance = None

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    @classmethod
    def get_instance(cls) -> "EventBroadcaster":
        if cls._instance is None:
            cls._instance = EventBroadcaster()
        return cls._instance

    async def subscribe(self) -> Any:
        """
        Yields generator allowing a client to subscribe to the event stream.
        """
        q: asyncio.Queue = asyncio.Queue()
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
            self._subscribers.remove(q)

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Publish an event to all active subscribers.
        """
        if not self._subscribers:
            return

        import json

        # SSE format: "event: type\ndata: json_payload\n\n"
        payload = json.dumps(data)
        message = f"event: {event_type}\ndata: {payload}\n\n"

        for q in self._subscribers:
            q.put_nowait(message)
