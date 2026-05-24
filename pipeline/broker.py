# pipeline/broker.py
"""
Message broker abstraction with ZeroMQ and Redis backends.

Provides a unified interface for pub/sub messaging regardless of the
underlying transport. The broker pattern decouples publishers and
subscribers, allowing easy switching between backends.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, Callable
import zmq
import config


@dataclass
class PriceTick:
    """A single price update for one asset."""
    ticker: str
    price: float
    volume: float
    timestamp: float       # Unix timestamp
    bid: float = 0.0
    ask: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "PriceTick":
        return cls(**json.loads(s))


class MessageBroker(ABC):
    """Abstract base class for message brokers."""

    @abstractmethod
    def publish(self, topic: str, message: str) -> None:
        """Publish a message on a topic."""
        ...

    @abstractmethod
    def subscribe(self, topics: list[str], callback: Callable[[str, str], None]) -> None:
        """Subscribe to topics and invoke callback(topic, message) on receipt."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...


class ZMQBroker(MessageBroker):
    """
    ZeroMQ-based message broker using PUB/SUB pattern.

    This is the default backend — requires no external services.
    ZMQ handles message queuing, fan-out, and reconnection internally.
    """

    def __init__(self, mode: str = "pub"):
        """
        Parameters
        ----------
        mode : str
            "pub" for publisher mode, "sub" for subscriber mode.
        """
        self.context = zmq.Context()
        self.mode = mode
        self.socket = None

        if mode == "pub":
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(config.ZMQ_PUB_ADDRESS)
            # Give subscribers time to connect
            time.sleep(0.5)
        elif mode == "sub":
            self.socket = self.context.socket(zmq.SUB)
            self.socket.connect(config.ZMQ_SUB_ADDRESS)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'pub' or 'sub'.")

    def publish(self, topic: str, message: str) -> None:
        """Publish a message with a topic prefix."""
        if self.mode != "pub":
            raise RuntimeError("Cannot publish in subscriber mode.")
        self.socket.send_string(f"{topic}|{message}")

    def subscribe(self, topics: list[str],
                  callback: Callable[[str, str], None],
                  timeout_ms: int = 1000) -> None:
        """
        Subscribe to topics and invoke callback on each message.

        This is a blocking call — runs until interrupted.

        Parameters
        ----------
        topics : list[str]
            Topics to subscribe to. Empty list subscribes to all.
        callback : callable
            Function called with (topic, message) for each received message.
        timeout_ms : int
            Poll timeout in milliseconds (allows periodic checking for stop signals).
        """
        if self.mode != "sub":
            raise RuntimeError("Cannot subscribe in publisher mode.")

        for topic in (topics or [""]):
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)

        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        self._running = True
        while self._running:
            events = dict(poller.poll(timeout_ms))
            if self.socket in events:
                raw = self.socket.recv_string()
                parts = raw.split("|", 1)
                if len(parts) == 2:
                    callback(parts[0], parts[1])

    def receive_one(self, timeout_ms: int = 1000) -> Optional[tuple[str, str]]:
        """
        Non-blocking receive of a single message.

        Returns
        -------
        Optional[tuple[str, str]]
            (topic, message) if available, None if timeout.
        """
        if self.socket.poll(timeout_ms):
            raw = self.socket.recv_string()
            parts = raw.split("|", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
        return None

    def stop(self) -> None:
        """Signal the subscribe loop to stop."""
        self._running = False

    def close(self) -> None:
        """Clean up ZMQ resources."""
        self._running = False
        if self.socket:
            self.socket.close()
        self.context.term()


class RedisBroker(MessageBroker):
    """
    Redis-based message broker using Redis Pub/Sub.

    Requires a running Redis server. Provides persistence and
    cross-process communication capabilities beyond ZMQ.
    """

    def __init__(self):
        from infra.redis_config import get_redis_client
        self.client = get_redis_client()
        self.pubsub = None

    def publish(self, topic: str, message: str) -> None:
        """Publish a message to a Redis channel."""
        self.client.publish(topic, message)

    def subscribe(self, topics: list[str],
                  callback: Callable[[str, str], None]) -> None:
        """
        Subscribe to Redis channels and invoke callback on each message.

        This is a blocking call.
        """
        self.pubsub = self.client.pubsub()
        self.pubsub.subscribe(*topics)

        self._running = True
        for msg in self.pubsub.listen():
            if not self._running:
                break
            if msg["type"] == "message":
                callback(msg["channel"], msg["data"])

    def stop(self) -> None:
        """Signal the subscribe loop to stop."""
        self._running = False

    def close(self) -> None:
        """Clean up Redis resources."""
        self._running = False
        if self.pubsub:
            self.pubsub.unsubscribe()
            self.pubsub.close()
        self.client.close()


def create_broker(mode: str = "pub", backend: str = None) -> MessageBroker:
    """
    Factory function to create the appropriate message broker.

    Parameters
    ----------
    mode : str
        "pub" or "sub" (only relevant for ZMQ).
    backend : str, optional
        "zmq" or "redis". Defaults to config.PIPELINE_BACKEND.

    Returns
    -------
    MessageBroker
        Configured broker instance.
    """
    backend = backend or config.PIPELINE_BACKEND

    if backend == "zmq":
        return ZMQBroker(mode=mode)
    elif backend == "redis":
        return RedisBroker()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'zmq' or 'redis'.")
