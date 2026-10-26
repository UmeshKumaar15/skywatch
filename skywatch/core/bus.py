"""Asynchronous publish/subscribe event bus.

Subsystems never call each other directly; they publish and subscribe to
topics. ``InMemoryBus`` runs everything inside one asyncio loop. Its
``publish`` awaits subscribers in registration order, which keeps the
pipeline deterministic for a given seed. Any broker (Redis Streams, NATS,
Kafka...) can be substituted by implementing ``EventBus``.
"""
from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable, Union

Handler = Callable[[Any], Union[Awaitable[None], None]]

# Canonical topic names
TOPIC_OBSERVATIONS = "sensor.observations"  # payload: list[Observation] (one sensor scan)
TOPIC_TRACKS = "tracking.tracks"  # payload: TrackerFrame
TOPIC_ENRICHED = "analytics.enriched"  # payload: SceneSnapshot dict
TOPIC_INSIGHTS = "analytics.insights"  # payload: list[Insight]


class EventBus:
    def subscribe(self, topic: str, handler: Handler) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    async def publish(self, topic: str, payload: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryBus(EventBus):
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        if handler in self._subs.get(topic, []):
            self._subs[topic].remove(handler)

    async def publish(self, topic: str, payload: Any) -> None:
        self.counters[topic] += 1
        for handler in list(self._subs.get(topic, [])):
            res = handler(payload)
            if inspect.isawaitable(res):
                await res


class QueuedSubscriber:
    """Decouples a slow consumer (e.g. a websocket) from the pipeline.

    Keeps only the newest ``maxsize`` messages so a slow client never
    back-pressures the perception loop.
    """

    def __init__(self, maxsize: int = 4) -> None:
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    def __call__(self, payload: Any) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover
                pass
        self.queue.put_nowait(payload)
