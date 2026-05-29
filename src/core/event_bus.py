"""
src/core/event_bus.py
=====================
Silent Sentinel — Edge AI Research Platform
Thread-safe, fault-tolerant Event Bus (Observer Pattern).

Architecture
------------
``EventBus`` is a synchronous, in-process message broker.  It intentionally
avoids asyncio or external queues so it can run on constrained edge hardware
without additional runtime dependencies.

Fault isolation
---------------
A misbehaving listener (unhandled exception, timeout, etc.) is caught and
logged *per-listener* so that the remaining subscribers for an event type
still execute.  The faulting listener is **not** automatically unsubscribed;
operators can inspect logs and decide whether to remove it programmatically.

Thread safety
-------------
``subscribe`` and ``unsubscribe`` mutate the internal registry; both are
guarded by a ``threading.Lock``.  ``publish`` acquires the same lock only
long enough to snapshot the current listener list, then releases it before
invoking callbacks — preventing deadlocks when a listener itself publishes.

Usage example
-------------
>>> from src.core.event_bus import EventBus
>>> from src.core.events import AcousticAnomalyEvent
>>>
>>> bus = EventBus()
>>>
>>> def on_anomaly(event: AcousticAnomalyEvent) -> None:
...     print(f"Anomaly at confidence={event.confidence:.2f}")
...
>>> bus.subscribe(AcousticAnomalyEvent, on_anomaly)
>>> bus.publish(AcousticAnomalyEvent(confidence=0.91, features=(0.1, 0.4, 0.9)))
Anomaly at confidence=0.91
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable, Type

from src.core.events import BaseEvent

logger = logging.getLogger(__name__)

# Type alias: any zero-or-one-arg callable that accepts a BaseEvent subclass.
ListenerCallback = Callable[[BaseEvent], None]


class EventBus:
    """Synchronous, thread-safe event bus implementing the Observer Pattern.

    Each event type maps to an ordered list of ``ListenerCallback`` functions.
    Listeners are invoked in subscription order; a failure in one does **not**
    prevent subsequent listeners from running.

    Attributes
    ----------
    _lock:
        Reentrant mutex protecting all mutations to ``_listeners``.
    _listeners:
        Registry mapping ``Type[BaseEvent]`` → ``list[ListenerCallback]``.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._listeners: dict[Type[BaseEvent], list[ListenerCallback]] = defaultdict(
            list
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: Type[BaseEvent],
        callback: ListenerCallback,
    ) -> None:
        """Register *callback* to be invoked whenever *event_type* is published.

        Duplicate subscriptions (same type + same callable object) are silently
        ignored to prevent double-firing during hot-reload or retry scenarios.

        Parameters
        ----------
        event_type:
            The ``BaseEvent`` subclass this listener is interested in.
        callback:
            A callable that accepts a single argument — an instance of
            *event_type*.  It must not raise; any exception will be caught,
            logged, and swallowed by ``publish``.

        Raises
        ------
        TypeError
            If *event_type* is not a subclass of ``BaseEvent``, or
            *callback* is not callable.
        """
        self._validate_event_type(event_type)
        if not callable(callback):
            raise TypeError(
                f"callback must be callable; got {type(callback).__name__!r}."
            )

        with self._lock:
            if callback not in self._listeners[event_type]:
                self._listeners[event_type].append(callback)
                logger.debug(
                    "Listener subscribed.",
                    extra={
                        "event_type": event_type.__name__,
                        "callback": _callback_name(callback),
                    },
                )
            else:
                logger.debug(
                    "Duplicate subscription ignored.",
                    extra={
                        "event_type": event_type.__name__,
                        "callback": _callback_name(callback),
                    },
                )

    def unsubscribe(
        self,
        event_type: Type[BaseEvent],
        callback: ListenerCallback,
    ) -> None:
        """Remove a previously registered *callback* for *event_type*.

        No-ops silently if the (event_type, callback) pair is not registered,
        so callers do not need to guard ``unsubscribe`` calls.

        Parameters
        ----------
        event_type:
            The event type from which to remove the listener.
        callback:
            The exact callable object that was passed to ``subscribe``.

        Raises
        ------
        TypeError
            If *event_type* is not a subclass of ``BaseEvent``.
        """
        self._validate_event_type(event_type)

        with self._lock:
            listeners = self._listeners.get(event_type, [])
            try:
                listeners.remove(callback)
                logger.debug(
                    "Listener unsubscribed.",
                    extra={
                        "event_type": event_type.__name__,
                        "callback": _callback_name(callback),
                    },
                )
            except ValueError:
                logger.debug(
                    "unsubscribe called for unregistered listener — no-op.",
                    extra={
                        "event_type": event_type.__name__,
                        "callback": _callback_name(callback),
                    },
                )

    def publish(self, event: BaseEvent) -> None:
        """Dispatch *event* to all listeners registered for its type.

        Listeners are invoked synchronously in subscription order.  A snapshot
        of the listener list is taken *before* iteration so that listeners that
        call ``subscribe`` or ``unsubscribe`` during dispatch do not affect the
        current delivery round.

        Any exception raised by a listener is caught, logged at ``ERROR``
        level, and swallowed.  The remaining listeners for this event still
        execute.

        Parameters
        ----------
        event:
            The event instance to dispatch.  Must be a ``BaseEvent`` subclass.

        Raises
        ------
        TypeError
            If *event* is not an instance of ``BaseEvent``.
        """
        if not isinstance(event, BaseEvent):
            raise TypeError(
                f"publish expects a BaseEvent instance; "
                f"got {type(event).__name__!r}."
            )

        event_type = type(event)

        # Snapshot: release the lock before calling user-supplied callbacks.
        with self._lock:
            listeners = list(self._listeners.get(event_type, []))

        if not listeners:
            logger.debug(
                "Event published with no listeners.",
                extra={"event_type": event_type.__name__},
            )
            return

        logger.debug(
            "Publishing event.",
            extra={
                "event_type": event_type.__name__,
                "listener_count": len(listeners),
                "event_timestamp": event.timestamp,
            },
        )

        for callback in listeners:
            try:
                callback(event)
            except Exception:  # noqa: BLE001
                # Broad catch is intentional: a misbehaving listener must
                # never propagate into the bus or starve subsequent listeners.
                logger.exception(
                    "Listener raised an unhandled exception; skipping.",
                    extra={
                        "event_type": event_type.__name__,
                        "callback": _callback_name(callback),
                    },
                )

    def listener_count(self, event_type: Type[BaseEvent]) -> int:
        """Return the number of listeners currently registered for *event_type*.

        Primarily intended for unit-test assertions and health-check endpoints.

        Parameters
        ----------
        event_type:
            The event type to query.
        """
        with self._lock:
            return len(self._listeners.get(event_type, []))

    def clear(self, event_type: Type[BaseEvent] | None = None) -> None:
        """Remove all listeners, optionally scoped to a single *event_type*.

        Parameters
        ----------
        event_type:
            When supplied, only listeners for this type are removed.
            When ``None`` (default), **all** listeners for **all** types are
            removed.  Use with caution in production; intended for teardown
            in tests.
        """
        with self._lock:
            if event_type is None:
                self._listeners.clear()
                logger.debug("EventBus cleared — all listeners removed.")
            else:
                self._validate_event_type(event_type)
                self._listeners.pop(event_type, None)
                logger.debug(
                    "EventBus cleared for event type.",
                    extra={"event_type": event_type.__name__},
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_event_type(event_type: object) -> None:
        """Raise ``TypeError`` if *event_type* is not a ``BaseEvent`` subclass."""
        if not (isinstance(event_type, type) and issubclass(event_type, BaseEvent)):
            raise TypeError(
                f"event_type must be a subclass of BaseEvent; "
                f"got {event_type!r}."
            )


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _callback_name(callback: ListenerCallback) -> str:
    """Return a human-readable identifier for a callback (for log messages)."""
    qualname: str = getattr(callback, "__qualname__", None) or ""
    module: str = getattr(callback, "__module__", None) or ""
    if qualname and module:
        return f"{module}.{qualname}"
    return repr(callback)