"""
src/core/events.py
==================
Silent Sentinel — Edge AI Research Platform
Core system event definitions.

All events are plain, immutable dataclasses so they can be safely passed
across thread boundaries and serialised without any framework dependency.

Design notes
------------
* ``frozen=True`` — events are value objects; they must never be mutated
  after construction.  Any attempt raises ``FrozenInstanceError`` at runtime.
* ``kw_only=True`` — enforces keyword-argument construction throughout,
  preventing accidental positional-argument mismatches as fields are added.
* ``slots=True`` — reduces per-instance memory overhead; beneficial when
  thousands of events are in-flight during burst detection windows.
* ``timestamp`` defaults to ``time.time()`` via ``field(default_factory=…)``
  so callers rarely need to supply it explicitly, but can override it for
  deterministic unit tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ===========================================================================
# Base event
# ===========================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class BaseEvent:
    """Abstract base for every Silent Sentinel system event.

    Attributes
    ----------
    timestamp:
        Unix epoch time (seconds, fractional) at which the event was created.
        Defaults to ``time.time()`` at instantiation.
    """

    timestamp: float = field(default_factory=time.time)


# ===========================================================================
# Acoustic pipeline events
# ===========================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class AcousticAnomalyEvent(BaseEvent):
    """Emitted by the acoustic pipeline when an anomaly exceeds the
    classification threshold defined in ``AcousticConfig``.

    Attributes
    ----------
    confidence:
        Posterior probability assigned to the anomaly class by the
        classifier.  Expected range: ``[0.0, 1.0]``.
    features:
        Raw feature vector (e.g. Mel-filterbank coefficients) that produced
        this classification.  Immutable tuple preserves the frozen contract.
    """

    confidence: float
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"AcousticAnomalyEvent.confidence must be in [0, 1]; "
                f"got {self.confidence!r}."
            )
        if not self.features:
            raise ValueError("AcousticAnomalyEvent.features must not be empty.")


# ===========================================================================
# Motion pipeline events
# ===========================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class MotionAnomalyEvent(BaseEvent):
    """Emitted by the IMU pipeline when either a high-G spike or an
    unexpected stillness condition is detected.

    Attributes
    ----------
    impact_g:
        Peak acceleration magnitude (g-force units) measured during the
        triggering window.  Must be >= 0.
    stillness_duration:
        Continuous stillness duration in seconds preceding or coinciding
        with the event.  Must be >= 0.
    """

    impact_g: float
    stillness_duration: float

    def __post_init__(self) -> None:
        if self.impact_g < 0.0:
            raise ValueError(
                f"MotionAnomalyEvent.impact_g must be >= 0; got {self.impact_g!r}."
            )
        if self.stillness_duration < 0.0:
            raise ValueError(
                f"MotionAnomalyEvent.stillness_duration must be >= 0; "
                f"got {self.stillness_duration!r}."
            )


# ===========================================================================
# FSM state-change events
# ===========================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class StateChangeEvent(BaseEvent):
    """Emitted by the Finite State Machine whenever a state transition occurs.

    Attributes
    ----------
    old_state:
        Name of the FSM state being exited.  Must be a non-empty string.
    new_state:
        Name of the FSM state being entered.  Must differ from ``old_state``.
    """

    old_state: str
    new_state: str

    def __post_init__(self) -> None:
        if not self.old_state:
            raise ValueError("StateChangeEvent.old_state must not be empty.")
        if not self.new_state:
            raise ValueError("StateChangeEvent.new_state must not be empty.")
        if self.old_state == self.new_state:
            raise ValueError(
                f"StateChangeEvent requires a real transition; "
                f"old_state and new_state are both {self.old_state!r}."
            )