"""
src/sensors/motion.py
=====================
Silent Sentinel — Edge AI Research Platform
Motion telemetry layer: IMU processing, micro-state tracking, anomaly detection.

State machine topology
----------------------

                      magnitude > high_g_threshold
    ┌────────┐  ─────────────────────────────────────►  ┌─────────────────┐
    │  IDLE  │                                           │ IMPACT_DETECTED │
    └────────┘  ◄─────────────────────────────────────  └─────────────────┘
        ▲          variance > variance_threshold OR             │
        │          stillness window elapsed w/o quiet           │
        │                                                        │ variance ≤ variance_threshold
        │                                                        │ for stillness_duration_sec
        │                                                        ▼
        │                                           ┌──────────────────────┐
        └─────────────────────────────────────────  │ STILLNESS_MONITORING │
               publish MotionAnomalyEvent +          └──────────────────────┘
               reset to IDLE

Design notes
------------
*  All timing uses ``time.monotonic()`` — immune to wall-clock adjustments.
*  The variance window is a fixed-size ``collections.deque`` so insertion
   and eviction are O(1); no dynamic allocation after initialisation.
*  The class is intentionally single-threaded.  If samples arrive from a
   background thread, the caller must serialise access (e.g. via a queue).
*  Sensor fusion weight constants (acoustic/motion) live in AppConfig; this
   layer only concerns itself with the motion side of the decision.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config_loader import AppConfig
    from src.core.event_bus import EventBus

from src.core.events import MotionAnomalyEvent
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


# ===========================================================================
# Internal state enumeration
# ===========================================================================


class _MotionState(Enum):
    """Micro-state labels for the internal IMU state machine."""

    IDLE = auto()
    IMPACT_DETECTED = auto()
    STILLNESS_MONITORING = auto()


# ===========================================================================
# Processor class
# ===========================================================================


class MotionTelemetryProcessor:
    """Process raw 3-axis accelerometer samples and publish motion anomalies.

    The processor implements a three-state micro-FSM:

    ``IDLE``
        Normal operation.  Each sample updates the variance window.
        A magnitude spike above ``high_g_threshold`` triggers transition
        to ``IMPACT_DETECTED``.

    ``IMPACT_DETECTED``
        A candidate impact has been registered.  The peak G value is
        latched.  The *very next* sample immediately re-evaluates variance
        to decide whether to enter ``STILLNESS_MONITORING`` (quiet) or
        revert to ``IDLE`` (movement resumed — false alarm path).

    ``STILLNESS_MONITORING``
        The device appears motionless after an impact.  Each sample
        checks whether the running variance remains below
        ``variance_threshold``.  Two exit paths exist:

        * **Verified anomaly** — stillness maintained for
          ``stillness_duration_sec`` → publish ``MotionAnomalyEvent``,
          reset to ``IDLE``.
        * **False alarm** — variance spikes above threshold before the
          window closes → reset to ``IDLE`` without publishing.

    Parameters
    ----------
    config:
        Validated ``AppConfig`` providing motion thresholds and timing.
    event_bus:
        Application-wide ``EventBus`` instance used to publish
        ``MotionAnomalyEvent`` on verified anomaly detection.
    """

    # Size of the rolling variance window (samples).  At a typical IMU poll
    # rate of 50 Hz this covers 400 ms — enough to distinguish micro-tremor
    # from genuine stillness without excessive memory.
    _VARIANCE_WINDOW: int = 20

    def __init__(self, config: AppConfig, event_bus: EventBus) -> None:
        self._high_g_threshold: float = config.motion.high_g_threshold
        self._stillness_duration: float = config.motion.stillness_duration_sec
        self._variance_threshold: float = config.motion.variance_threshold
        self._event_bus: EventBus = event_bus

        # --- FSM state ---
        self._state: _MotionState = _MotionState.IDLE

        # --- Impact tracking ---
        self._peak_g: float = 0.0
        self._impact_timestamp: float = 0.0

        # --- Stillness monitoring ---
        self._stillness_start: float = 0.0

        # --- Rolling magnitude window for variance estimation ---
        self._magnitude_window: deque[float] = deque(
            maxlen=self._VARIANCE_WINDOW
        )

        logger.info(
            "MotionTelemetryProcessor initialised.",
            extra={
                "high_g_threshold": self._high_g_threshold,
                "stillness_duration_sec": self._stillness_duration,
                "variance_threshold": self._variance_threshold,
                "variance_window_size": self._VARIANCE_WINDOW,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_accelerometer_data(
        self,
        x: float,
        y: float,
        z: float,
    ) -> None:
        """Process one 3-axis accelerometer sample through the micro-FSM.

        Parameters
        ----------
        x, y, z:
            Raw acceleration components in g-force units along the sensor's
            three orthogonal axes.  A device resting flat on a surface will
            read approximately (0, 0, 1) when z is the gravity axis.

        Side effects
        ------------
        * Updates the internal rolling magnitude window.
        * May transition the internal FSM state.
        * Publishes a ``MotionAnomalyEvent`` on the ``EventBus`` if a
          verified stillness anomaly is confirmed.
        """
        magnitude: float = math.sqrt(x * x + y * y + z * z)
        now: float = time.monotonic()

        self._magnitude_window.append(magnitude)
        current_variance: float = self._compute_variance()

        logger.debug(
            "IMU sample received.",
            extra={
                "state": self._state.name,
                "magnitude": round(magnitude, 4),
                "variance": round(current_variance, 6),
            },
        )

        # --- Dispatch to the appropriate state handler ---
        if self._state is _MotionState.IDLE:
            self._handle_idle(magnitude, now)

        elif self._state is _MotionState.IMPACT_DETECTED:
            self._handle_impact_detected(magnitude, current_variance, now)

        elif self._state is _MotionState.STILLNESS_MONITORING:
            self._handle_stillness_monitoring(magnitude, current_variance, now)

    def simulate_motion_stream(self, scenario: str = "true_crisis_fall") -> None:
        """Feed a pre-programmed sequence of IMU samples to exercise the FSM.

        Three scenarios are available:

        ``"normal_walking"``
            Sinusoidal oscillation at ~2 Hz around 1 g (gravity) with
            low-amplitude Gaussian jitter.  No spikes; FSM stays in IDLE.

        ``"accidental_drop_and_pickup"``
            Free-fall phase (≈ 0 g), high-impact spike on landing, then
            immediate vigorous movement (phone is picked up).  FSM enters
            IMPACT_DETECTED then reverts to IDLE without publishing — the
            canonical false-alarm path.

        ``"true_crisis_fall"``
            Free-fall phase, high-impact spike, then absolute stillness for
            longer than ``stillness_duration_sec``.  FSM traverses
            IDLE → IMPACT_DETECTED → STILLNESS_MONITORING → publishes
            ``MotionAnomalyEvent`` → IDLE.

        Parameters
        ----------
        scenario:
            One of ``"normal_walking"``, ``"accidental_drop_and_pickup"``,
            or ``"true_crisis_fall"``.  Raises ``ValueError`` for unknown keys.
        """
        _KNOWN = {"normal_walking", "accidental_drop_and_pickup", "true_crisis_fall"}
        if scenario not in _KNOWN:
            raise ValueError(
                f"Unknown scenario {scenario!r}. Choose from: {_KNOWN}."
            )

        logger.info("Simulated motion stream starting.", extra={"scenario": scenario})

        samples: list[tuple[float, float, float]]

        if scenario == "normal_walking":
            samples = self._gen_normal_walking()

        elif scenario == "accidental_drop_and_pickup":
            samples = self._gen_accidental_drop_and_pickup()

        else:  # true_crisis_fall
            samples = self._gen_true_crisis_fall()

        for x, y, z in samples:
            self.process_accelerometer_data(x, y, z)

        logger.info(
            "Simulated motion stream complete.",
            extra={"scenario": scenario, "total_samples": len(samples)},
        )

    # ------------------------------------------------------------------
    # FSM state handlers
    # ------------------------------------------------------------------

    def _handle_idle(self, magnitude: float, now: float) -> None:
        """Process a sample while in the IDLE state."""
        if magnitude > self._high_g_threshold:
            self._peak_g = magnitude
            self._impact_timestamp = now
            self._transition_to(_MotionState.IMPACT_DETECTED)

    def _handle_impact_detected(
        self,
        magnitude: float,
        variance: float,
        now: float,
    ) -> None:
        """Process a sample immediately following an impact detection.

        The window has just received a high-G sample.  The variance will be
        elevated.  We evaluate the *current* sample to decide which of two
        paths to take — this avoids a one-sample stall in the state machine.
        """
        # Latch the highest G seen so far in case the impact spans > 1 sample.
        if magnitude > self._peak_g:
            self._peak_g = magnitude

        if variance <= self._variance_threshold:
            # The device went quiet — begin timing the stillness window.
            self._stillness_start = now
            self._transition_to(_MotionState.STILLNESS_MONITORING)
        else:
            # Movement persists — likely a drop-and-pickup; reset.
            logger.debug(
                "High variance after impact — false alarm, resetting.",
                extra={"variance": round(variance, 6), "peak_g": round(self._peak_g, 4)},
            )
            self._reset_to_idle()

    def _handle_stillness_monitoring(
        self,
        magnitude: float,
        variance: float,
        now: float,
    ) -> None:
        """Process a sample while monitoring post-impact stillness."""
        if variance > self._variance_threshold:
            # Movement resumed before the window closed — false alarm.
            logger.debug(
                "Movement detected during stillness window — resetting.",
                extra={"variance": round(variance, 6)},
            )
            self._reset_to_idle()
            return

        stillness_elapsed: float = now - self._stillness_start

        if stillness_elapsed >= self._stillness_duration:
            # Verified anomaly — the device has been motionless long enough.
            self._publish_anomaly(stillness_elapsed)
            self._reset_to_idle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_variance(self) -> float:
        """Compute population variance of the current magnitude window.

        Returns 0.0 if the window contains fewer than 2 samples (variance
        is undefined for a single-element population).

        Returns
        -------
        float
            Population variance of magnitudes in ``_magnitude_window``.
        """
        n: int = len(self._magnitude_window)
        if n < 2:
            return 0.0

        values = list(self._magnitude_window)
        mean: float = sum(values) / n
        return sum((v - mean) ** 2 for v in values) / n

    def _transition_to(self, new_state: _MotionState) -> None:
        """Log and execute an FSM state transition."""
        logger.info(
            "IMU state transition.",
            extra={
                "from_state": self._state.name,
                "to_state": new_state.name,
                "peak_g": round(self._peak_g, 4),
            },
        )
        self._state = new_state

    def _reset_to_idle(self) -> None:
        """Clear transient tracking state and return the FSM to IDLE."""
        self._state = _MotionState.IDLE
        self._peak_g = 0.0
        self._impact_timestamp = 0.0
        self._stillness_start = 0.0
        self._magnitude_window.clear()
        logger.debug("FSM reset to IDLE.")

    def _publish_anomaly(self, stillness_elapsed: float) -> None:
        """Construct and publish a ``MotionAnomalyEvent``.

        Parameters
        ----------
        stillness_elapsed:
            Confirmed stillness duration in seconds at time of publishing.
        """
        event = MotionAnomalyEvent(
            impact_g=self._peak_g,
            stillness_duration=stillness_elapsed,
        )
        self._event_bus.publish(event)
        logger.warning(
            "MotionAnomalyEvent published — verified crisis fall.",
            extra={
                "peak_impact_g": round(self._peak_g, 4),
                "stillness_duration_sec": round(stillness_elapsed, 3),
            },
        )

    # ------------------------------------------------------------------
    # Scenario generators
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_normal_walking() -> list[tuple[float, float, float]]:
        """60 samples of sinusoidal gait oscillation around 1 g (Z axis).

        Mimics a device in a trouser pocket during brisk walking.
        Peak magnitude ≈ 1.3 g; no spikes above a typical 2.5 g threshold.
        """
        import math as _math  # local import to avoid polluting module namespace

        samples: list[tuple[float, float, float]] = []
        for i in range(60):
            t = i / 50.0  # 50 Hz poll rate
            # Lateral sway (~0.15 g), fore-aft heel-strike (~0.25 g), vertical gravity + bounce
            x = 0.15 * _math.sin(2 * _math.pi * 1.8 * t + 0.3)
            y = 0.25 * _math.sin(2 * _math.pi * 1.8 * t)
            z = 1.0 + 0.10 * _math.sin(2 * _math.pi * 3.6 * t)  # double cadence bounce
            # Small Gaussian jitter (deterministic approximation)
            jitter = 0.02 * _math.sin(137.0 * t)  # pseudo-noise via irrational freq
            samples.append((x + jitter, y + jitter, z + jitter))
        return samples

    def _gen_accidental_drop_and_pickup(self) -> list[tuple[float, float, float]]:
        """Simulate a phone drop followed by immediate retrieval.

        Phases
        ------
        1. **Free-fall** (5 samples) — all axes ≈ 0 g (weightlessness).
        2. **Impact** (2 samples) — spike well above ``high_g_threshold``.
        3. **Pickup movement** (25 samples) — high, erratic motion as the
           user grabs and pockets the device.  Variance stays elevated,
           preventing the FSM from reaching STILLNESS_MONITORING.
        """
        samples: list[tuple[float, float, float]] = []

        # Phase 1 — free-fall
        for _ in range(5):
            samples.append((0.02, 0.01, 0.03))  # near-zero-g

        # Phase 2 — impact spike (3 × threshold to ensure detection)
        spike: float = self._high_g_threshold * 3.0
        samples.append((spike * 0.6, spike * 0.3, spike * 0.75))
        samples.append((spike * 0.4, spike * 0.2, spike * 0.55))

        # Phase 3 — vigorous pickup movement
        import math as _math
        for i in range(25):
            t = i / 50.0
            x = 1.5 * _math.sin(2 * _math.pi * 4.0 * t + 0.7)
            y = 1.2 * _math.cos(2 * _math.pi * 3.5 * t)
            z = 1.0 + 0.8 * _math.sin(2 * _math.pi * 5.0 * t + 1.2)
            samples.append((x, y, z))

        return samples

    def _gen_true_crisis_fall(self) -> list[tuple[float, float, float]]:
        """Simulate a genuine crisis fall with prolonged post-impact stillness.

        Phases
        ------
        1. **Normal activity** (10 samples) — gentle walking to fill the
           variance window with baseline data.
        2. **Free-fall** (6 samples) — near-zero-g tumble.
        3. **Impact** (2 samples) — spike well above ``high_g_threshold``.
        4. **Absolute stillness** (samples sufficient to exceed
           ``stillness_duration_sec`` at a 50 Hz notional poll rate,
           plus a 20 % margin) — all axes read only gravitational
           component; variance collapses to essentially zero.
        """
        samples: list[tuple[float, float, float]] = []

        # Phase 1 — baseline activity (fills variance window)
        import math as _math
        for i in range(10):
            t = i / 50.0
            samples.append((
                0.1 * _math.sin(2 * _math.pi * 1.5 * t),
                0.1 * _math.cos(2 * _math.pi * 1.5 * t),
                1.0 + 0.05 * _math.sin(2 * _math.pi * 3.0 * t),
            ))

        # Phase 2 — free-fall
        for _ in range(6):
            samples.append((0.01, 0.01, 0.02))

        # Phase 3 — impact spike
        spike: float = self._high_g_threshold * 3.5
        samples.append((spike * 0.55, spike * 0.40, spike * 0.70))
        samples.append((spike * 0.30, spike * 0.25, spike * 0.45))

        # Phase 4 — stillness.
        # At 50 Hz, stillness_duration_sec × 50 samples covers the required
        # window; add 20 % margin and a minimum of 10 extra samples.
        still_count: int = max(
            int(self._stillness_duration * 50 * 1.2),
            int(self._stillness_duration * 50) + 10,
        )
        # Device lying face-down: gravity on Z axis, near-zero X and Y.
        for _ in range(still_count):
            samples.append((0.002, 0.001, 0.999))  # sub-milligee jitter

        return samples


# ===========================================================================
# Smoke-test entry point
# ===========================================================================

if __name__ == "__main__":
    import sys

    from src.config_loader import load_config
    from src.core.event_bus import EventBus
    from src.core.events import MotionAnomalyEvent as _MAE
    from src.utils.logger import setup_logging

    setup_logging(level="DEBUG")

    cfg = load_config()
    bus = EventBus()

    received_events: list[MotionAnomalyEvent] = []

    def _on_motion(event: _MAE) -> None:  # type: ignore[valid-type]
        received_events.append(event)
        logger.info(
            "TEST LISTENER — MotionAnomalyEvent received.",
            extra={
                "impact_g": round(event.impact_g, 4),
                "stillness_sec": round(event.stillness_duration, 3),
            },
        )

    bus.subscribe(_MAE, _on_motion)
    processor = MotionTelemetryProcessor(config=cfg, event_bus=bus)

    # --- Scenario A: normal walking — expect 0 events ---
    processor.simulate_motion_stream("normal_walking")
    assert len(received_events) == 0, "normal_walking must not fire an event"
    logger.info("Scenario A passed — no anomaly events for normal walking.")

    # --- Scenario B: accidental drop — expect 0 events ---
    processor.simulate_motion_stream("accidental_drop_and_pickup")
    assert len(received_events) == 0, "accidental_drop must not fire an event"
    logger.info("Scenario B passed — no anomaly event for drop-and-pickup.")

    # --- Scenario C: crisis fall — expect exactly 1 event ---
    processor.simulate_motion_stream("true_crisis_fall")
    assert len(received_events) == 1, (
        f"true_crisis_fall must fire exactly 1 event; got {len(received_events)}"
    )
    assert received_events[0].impact_g > cfg.motion.high_g_threshold
    assert received_events[0].stillness_duration >= cfg.motion.stillness_duration_sec
    logger.info("Scenario C passed — exactly 1 MotionAnomalyEvent published.")

    print("\n[OK] All motion telemetry smoke tests passed.", file=sys.stderr)