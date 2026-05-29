"""
src/core/fsm.py
===============
Silent Sentinel — Edge AI Research Platform
Dead Man's Switch Finite State Machine (FSM).

State topology
--------------

                   CrisisVerifiedEvent received
    ┌─────────────┐  ──────────────────────────►  ┌──────────────────┐
    │  MONITORING │                                │ ANOMALY_DETECTED │
    └─────────────┘  ◄──────────────────────────  └──────────────────┘
          ▲              clear_alarm() called               │ immediate
          │              from any active state              ▼
          │                                        ┌──────────────────┐
          │    clear_alarm()                       │ COUNTDOWN_ACTIVE │
          │  ◄─────────────────────────────────── └──────────────────┘
          │                                                 │
          │   ┌─────────┐    clear_alarm()                 │ countdown
          │   │ CLEARED │  ◄──────────────────────         │ expires
          │   └─────────┘                                  ▼
          │        │                             ┌──────────────────┐
          └────────┘  (auto-reset)               │ CRISIS_VERIFIED  │
            after log                            └──────────────────┘
                                                          │
                                                          │ (terminal —
                                                          │  manual reset
                                                          │  required)

Threading model
---------------
*  ``DeadMansSwitchFSM`` is safe to call from any thread.
*  All state mutations are serialised behind a single ``threading.Lock``.
*  The countdown runs on a dedicated ``threading.Thread`` (daemon).  It
   wakes every second via ``threading.Event.wait(timeout=1)`` so it can
   be interrupted immediately when ``clear_alarm`` is called.
*  The ``_stop_event`` pattern is preferred over ``threading.Timer``
   because it allows sub-second cancellation and per-tick logging without
   spawning a new timer object for every tick.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config_loader import AppConfig
    from src.core.event_bus import EventBus

from src.core.events import BaseEvent
from src.core.fusion import CrisisVerifiedEvent
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


# ===========================================================================
# State enumeration
# ===========================================================================


class SwitchState(Enum):
    """All legal states of the Dead Man's Switch FSM."""

    MONITORING = auto()         # Normal operation; no anomaly in-flight.
    ANOMALY_DETECTED = auto()   # CrisisVerifiedEvent received; arm sequence started.
    COUNTDOWN_ACTIVE = auto()   # Countdown timer is running.
    CRISIS_VERIFIED = auto()    # Countdown expired without user clearance.
    CLEARED = auto()            # User cleared the alarm before expiry.


# ===========================================================================
# Allowed transitions (adjacency table)
# ===========================================================================

# Maps each state to the set of states it is permitted to enter.
# Any transition not listed here raises RuntimeError.
_ALLOWED_TRANSITIONS: dict[SwitchState, frozenset[SwitchState]] = {
    SwitchState.MONITORING:        frozenset({SwitchState.ANOMALY_DETECTED}),
    SwitchState.ANOMALY_DETECTED:  frozenset({SwitchState.COUNTDOWN_ACTIVE,
                                               SwitchState.CLEARED}),
    SwitchState.COUNTDOWN_ACTIVE:  frozenset({SwitchState.CRISIS_VERIFIED,
                                               SwitchState.CLEARED}),
    SwitchState.CRISIS_VERIFIED:   frozenset({SwitchState.MONITORING}),   # manual reset
    SwitchState.CLEARED:           frozenset({SwitchState.MONITORING}),
}


# ===========================================================================
# FSM class
# ===========================================================================


class DeadMansSwitchFSM:
    """Thread-safe Dead Man's Switch FSM with non-blocking countdown.

    The FSM listens for ``CrisisVerifiedEvent`` on the shared ``EventBus``.
    When received it arms a countdown timer.  If the operator does not call
    ``clear_alarm`` within ``countdown_seconds``, the machine transitions to
    ``CRISIS_VERIFIED`` and emits a critical alert log.

    Parameters
    ----------
    config:
        Validated ``AppConfig``; ``config.fsm.countdown_seconds`` sets the
        timer duration.
    event_bus:
        Shared ``EventBus`` instance; used to subscribe to
        ``CrisisVerifiedEvent``.
    """

    def __init__(self, config: AppConfig, event_bus: EventBus) -> None:
        self._countdown_seconds: int = config.fsm.countdown_seconds
        self._event_bus: EventBus = event_bus

        # --- FSM state (protected by _lock) ---
        self._state: SwitchState = SwitchState.MONITORING
        self._lock: threading.Lock = threading.Lock()

        # --- Countdown thread coordination ---
        # _stop_event is set to interrupt the countdown early.
        self._stop_event: threading.Event = threading.Event()
        self._countdown_thread: threading.Thread | None = None

        # --- Contextual metadata for logging ---
        self._trigger_event: CrisisVerifiedEvent | None = None

        event_bus.subscribe(CrisisVerifiedEvent, self._on_crisis_verified)

        logger.info(
            "DeadMansSwitchFSM initialised.",
            extra={
                "initial_state": self._state.name,
                "countdown_seconds": self._countdown_seconds,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SwitchState:
        """Current FSM state (thread-safe read)."""
        with self._lock:
            return self._state

    def clear_alarm(self) -> bool:
        """Attempt to clear an active alarm before the countdown expires.

        Safe to call from any thread at any time.  No-ops gracefully when
        the FSM is already in ``MONITORING`` or ``CLEARED``.

        Returns
        -------
        bool
            ``True`` if the alarm was successfully cleared; ``False`` if
            there was no active alarm to clear (idempotent).
        """
        with self._lock:
            current = self._state

            if current not in (
                SwitchState.ANOMALY_DETECTED,
                SwitchState.COUNTDOWN_ACTIVE,
            ):
                logger.info(
                    "clear_alarm called but no active alarm to clear.",
                    extra={"current_state": current.name},
                )
                return False

            # Signal the countdown thread to stop before transitioning.
            self._stop_event.set()
            self._transition_to(SwitchState.CLEARED)

        # Join the countdown thread outside the lock to avoid deadlock.
        self._join_countdown_thread()

        # Auto-advance CLEARED → MONITORING.
        with self._lock:
            self._transition_to(SwitchState.MONITORING)

        logger.info(
            "Alarm cleared by operator. FSM returned to MONITORING.",
            extra={"countdown_seconds_remaining": self._countdown_seconds},
        )
        return True

    def reset_from_crisis(self) -> bool:
        """Manually reset the FSM from ``CRISIS_VERIFIED`` back to
        ``MONITORING``.

        This is the only exit path from ``CRISIS_VERIFIED``; it must be
        called explicitly by an operator or external escalation system after
        a confirmed crisis event has been acknowledged.

        Returns
        -------
        bool
            ``True`` if reset was performed; ``False`` if the FSM was not
            in ``CRISIS_VERIFIED``.
        """
        with self._lock:
            if self._state is not SwitchState.CRISIS_VERIFIED:
                logger.warning(
                    "reset_from_crisis called but state is not CRISIS_VERIFIED.",
                    extra={"current_state": self._state.name},
                )
                return False
            self._transition_to(SwitchState.MONITORING)

        logger.info("FSM manually reset from CRISIS_VERIFIED to MONITORING.")
        return True

    # ------------------------------------------------------------------
    # EventBus callback
    # ------------------------------------------------------------------

    def _on_crisis_verified(self, event: BaseEvent) -> None:
        """Handle an incoming ``CrisisVerifiedEvent`` from the fusion engine."""
        if not isinstance(event, CrisisVerifiedEvent):
            return

        with self._lock:
            if self._state is not SwitchState.MONITORING:
                logger.warning(
                    "CrisisVerifiedEvent received but FSM is not in MONITORING; "
                    "ignoring to prevent concurrent alarm stacking.",
                    extra={
                        "current_state": self._state.name,
                        "event_fused_score": event.fused_score,
                    },
                )
                return

            self._trigger_event = event
            self._stop_event.clear()

            # MONITORING → ANOMALY_DETECTED → COUNTDOWN_ACTIVE
            self._transition_to(SwitchState.ANOMALY_DETECTED)
            self._transition_to(SwitchState.COUNTDOWN_ACTIVE)

        # Launch countdown thread outside the lock.
        self._launch_countdown()

    # ------------------------------------------------------------------
    # Countdown machinery
    # ------------------------------------------------------------------

    def _launch_countdown(self) -> None:
        """Start the countdown thread (daemon so it never blocks shutdown)."""
        thread = threading.Thread(
            target=self._run_countdown,
            name="fsm-countdown",
            daemon=True,
        )
        with self._lock:
            self._countdown_thread = thread

        thread.start()
        logger.info(
            "Countdown thread launched.",
            extra={"countdown_seconds": self._countdown_seconds},
        )

    def _run_countdown(self) -> None:
        """Countdown loop: tick every second, log remaining time, handle expiry."""
        remaining: int = self._countdown_seconds
        start_time: float = time.monotonic()

        while remaining > 0:
            # Block for up to 1 second OR until _stop_event is set.
            interrupted: bool = self._stop_event.wait(timeout=1.0)

            if interrupted:
                logger.info(
                    "Countdown interrupted by stop signal.",
                    extra={"remaining_seconds": remaining},
                )
                return  # clear_alarm already handles the state transition.

            elapsed: float = time.monotonic() - start_time
            remaining = max(0, self._countdown_seconds - int(elapsed))

            logger.info(
                "Countdown tick.",
                extra={
                    "remaining_seconds": remaining,
                    "elapsed_seconds": round(elapsed, 2),
                    "state": self.state.name,
                },
            )

        # Countdown reached zero without interruption.
        self._handle_countdown_expired()

    def _handle_countdown_expired(self) -> None:
        """Transition to CRISIS_VERIFIED and emit the critical alert log."""
        with self._lock:
            if self._state is not SwitchState.COUNTDOWN_ACTIVE:
                # Race condition guard: clear_alarm may have fired concurrently.
                logger.debug(
                    "Countdown expired but state is no longer COUNTDOWN_ACTIVE; "
                    "skipping crisis transition.",
                    extra={"current_state": self._state.name},
                )
                return

            self._transition_to(SwitchState.CRISIS_VERIFIED)
            trigger = self._trigger_event

        # Critical structured log — intended for PagerDuty / SIEM ingestion.
        logger.critical(
            "CRISIS CONFIRMED — countdown expired without operator clearance.",
            extra={
                "state": SwitchState.CRISIS_VERIFIED.name,
                "countdown_seconds": self._countdown_seconds,
                "trigger_fused_score": (
                    round(trigger.fused_score, 6) if trigger else None
                ),
                "trigger_acoustic_confidence": (
                    round(trigger.acoustic_confidence, 6) if trigger else None
                ),
                "trigger_motion_impact_g": (
                    round(trigger.motion_impact_g, 6) if trigger else None
                ),
                "trigger_entropy": (
                    round(trigger.entropy, 6) if trigger else None
                ),
                "trigger_stillness_duration_sec": (
                    round(trigger.stillness_duration, 4) if trigger else None
                ),
                "action_required": "Dispatch emergency response immediately.",
            },
        )

    def _join_countdown_thread(self, timeout: float = 2.0) -> None:
        """Wait for the countdown thread to finish, with a timeout guard."""
        thread: threading.Thread | None
        with self._lock:
            thread = self._countdown_thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning(
                    "Countdown thread did not terminate within timeout.",
                    extra={"timeout_sec": timeout},
                )

    # ------------------------------------------------------------------
    # Transition helper
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: SwitchState) -> None:
        """Execute and log a state transition.

        Must be called with ``_lock`` held.

        Parameters
        ----------
        new_state:
            Target FSM state.

        Raises
        ------
        RuntimeError
            If the transition is not listed in ``_ALLOWED_TRANSITIONS``.
        """
        old_state: SwitchState = self._state
        allowed: frozenset[SwitchState] = _ALLOWED_TRANSITIONS.get(
            old_state, frozenset()
        )

        if new_state not in allowed:
            msg = (
                f"Illegal FSM transition: {old_state.name} → {new_state.name}. "
                f"Allowed from {old_state.name}: "
                f"{[s.name for s in allowed] or 'none'}."
            )
            logger.error(msg, extra={"old_state": old_state.name, "new_state": new_state.name})
            raise RuntimeError(msg)

        self._state = new_state

        logger.info(
            "FSM state transition.",
            extra={
                "old_state": old_state.name,
                "new_state": new_state.name,
            },
        )


# ===========================================================================
# Smoke-test entry point
# ===========================================================================

if __name__ == "__main__":
    import sys

    from src.config_loader import load_config
    from src.core.event_bus import EventBus
    from src.core.fusion import CrisisVerifiedEvent
    from src.utils.logger import setup_logging

    setup_logging(level="DEBUG")

    cfg = load_config()
    bus = EventBus()
    fsm = DeadMansSwitchFSM(config=cfg, event_bus=bus)

    # ---------------------------------------------------------------
    # Test 1: Normal arm → clear before expiry
    # ---------------------------------------------------------------
    logger.info("=== Test 1: arm then clear_alarm ===")
    assert fsm.state is SwitchState.MONITORING

    bus.publish(CrisisVerifiedEvent(
        fused_score=0.85,
        acoustic_confidence=0.80,
        motion_impact_g=7.5,
        entropy=0.3,
        veto_applied=None,
        stillness_duration=4.0,
    ))

    # Give the thread a moment to start the countdown.
    time.sleep(0.2)
    assert fsm.state is SwitchState.COUNTDOWN_ACTIVE, (
        f"Expected COUNTDOWN_ACTIVE, got {fsm.state.name}"
    )

    # Clear well before expiry (countdown is 10 s; we clear at ~0.2 s).
    cleared = fsm.clear_alarm()
    assert cleared is True, "clear_alarm should return True"
    assert fsm.state is SwitchState.MONITORING, (
        f"Expected MONITORING after clear, got {fsm.state.name}"
    )
    logger.info("Test 1 passed — alarm cleared before expiry.")

    # ---------------------------------------------------------------
    # Test 2: clear_alarm when no alarm active (idempotent)
    # ---------------------------------------------------------------
    logger.info("=== Test 2: clear_alarm when idle ===")
    result = fsm.clear_alarm()
    assert result is False, "clear_alarm should return False when no alarm active"
    logger.info("Test 2 passed — clear_alarm is a safe no-op when idle.")

    # ---------------------------------------------------------------
    # Test 3: Countdown expiry (use a 1-second config override via a
    #         minimal in-process patch to avoid waiting 10 seconds).
    # ---------------------------------------------------------------
    logger.info("=== Test 3: countdown expiry ===")

    # Patch countdown duration to 2 seconds for the test.
    fsm._countdown_seconds = 2  # noqa: SLF001

    bus.publish(CrisisVerifiedEvent(
        fused_score=0.91,
        acoustic_confidence=0.88,
        motion_impact_g=9.0,
        entropy=0.25,
        veto_applied=None,
        stillness_duration=5.2,
    ))

    # Wait for countdown to expire (2 s + 0.5 s buffer).
    time.sleep(3.0)
    assert fsm.state is SwitchState.CRISIS_VERIFIED, (
        f"Expected CRISIS_VERIFIED after expiry, got {fsm.state.name}"
    )
    logger.info("Test 3 passed — FSM reached CRISIS_VERIFIED after countdown expiry.")

    # Manual reset after acknowledged crisis.
    reset_ok = fsm.reset_from_crisis()
    assert reset_ok is True
    assert fsm.state is SwitchState.MONITORING
    logger.info("Test 3 continued — manual reset from CRISIS_VERIFIED succeeded.")

    print("\n[OK] All FSM smoke tests passed.", file=sys.stderr)