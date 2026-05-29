"""
src/core/fusion.py
==================
Silent Sentinel — Edge AI Research Platform
Bayesian Sensor Fusion Engine with Shannon Entropy and Smart Veto logic.

Architecture overview
---------------------
``BayesianFusionEngine`` subscribes to both ``AcousticAnomalyEvent`` and
``MotionAnomalyEvent`` on the shared ``EventBus``.  Incoming events are held
in a bounded time-window cache.  Every new arrival triggers ``fuse_signals``,
which combines the most recent evidence via a weighted Bayesian score, applies
Shannon Entropy degradation, evaluates smart veto rules, and — if the gated
threshold is exceeded — publishes a ``CrisisVerifiedEvent``.

Signal processing pipeline (per ``fuse_signals`` call)
-------------------------------------------------------
1.  **Evidence extraction** — pull the freshest acoustic and motion readings
    from the time-window cache.
2.  **Base score**
    ``base = (acoustic_conf × acoustic_weight) + (motion_norm × motion_weight)``
    where ``motion_norm = tanh(impact_g / high_g_threshold)`` maps raw g-force
    to ``(0, 1)`` without hard clipping.
3.  **Shannon Entropy degradation** — compute H over the two normalised
    probability masses ``[p, 1-p]`` (binary entropy).  High entropy (signals
    near 0.5) reflects maximal uncertainty and attenuates the base score:
    ``degraded = base × (1 − entropy_weight × H_norm)``
4.  **Smart veto rules** — explicit logical gates that force the score below
    ``dynamic_confidence_gate`` regardless of the weighted result.
5.  **Threshold gate** — if ``final_score ≥ dynamic_confidence_gate``, publish
    ``CrisisVerifiedEvent``.

Veto catalogue
--------------
``VETO_MOTION_NO_ACOUSTIC``
    High-G impact (> threshold) but acoustic confidence is negligible
    (< ``ACOUSTIC_FLOOR``).  Covers dropped-device false alarms on hard
    surfaces.

``VETO_ACOUSTIC_NO_MOTION``
    Strong acoustic signal but motion impact is negligible
    (< ``MOTION_FLOOR``).  Guards against loud ambient noise (glass breaking
    nearby, door slam) without accompanying physical disturbance.

``VETO_STALE_EVIDENCE``
    The two signals arrived more than ``CROSS_MODAL_STALENESS_SEC`` seconds
    apart; they are unlikely to share a causal origin.

Threading model
---------------
Event callbacks execute on whichever thread calls ``EventBus.publish``.  The
cache and state are protected by ``threading.Lock`` to allow safe concurrent
sensor pipelines.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config_loader import AppConfig
    from src.core.event_bus import EventBus

from src.core.events import AcousticAnomalyEvent, BaseEvent, MotionAnomalyEvent
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


# ===========================================================================
# Additional event type produced by this layer
# ===========================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class CrisisVerifiedEvent(BaseEvent):
    """Emitted when fused sensor evidence surpasses the confidence gate.

    This is the terminal output of the fusion pipeline and the primary
    trigger for downstream alerting (FSM, notification, escalation).

    Attributes
    ----------
    fused_score:
        Final post-veto, entropy-degraded confidence in ``[0, 1]``.
    acoustic_confidence:
        Raw confidence from the contributing ``AcousticAnomalyEvent``,
        or ``0.0`` if no acoustic evidence was present.
    motion_impact_g:
        Peak g-force from the contributing ``MotionAnomalyEvent``,
        or ``0.0`` if no motion evidence was present.
    entropy:
        Normalised Shannon binary entropy of the evidence mix at decision
        time.  ``0.0`` = perfectly certain, ``1.0`` = maximum uncertainty.
    veto_applied:
        Name of any veto rule that was evaluated (even if it did not fire),
        for audit logging.  ``None`` if no veto was considered.
    stillness_duration:
        Stillness duration in seconds from the motion event, or ``0.0``.
    """

    fused_score: float
    acoustic_confidence: float
    motion_impact_g: float
    entropy: float
    veto_applied: str | None
    stillness_duration: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.fused_score <= 1.0:
            raise ValueError(
                f"CrisisVerifiedEvent.fused_score must be in [0, 1]; "
                f"got {self.fused_score!r}."
            )
        if not 0.0 <= self.entropy <= 1.0:
            raise ValueError(
                f"CrisisVerifiedEvent.entropy must be in [0, 1]; "
                f"got {self.entropy!r}."
            )


# ===========================================================================
# Internal evidence snapshot
# ===========================================================================


@dataclass(slots=True)
class _EvidenceSnapshot:
    """Mutable container for the most recent event of each modality.

    Both fields are replaced atomically (under lock) each time a newer
    event of the same type arrives.
    """

    acoustic: AcousticAnomalyEvent | None = field(default=None)
    motion: MotionAnomalyEvent | None = field(default=None)


# ===========================================================================
# Veto result
# ===========================================================================


@dataclass(frozen=True, slots=True)
class _VetoResult:
    """Outcome of evaluating the full veto rule set.

    Attributes
    ----------
    vetoed:
        ``True`` if at least one rule fired and the score must be clamped.
    rule_name:
        Identifier of the rule that fired, or ``None``.
    reason:
        Human-readable explanation for log/audit output.
    """

    vetoed: bool
    rule_name: str | None
    reason: str


# ===========================================================================
# Main fusion engine
# ===========================================================================


class BayesianFusionEngine:
    """Weighted Bayesian fusion with entropy degradation and smart veto logic.

    Parameters
    ----------
    config:
        Validated ``AppConfig`` providing fusion weights, thresholds, and
        the dynamic confidence gate.
    event_bus:
        Shared ``EventBus`` instance.  The engine subscribes to
        ``AcousticAnomalyEvent`` and ``MotionAnomalyEvent``, and publishes
        ``CrisisVerifiedEvent``.
    """

    # ------------------------------------------------------------------
    # Class-level tuning constants (not user-configurable)
    # ------------------------------------------------------------------

    #: Maximum age (seconds) of an event still considered *current* evidence.
    _EVIDENCE_TTL_SEC: float = 5.0

    #: Maximum temporal gap between acoustic and motion events before the
    #: ``VETO_STALE_EVIDENCE`` rule fires.
    _CROSS_MODAL_STALENESS_SEC: float = 3.0

    #: Acoustic confidence below this value is treated as "negligible" by
    #: the ``VETO_MOTION_NO_ACOUSTIC`` rule.
    _ACOUSTIC_FLOOR: float = 0.10

    #: Normalised motion score below this value is treated as "negligible"
    #: by the ``VETO_ACOUSTIC_NO_MOTION`` rule.
    _MOTION_FLOOR: float = 0.05

    #: Weight of entropy degradation.  At ``1.0`` a maximally uncertain
    #: signal (H=1) would be zeroed out; ``0.4`` provides a softer penalty.
    _ENTROPY_WEIGHT: float = 0.40

    #: Veto clamp — a vetoed score is forced to this ceiling, which sits
    #: safely below any reasonable ``dynamic_confidence_gate`` (≥ 0.5).
    _VETO_SCORE_CEILING: float = 0.20

    def __init__(self, config: AppConfig, event_bus: EventBus) -> None:
        # --- Config ---
        self._acoustic_weight: float = config.fusion.acoustic_weight
        self._motion_weight: float = config.fusion.motion_weight
        self._confidence_gate: float = config.fusion.dynamic_confidence_gate
        self._high_g_threshold: float = config.motion.high_g_threshold
        self._event_bus: EventBus = event_bus

        # --- Shared mutable state (protected by _lock) ---
        self._lock: threading.Lock = threading.Lock()
        self._evidence: _EvidenceSnapshot = _EvidenceSnapshot()

        # --- Subscribe to both sensor modalities ---
        event_bus.subscribe(AcousticAnomalyEvent, self._on_acoustic_event)
        event_bus.subscribe(MotionAnomalyEvent, self._on_motion_event)

        logger.info(
            "BayesianFusionEngine initialised and subscribed.",
            extra={
                "acoustic_weight": self._acoustic_weight,
                "motion_weight": self._motion_weight,
                "confidence_gate": self._confidence_gate,
                "evidence_ttl_sec": self._EVIDENCE_TTL_SEC,
                "entropy_weight": self._ENTROPY_WEIGHT,
                "acoustic_floor": self._ACOUSTIC_FLOOR,
                "motion_floor": self._MOTION_FLOOR,
                "cross_modal_staleness_sec": self._CROSS_MODAL_STALENESS_SEC,
            },
        )

    # ------------------------------------------------------------------
    # EventBus callbacks
    # ------------------------------------------------------------------

    def _on_acoustic_event(self, event: BaseEvent) -> None:
        """Store the incoming acoustic event and trigger fusion."""
        if not isinstance(event, AcousticAnomalyEvent):
            return
        with self._lock:
            self._evidence.acoustic = event
        logger.debug(
            "Acoustic evidence cached.",
            extra={"confidence": round(event.confidence, 4), "ts": event.timestamp},
        )
        self.fuse_signals()

    def _on_motion_event(self, event: BaseEvent) -> None:
        """Store the incoming motion event and trigger fusion."""
        if not isinstance(event, MotionAnomalyEvent):
            return
        with self._lock:
            self._evidence.motion = event
        logger.debug(
            "Motion evidence cached.",
            extra={
                "impact_g": round(event.impact_g, 4),
                "stillness_sec": round(event.stillness_duration, 3),
                "ts": event.timestamp,
            },
        )
        self.fuse_signals()

    # ------------------------------------------------------------------
    # Public fusion API
    # ------------------------------------------------------------------

    def fuse_signals(self) -> float:
        """Run the full fusion pipeline over the current evidence window.

        Acquires a snapshot of the latest evidence (under lock), then
        executes the five-stage pipeline described in the module docstring.

        Returns
        -------
        float
            Final post-veto, entropy-degraded fused score in ``[0, 1]``.
            Also returns ``0.0`` if no valid evidence is available.

        Side effects
        ------------
        Publishes a ``CrisisVerifiedEvent`` if ``final_score ≥ confidence_gate``.
        """
        now: float = time.time()

        # --- Snapshot evidence under lock; release before heavy computation ---
        with self._lock:
            acoustic_evt = self._evidence.acoustic
            motion_evt = self._evidence.motion

        # --- Stage 1: Evidence validity / staleness ---
        acoustic_valid, acoustic_age = self._check_event_age(acoustic_evt, now)
        motion_valid, motion_age = self._check_event_age(motion_evt, now)

        if not acoustic_valid and not motion_valid:
            logger.debug("fuse_signals called with no valid evidence; skipping.")
            return 0.0

        # --- Extract scalar evidence values ---
        acoustic_conf: float = acoustic_evt.confidence if acoustic_valid else 0.0
        impact_g: float = motion_evt.impact_g if motion_valid else 0.0
        stillness_dur: float = motion_evt.stillness_duration if motion_valid else 0.0

        # Normalise g-force to (0, 1) via tanh — smooth, bounded, no hard clip.
        motion_norm: float = math.tanh(impact_g / max(self._high_g_threshold, 1e-9))

        logger.info(
            "Fusion pipeline — Stage 1: evidence extracted.",
            extra={
                "acoustic_valid": acoustic_valid,
                "motion_valid": motion_valid,
                "acoustic_conf": round(acoustic_conf, 4),
                "impact_g": round(impact_g, 4),
                "motion_norm": round(motion_norm, 4),
                "acoustic_age_sec": round(acoustic_age, 3) if acoustic_valid else None,
                "motion_age_sec": round(motion_age, 3) if motion_valid else None,
            },
        )

        # --- Stage 2: Weighted base score ---
        base_score: float = (
            acoustic_conf * self._acoustic_weight
            + motion_norm * self._motion_weight
        )

        logger.info(
            "Fusion pipeline — Stage 2: weighted base score.",
            extra={
                "acoustic_term": round(acoustic_conf * self._acoustic_weight, 6),
                "motion_term": round(motion_norm * self._motion_weight, 6),
                "base_score": round(base_score, 6),
            },
        )

        # --- Stage 3: Shannon Entropy degradation ---
        entropy_norm, degraded_score = self._apply_entropy_degradation(
            base_score, acoustic_conf, motion_norm
        )

        logger.info(
            "Fusion pipeline — Stage 3: entropy degradation.",
            extra={
                "entropy_norm": round(entropy_norm, 6),
                "entropy_weight": self._ENTROPY_WEIGHT,
                "degraded_score": round(degraded_score, 6),
            },
        )

        # --- Stage 4: Smart veto rules ---
        temporal_gap: float | None = (
            abs(acoustic_evt.timestamp - motion_evt.timestamp)
            if acoustic_valid and motion_valid
            else None
        )

        veto: _VetoResult = self._evaluate_veto_rules(
            acoustic_conf=acoustic_conf,
            motion_norm=motion_norm,
            acoustic_valid=acoustic_valid,
            motion_valid=motion_valid,
            temporal_gap=temporal_gap,
        )

        final_score: float = degraded_score
        if veto.vetoed:
            final_score = min(degraded_score, self._VETO_SCORE_CEILING)

        logger.info(
            "Fusion pipeline — Stage 4: veto evaluation.",
            extra={
                "veto_fired": veto.vetoed,
                "veto_rule": veto.rule_name,
                "veto_reason": veto.reason,
                "pre_veto_score": round(degraded_score, 6),
                "post_veto_score": round(final_score, 6),
            },
        )

        # --- Stage 5: Confidence gate & event publication ---
        gate_passed: bool = final_score >= self._confidence_gate

        logger.info(
            "Fusion pipeline — Stage 5: confidence gate.",
            extra={
                "final_score": round(final_score, 6),
                "confidence_gate": self._confidence_gate,
                "gate_passed": gate_passed,
            },
        )

        if gate_passed:
            self._publish_crisis(
                fused_score=final_score,
                acoustic_confidence=acoustic_conf,
                motion_impact_g=impact_g,
                entropy=entropy_norm,
                veto_applied=veto.rule_name,
                stillness_duration=stillness_dur,
            )

        return final_score

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _apply_entropy_degradation(
        self,
        base_score: float,
        acoustic_conf: float,
        motion_norm: float,
    ) -> tuple[float, float]:
        """Compute binary Shannon Entropy and attenuate *base_score*.

        Shannon binary entropy for a Bernoulli(p) distribution:

        .. math::

            H(p) = -p \\log_2 p - (1-p) \\log_2 (1-p)

        The entropy is computed over the *evidence mix ratio*:
        ``p = acoustic_term / (acoustic_term + motion_term + ε)``

        A balanced mix (p ≈ 0.5, H → 1) means the two signals are of
        equal magnitude and neither dominates — high uncertainty.
        A skewed mix (p → 0 or 1, H → 0) means one signal dominates —
        lower uncertainty.

        The degraded score is:
        ``degraded = base × (1 − entropy_weight × H_norm)``

        where ``H_norm ∈ [0, 1]`` is H / H_max = H / 1.0.

        Parameters
        ----------
        base_score:
            Weighted sum of evidence before entropy adjustment.
        acoustic_conf:
            Normalised acoustic confidence in ``[0, 1]``.
        motion_norm:
            Normalised motion score in ``[0, 1]``.

        Returns
        -------
        tuple[float, float]
            ``(entropy_norm, degraded_score)`` both in ``[0, 1]``.
        """
        eps: float = 1e-12
        acoustic_term: float = acoustic_conf * self._acoustic_weight
        motion_term: float = motion_norm * self._motion_weight
        total: float = acoustic_term + motion_term + eps

        p: float = acoustic_term / total  # fraction due to acoustic signal

        # Binary entropy H(p) in bits; max = 1.0 at p = 0.5.
        def _h(prob: float) -> float:
            if prob <= 0.0 or prob >= 1.0:
                return 0.0
            return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)

        entropy_norm: float = _h(p)  # already normalised to [0, 1] for binary case

        degradation_factor: float = 1.0 - self._ENTROPY_WEIGHT * entropy_norm
        degraded_score: float = max(0.0, base_score * degradation_factor)

        return entropy_norm, degraded_score

    def _evaluate_veto_rules(
        self,
        acoustic_conf: float,
        motion_norm: float,
        acoustic_valid: bool,
        motion_valid: bool,
        temporal_gap: float | None,
    ) -> _VetoResult:
        """Evaluate all smart veto rules in priority order.

        Rules are checked sequentially; the first match short-circuits.

        Parameters
        ----------
        acoustic_conf:
            Raw acoustic confidence (``0.0`` if no acoustic evidence).
        motion_norm:
            Normalised motion score in ``[0, 1]`` (``0.0`` if no motion).
        acoustic_valid:
            Whether a non-stale acoustic event is present.
        motion_valid:
            Whether a non-stale motion event is present.
        temporal_gap:
            Absolute time difference (seconds) between the two events,
            or ``None`` if only one modality has evidence.

        Returns
        -------
        _VetoResult
            Describes whether a veto fired and which rule matched.
        """
        # --- Rule 1: High impact, negligible acoustics ---
        # A hard impact (above high_g_threshold normalised to > 0.5 via tanh)
        # with no vocal / acoustic distress is almost always a device drop.
        if motion_valid and motion_norm > 0.5 and acoustic_conf < self._ACOUSTIC_FLOOR:
            return _VetoResult(
                vetoed=True,
                rule_name="VETO_MOTION_NO_ACOUSTIC",
                reason=(
                    f"High motion impact (norm={motion_norm:.4f}) with negligible "
                    f"acoustic confidence ({acoustic_conf:.4f} < floor "
                    f"{self._ACOUSTIC_FLOOR}). Likely dropped device, not crisis."
                ),
            )

        # --- Rule 2: Strong acoustics, negligible motion ---
        # Loud acoustic anomaly without any physical disturbance suggests
        # an ambient sound event (nearby impact, alarm, TV) rather than a
        # personal crisis.
        if acoustic_valid and acoustic_conf > 0.6 and motion_norm < self._MOTION_FLOOR:
            return _VetoResult(
                vetoed=True,
                rule_name="VETO_ACOUSTIC_NO_MOTION",
                reason=(
                    f"Strong acoustic signal (conf={acoustic_conf:.4f}) with "
                    f"negligible motion (norm={motion_norm:.4f} < floor "
                    f"{self._MOTION_FLOOR}). Likely ambient noise, not crisis."
                ),
            )

        # --- Rule 3: Temporal staleness — events too far apart ---
        if temporal_gap is not None and temporal_gap > self._CROSS_MODAL_STALENESS_SEC:
            return _VetoResult(
                vetoed=True,
                rule_name="VETO_STALE_EVIDENCE",
                reason=(
                    f"Cross-modal temporal gap {temporal_gap:.3f}s exceeds "
                    f"staleness threshold {self._CROSS_MODAL_STALENESS_SEC}s. "
                    "Events are unlikely to share a causal origin."
                ),
            )

        # --- No veto ---
        return _VetoResult(
            vetoed=False,
            rule_name=None,
            reason="No veto rules matched; evidence accepted.",
        )

    # ------------------------------------------------------------------
    # Publication helpers
    # ------------------------------------------------------------------

    def _publish_crisis(
        self,
        fused_score: float,
        acoustic_confidence: float,
        motion_impact_g: float,
        entropy: float,
        veto_applied: str | None,
        stillness_duration: float,
    ) -> None:
        """Construct and publish a ``CrisisVerifiedEvent``."""
        event = CrisisVerifiedEvent(
            fused_score=round(fused_score, 6),
            acoustic_confidence=round(acoustic_confidence, 6),
            motion_impact_g=round(motion_impact_g, 6),
            entropy=round(entropy, 6),
            veto_applied=veto_applied,
            stillness_duration=round(stillness_duration, 4),
        )
        self._event_bus.publish(event)
        logger.warning(
            "CrisisVerifiedEvent published — crisis threshold exceeded.",
            extra={
                "fused_score": event.fused_score,
                "acoustic_confidence": event.acoustic_confidence,
                "motion_impact_g": event.motion_impact_g,
                "entropy": event.entropy,
                "veto_applied": event.veto_applied,
                "stillness_duration_sec": event.stillness_duration,
            },
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _check_event_age(
        event: BaseEvent | None,
        now: float,
    ) -> tuple[bool, float]:
        """Check whether *event* is non-None and within the evidence TTL.

        Parameters
        ----------
        event:
            Candidate evidence event, or ``None``.
        now:
            Current Unix timestamp for age calculation.

        Returns
        -------
        tuple[bool, float]
            ``(is_valid, age_seconds)`` — ``age_seconds`` is ``0.0`` when
            ``event`` is ``None``.
        """
        if event is None:
            return False, 0.0
        age: float = now - event.timestamp
        valid: bool = age <= BayesianFusionEngine._EVIDENCE_TTL_SEC
        return valid, age


# ===========================================================================
# Smoke-test entry point
# ===========================================================================

if __name__ == "__main__":
    import sys

    from src.config_loader import load_config
    from src.core.event_bus import EventBus
    from src.utils.logger import setup_logging

    setup_logging(level="DEBUG")

    cfg = load_config()
    bus = EventBus()

    crisis_events: list[CrisisVerifiedEvent] = []

    def _on_crisis(event: CrisisVerifiedEvent) -> None:
        crisis_events.append(event)
        logger.warning(
            "TEST LISTENER — CrisisVerifiedEvent received.",
            extra={
                "fused_score": event.fused_score,
                "entropy": event.entropy,
                "veto_applied": event.veto_applied,
            },
        )

    bus.subscribe(CrisisVerifiedEvent, _on_crisis)
    engine = BayesianFusionEngine(config=cfg, event_bus=bus)

    # --- Test 1: Veto — high motion, no acoustics (dropped phone) ---
    logger.info("=== Test 1: VETO_MOTION_NO_ACOUSTIC ===")
    bus.publish(MotionAnomalyEvent(impact_g=8.0, stillness_duration=3.5))
    assert len(crisis_events) == 0, "Veto should suppress crisis event"
    logger.info("Test 1 passed — veto correctly suppressed dropped-phone scenario.")

    # --- Test 2: Veto — strong acoustics, negligible motion ---
    logger.info("=== Test 2: VETO_ACOUSTIC_NO_MOTION ===")
    bus.publish(AcousticAnomalyEvent(confidence=0.92, features=(0.1,) * 13))
    assert len(crisis_events) == 0, "Veto should suppress acoustic-only event"
    logger.info("Test 2 passed — veto correctly suppressed ambient-noise scenario.")

    # --- Test 3: True crisis — co-incident high-confidence events ---
    logger.info("=== Test 3: True crisis — co-incident evidence ===")
    # Reset engine's evidence cache by creating a fresh engine.
    bus2 = EventBus()
    crisis_events2: list[CrisisVerifiedEvent] = []

    def _on_crisis2(event: CrisisVerifiedEvent) -> None:
        crisis_events2.append(event)

    bus2.subscribe(CrisisVerifiedEvent, _on_crisis2)
    engine2 = BayesianFusionEngine(config=cfg, event_bus=bus2)

    bus2.publish(MotionAnomalyEvent(impact_g=9.5, stillness_duration=5.0))
    bus2.publish(AcousticAnomalyEvent(confidence=0.93, features=(0.5,) * 13))

    assert len(crisis_events2) >= 1, (
        f"Expected at least 1 CrisisVerifiedEvent; got {len(crisis_events2)}"
    )
    assert crisis_events2[-1].fused_score >= cfg.fusion.dynamic_confidence_gate
    logger.info(
        "Test 3 passed — CrisisVerifiedEvent correctly published.",
        extra={"fused_score": crisis_events2[-1].fused_score},
    )

    print("\n[OK] All fusion engine smoke tests passed.", file=sys.stderr)