"""
src/config_loader.py
====================
Silent Sentinel — Edge AI Research Platform
Configuration loader and validator (Pydantic V2).

Public API
----------
    from src.config_loader import load_config, AppConfig

    cfg = load_config()                          # uses default path
    cfg = load_config("config/config.yaml")      # explicit path
    cfg = load_config(config_path="/abs/path")   # absolute path
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ===========================================================================
# Section models
# ===========================================================================


class AcousticConfig(BaseModel):
    """Acoustic sensing pipeline configuration.

    Attributes
    ----------
    sample_rate:
        ADC sample rate in Hz. Must be > 0.
    frame_length_ms:
        Analysis frame duration in milliseconds. Must be > 0.
    feature_dimensions:
        Length of the extracted feature vector. Must be >= 1.
    classification_threshold:
        Minimum posterior probability for forwarding an acoustic event
        to the fusion layer. Range: [0.0, 1.0].
    """

    model_config = ConfigDict(frozen=True)

    sample_rate: int = Field(..., gt=0, description="ADC sample rate in Hz.")
    frame_length_ms: int = Field(..., gt=0, description="Frame duration in ms.")
    feature_dimensions: int = Field(..., ge=1, description="Feature vector length.")
    classification_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum posterior probability for event forwarding.",
    )


class MotionConfig(BaseModel):
    """IMU / motion sensing pipeline configuration.

    Attributes
    ----------
    high_g_threshold:
        Acceleration magnitude (g-force) above which a high-G event is
        flagged. Must be > 0.
    stillness_duration_sec:
        Continuous seconds of sub-threshold motion required to assert
        the stillness flag. Must be > 0.
    variance_threshold:
        Maximum windowed accelerometer variance (g²) for stillness to
        hold. Must be >= 0.
    """

    model_config = ConfigDict(frozen=True)

    high_g_threshold: float = Field(
        ..., gt=0.0, description="High-G event threshold in g-force units."
    )
    stillness_duration_sec: float = Field(
        ..., gt=0.0, description="Stillness assertion window in seconds."
    )
    variance_threshold: float = Field(
        ..., ge=0.0, description="Max windowed variance (g²) for stillness."
    )


class FusionConfig(BaseModel):
    """Multi-modal evidence fusion layer configuration.

    Attributes
    ----------
    acoustic_weight:
        Scalar weight applied to the acoustic confidence score.
    motion_weight:
        Scalar weight applied to the motion confidence score.
    dynamic_confidence_gate:
        Minimum fused confidence required before propagating a decision
        to the FSM. Range: [0.0, 1.0].

    Notes
    -----
    A model-level validator enforces ``acoustic_weight + motion_weight == 1.0``
    (tolerance ±1e-6) so the weighted sum remains a proper probability.
    """

    model_config = ConfigDict(frozen=True)

    acoustic_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Acoustic evidence weight."
    )
    motion_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Motion evidence weight."
    )
    dynamic_confidence_gate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum fused confidence for FSM propagation.",
    )

    @model_validator(mode="after")
    def _weights_must_sum_to_one(self) -> "FusionConfig":
        """Ensure the evidence weights form a valid convex combination."""
        total = self.acoustic_weight + self.motion_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"acoustic_weight + motion_weight must equal 1.0; got {total:.8f}."
            )
        return self


class FSMConfig(BaseModel):
    """Finite State Machine controller configuration.

    Attributes
    ----------
    countdown_seconds:
        Duration (seconds) of the pre-alert countdown before the FSM
        commits to the CRISIS_VERIFIED state. Must be > 0.
    """

    model_config = ConfigDict(frozen=True)

    countdown_seconds: int = Field(
        ..., gt=0, description="Pre-alert countdown duration in seconds."
    )


class LoggingConfig(BaseModel):
    """Application-wide logging subsystem configuration.

    Attributes
    ----------
    level:
        Standard Python logging level. Accepted values (case-insensitive):
        ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    format:
        Log record serialisation format.
        ``JSON`` emits structured JSON lines; ``TEXT`` emits plain text.
    """

    model_config = ConfigDict(frozen=True)

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        ..., description="Python logging level."
    )
    format: Literal["JSON", "TEXT"] = Field(
        ..., description="Log record serialisation format."
    )

    @field_validator("level", mode="before")
    @classmethod
    def _normalise_level(cls, v: object) -> object:
        if isinstance(v, str):
            return v.upper()
        return v

    @field_validator("format", mode="before")
    @classmethod
    def _normalise_format(cls, v: object) -> object:
        if isinstance(v, str):
            return v.upper()
        return v


# ===========================================================================
# Master configuration model
# ===========================================================================


class AppConfig(BaseModel):
    """Top-level application configuration for Silent Sentinel.

    Aggregates all section-level models into a single immutable object
    that is passed through the application's dependency graph.

    Attributes
    ----------
    acoustic:
        Acoustic pipeline parameters.
    motion:
        IMU / motion pipeline parameters.
    fusion:
        Multi-modal fusion layer parameters.
    fsm:
        Finite State Machine controller parameters.
    logging:
        Logging subsystem parameters.

    Example
    -------
    >>> from src.config_loader import load_config, AppConfig
    >>> cfg: AppConfig = load_config()
    >>> cfg.acoustic.sample_rate
    16000
    """

    model_config = ConfigDict(frozen=True)

    acoustic: AcousticConfig
    motion: MotionConfig
    fusion: FusionConfig
    fsm: FSMConfig
    logging: LoggingConfig


# ===========================================================================
# Public loader function
# ===========================================================================


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    """Read, parse, and validate the Silent Sentinel configuration file.

    This function is the single authoritative entry-point for loading
    configuration. It is safe to call from any module:

        from src.config_loader import load_config, AppConfig
        cfg: AppConfig = load_config()

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  May be relative (resolved
        against the current working directory) or absolute.  Defaults to
        ``"config/config.yaml"``.

    Returns
    -------
    AppConfig
        A fully validated, immutable Pydantic V2 model instance.

    Raises
    ------
    FileNotFoundError
        When *config_path* does not point to an existing regular file.
    ValueError
        When the file exists but contains syntactically invalid YAML, or
        when the top-level YAML value is not a mapping.
    pydantic.ValidationError
        When the parsed mapping violates the schema (wrong types,
        out-of-range values, missing required fields, constraint
        violations such as weights not summing to 1.0, etc.).

    Notes
    -----
    ``pydantic.ValidationError`` is intentionally **not** caught here.
    Its structured error report (field path + message per violation) is
    far more useful to operators than any generic wrapper would be.
    The exception is logged at ERROR level before re-raising so that
    structured log pipelines (Fluentd, Loki, etc.) capture the event.
    """
    path = Path(config_path)

    # ------------------------------------------------------------------
    # 1. File existence guard
    # ------------------------------------------------------------------
    if not path.is_file():
        msg = f"Configuration file not found: '{path.resolve()}'"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # 2. YAML deserialisation
    # ------------------------------------------------------------------
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"Failed to parse YAML from '{path}': {exc}"
        logger.error(msg)
        raise ValueError(msg) from exc

    if not isinstance(raw, dict):
        msg = (
            f"Expected a YAML mapping at the top level of '{path}'; "
            f"got {type(raw).__name__!r}."
        )
        logger.error(msg)
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # 3. Pydantic V2 validation
    # ------------------------------------------------------------------
    try:
        config: AppConfig = AppConfig.model_validate(raw)
    except Exception as exc:
        logger.error("Configuration validation failed: %s", exc)
        raise

    logger.info("Configuration loaded and validated from '%s'.", path.resolve())
    return config


# ===========================================================================
# Smoke-test entry point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )

    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError) as exc:
        logger.critical("Startup aborted — could not load config: %s", exc)
        sys.exit(1)

    print(cfg.model_dump_json(indent=2))

    assert cfg.acoustic.sample_rate == 16_000
    assert abs(cfg.fusion.acoustic_weight + cfg.fusion.motion_weight - 1.0) < 1e-6
    assert cfg.logging.level == "INFO"
    assert cfg.fsm.countdown_seconds == 10

    print("\n[OK] All spot-checks passed.")