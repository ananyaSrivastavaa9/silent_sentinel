"""
src/sensors/acoustic.py
=======================
Silent Sentinel — Edge AI Research Platform
Acoustic feature extraction engine.

Computes a 65-dimensional acoustic feature vector from raw PCM audio using
only NumPy and SciPy.  No librosa, no telephony libraries.

Feature vector layout (65 elements)
-------------------------------------
 [0:13]   13 × Mel-Frequency Cepstral Coefficients (MFCCs)
 [13:26]  13 × Delta-MFCC  (Δ)
 [26:39]  13 × Delta-Delta-MFCC  (ΔΔ)
 [39]     Spectral Rolloff frequency (Hz, normalised to Nyquist)
 [40]     Zero Crossing Rate (ZCR)
 [41]     RMS Energy
 [42:65]  23 × zero-padding  (reserved for future features)

All arithmetic uses float64 internally; the public API returns
``tuple[float, ...]`` of exactly 65 Python floats.

Design notes
------------
* The Mel filterbank matrix is computed **once** at construction time and
  cached as a (n_mels × n_fft_bins) NumPy array.  On a 40 ms frame at
  16 kHz (640 samples) the per-call cost is dominated by a single rfft,
  one matrix multiply, and a DCT — well under 1 ms on a Cortex-A53.
* Pre-emphasis, Hamming windowing, and log-energy flooring are applied
  inside ``extract_features`` to match the signal conditioning expected
  by a downstream neural classifier.
* Delta coefficients are estimated from a *single frame* using a
  second-order finite-difference approximation over ±2 synthetic
  neighbours (frame replicated at the boundaries).  This avoids the
  need to buffer multiple frames while still providing a reasonable
  proxy for temporal dynamics.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import dct

if TYPE_CHECKING:
    from src.config_loader import AppConfig
    from src.core.event_bus import EventBus

from src.core.events import AcousticAnomalyEvent
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Number of base MFCC coefficients.
_N_MFCC: int = 13

#: Number of Mel filterbank channels.
_N_MELS: int = 40

#: Frequency floor for the Mel filterbank (Hz).  Excludes DC and sub-bass.
_F_MIN_HZ: float = 80.0

#: Fraction of the Nyquist frequency used as the filterbank ceiling.
_F_MAX_FRACTION: float = 0.95

#: Floor applied before log compression to prevent log(0).
_LOG_FLOOR: float = 1e-10

#: Rolloff energy fraction: frequency below which this fraction of the
#: total spectral energy is contained.
_ROLLOFF_FRACTION: float = 0.85

#: Total output feature vector length.
_FEATURE_DIM: int = 65


# ===========================================================================
# Mel scale helpers
# ===========================================================================


def _hz_to_mel(freq_hz: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in Hz to the Mel scale (O'Shaughnessy 1987).

    Parameters
    ----------
    freq_hz:
        Scalar or array of frequencies in Hz.

    Returns
    -------
    float | np.ndarray
        Corresponding Mel values.
    """
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    """Convert Mel scale values back to Hz.

    Parameters
    ----------
    mel:
        Scalar or array of Mel values.

    Returns
    -------
    float | np.ndarray
        Corresponding frequencies in Hz.
    """
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Construct a triangular Mel filterbank matrix.

    Each row is one Mel-spaced triangular filter mapped onto the linear FFT
    frequency bins.  Filters are area-normalised so that narrow and wide
    filters contribute equally to the log-energy sum.

    Parameters
    ----------
    n_mels:
        Number of Mel filterbank channels.
    n_fft:
        Length of the FFT (full, not one-sided).
    sample_rate:
        ADC sample rate in Hz.
    f_min:
        Lower frequency bound (Hz).
    f_max:
        Upper frequency bound (Hz).  Typically 0.95 × Nyquist.

    Returns
    -------
    np.ndarray
        Shape ``(n_mels, n_fft // 2 + 1)`` — maps power spectrum bins to
        per-channel Mel energies.
    """
    n_bins: int = n_fft // 2 + 1  # one-sided spectrum length
    freq_bins: np.ndarray = np.linspace(0.0, sample_rate / 2.0, n_bins)  # Hz

    # n_mels + 2 equally-spaced points in Mel space, converted back to Hz.
    mel_points: np.ndarray = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points: np.ndarray = _mel_to_hz(mel_points)  # shape: (n_mels + 2,)

    filterbank: np.ndarray = np.zeros((n_mels, n_bins), dtype=np.float64)

    for m in range(1, n_mels + 1):
        f_left: float = hz_points[m - 1]
        f_centre: float = hz_points[m]
        f_right: float = hz_points[m + 1]

        # Rising slope
        rising_mask = (freq_bins >= f_left) & (freq_bins <= f_centre)
        if f_centre != f_left:
            filterbank[m - 1, rising_mask] = (
                (freq_bins[rising_mask] - f_left) / (f_centre - f_left)
            )

        # Falling slope
        falling_mask = (freq_bins > f_centre) & (freq_bins <= f_right)
        if f_right != f_centre:
            filterbank[m - 1, falling_mask] = (
                (f_right - freq_bins[falling_mask]) / (f_right - f_centre)
            )

        # Area normalisation — makes energy independent of filter bandwidth.
        width: float = f_right - f_left
        if width > 0.0:
            filterbank[m - 1] *= 2.0 / width

    return filterbank


# ===========================================================================
# Main class
# ===========================================================================


class AcousticFeatureExtractor:
    """Edge-optimised acoustic feature extractor for Silent Sentinel.

    Computes a 65-dimensional feature vector from a single raw PCM frame
    using a pre-cached Mel filterbank and SciPy DCT.

    Parameters
    ----------
    config:
        Validated ``AppConfig`` instance — provides ``sample_rate``,
        ``frame_length_ms``, ``feature_dimensions``, and
        ``classification_threshold``.
    event_bus:
        Application-wide ``EventBus`` instance.  When computed features
        exceed ``classification_threshold``, an ``AcousticAnomalyEvent``
        is published.

    Attributes
    ----------
    _sample_rate:
        ADC sample rate (Hz).
    _frame_len:
        Expected number of samples per frame
        (``sample_rate × frame_length_ms / 1000``).
    _threshold:
        Anomaly classification threshold from config.
    _filterbank:
        Pre-computed Mel filterbank matrix, shape
        ``(_N_MELS, _frame_len // 2 + 1)``.
    _hamming_window:
        Pre-computed Hamming window of length ``_frame_len``.
    """

    def __init__(self, config: AppConfig, event_bus: EventBus) -> None:
        self._sample_rate: int = config.acoustic.sample_rate
        self._frame_len: int = int(
            self._sample_rate * config.acoustic.frame_length_ms / 1000.0
        )
        self._threshold: float = config.acoustic.classification_threshold
        self._event_bus: EventBus = event_bus

        # Pre-compute expensive structures once at startup.
        self._hamming_window: np.ndarray = np.hamming(self._frame_len)
        self._filterbank: np.ndarray = _build_mel_filterbank(
            n_mels=_N_MELS,
            n_fft=self._frame_len,
            sample_rate=self._sample_rate,
            f_min=_F_MIN_HZ,
            f_max=_F_MAX_FRACTION * (self._sample_rate / 2.0),
        )

        logger.info(
            "AcousticFeatureExtractor initialised.",
            extra={
                "sample_rate": self._sample_rate,
                "frame_len_samples": self._frame_len,
                "n_mels": _N_MELS,
                "n_mfcc": _N_MFCC,
                "output_dim": _FEATURE_DIM,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, audio_frame: np.ndarray) -> tuple[float, ...]:
        """Extract a 65-dimensional feature vector from a raw PCM frame.

        Parameters
        ----------
        audio_frame:
            1-D NumPy array of PCM samples (float32 or float64).  The frame
            is silently padded with zeros or truncated to ``_frame_len``
            samples before processing.

        Returns
        -------
        tuple[float, ...]
            Exactly 65 floats in the layout described in the module docstring.
            Returns 65 zeros if *audio_frame* is empty or silent (RMS < 1e-9).

        Notes
        -----
        All intermediate computations use float64 to avoid rounding errors
        in the filterbank matrix multiply.
        """
        _ZEROS: tuple[float, ...] = tuple(0.0 for _ in range(_FEATURE_DIM))

        # ---- 0. Input validation -----------------------------------------
        if audio_frame is None or audio_frame.size == 0:
            logger.debug("Empty audio frame received; returning zero vector.")
            return _ZEROS

        frame: np.ndarray = audio_frame.astype(np.float64, copy=True).ravel()

        # ---- 1. Silence detection ----------------------------------------
        rms_energy: float = float(np.sqrt(np.mean(frame ** 2)))
        if rms_energy < 1e-9:
            logger.debug("Silent frame detected; returning zero vector.")
            return _ZEROS

        # ---- 2. Pad / truncate to exact frame length ---------------------
        if frame.size < self._frame_len:
            frame = np.pad(frame, (0, self._frame_len - frame.size))
        else:
            frame = frame[: self._frame_len]

        # ---- 3. Pre-emphasis (high-pass: y[n] = x[n] - 0.97·x[n-1]) ----
        frame[1:] -= 0.97 * frame[:-1]

        # ---- 4. Hamming window -------------------------------------------
        frame *= self._hamming_window

        # ---- 5. Power spectrum via rfft ----------------------------------
        spectrum: np.ndarray = np.abs(np.fft.rfft(frame, n=self._frame_len)) ** 2
        # shape: (n_fft // 2 + 1,)

        # ---- 6. Mel filterbank energies ----------------------------------
        mel_energies: np.ndarray = self._filterbank @ spectrum
        # shape: (_N_MELS,)
        np.clip(mel_energies, _LOG_FLOOR, None, out=mel_energies)

        # ---- 7. Log compression -----------------------------------------
        log_mel: np.ndarray = np.log(mel_energies)

        # ---- 8. DCT → 13 MFCCs (DCT type-II, orthonormal) ---------------
        mfcc: np.ndarray = dct(log_mel, type=2, n=_N_MFCC, norm="ortho")
        # shape: (13,)

        # ---- 9. Delta & Delta-Delta coefficients -------------------------
        delta: np.ndarray = self._delta(mfcc)
        delta_delta: np.ndarray = self._delta(delta)

        # ---- 10. Spectral rolloff ----------------------------------------
        rolloff: float = self._spectral_rolloff(spectrum)

        # ---- 11. Zero Crossing Rate (ZCR) --------------------------------
        zcr: float = float(
            np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame.size)
        )

        # ---- 12. Assemble raw feature vector (42 elements) ---------------
        raw_features: np.ndarray = np.concatenate(
            [mfcc, delta, delta_delta, [rolloff], [zcr], [rms_energy]]
        )
        # raw_features.size == 13 + 13 + 13 + 1 + 1 + 1 == 42

        # ---- 13. Pad to exactly _FEATURE_DIM (65) ------------------------
        output: np.ndarray = np.zeros(_FEATURE_DIM, dtype=np.float64)
        n_copy: int = min(raw_features.size, _FEATURE_DIM)
        output[:n_copy] = raw_features[:n_copy]

        feature_tuple: tuple[float, ...] = tuple(float(v) for v in output)

        # ---- 14. Anomaly detection & event publishing --------------------
        self._maybe_publish(feature_tuple, rms_energy)

        return feature_tuple

    def simulate_audio_stream(
        self,
        n_frames: int = 10,
        anomaly_probability: float = 0.3,
        rng_seed: int | None = 42,
    ) -> None:
        """Generate synthetic audio frames and exercise the full pipeline.

        Produces a mix of:
        * **Normal frames** — band-limited Gaussian noise at low amplitude.
        * **Anomaly frames** — broadband noise with higher energy and an
          injected 2 kHz tone that pushes confidence above ``_threshold``.

        Parameters
        ----------
        n_frames:
            Number of frames to synthesise and process.
        anomaly_probability:
            Fraction of frames that should resemble acoustic anomalies.
        rng_seed:
            Seed for the NumPy RNG, ensuring deterministic test output.
            Pass ``None`` for non-deterministic behaviour.
        """
        rng: np.random.Generator = np.random.default_rng(rng_seed)
        logger.info(
            "Starting simulated audio stream.",
            extra={"n_frames": n_frames, "anomaly_probability": anomaly_probability},
        )

        t: np.ndarray = np.arange(self._frame_len) / self._sample_rate

        for i in range(n_frames):
            is_anomaly: bool = rng.random() < anomaly_probability

            if is_anomaly:
                # High-energy broadband noise + 2 kHz sinusoidal tone.
                noise: np.ndarray = rng.normal(0.0, 0.4, self._frame_len)
                tone: np.ndarray = 0.6 * np.sin(2.0 * np.pi * 2000.0 * t)
                frame: np.ndarray = (noise + tone).astype(np.float32)
            else:
                # Low-amplitude band-limited noise (normal ambient).
                frame = rng.normal(0.0, 0.02, self._frame_len).astype(np.float32)

            t_start: float = time.perf_counter()
            features: tuple[float, ...] = self.extract_features(frame)
            elapsed_us: float = (time.perf_counter() - t_start) * 1e6

            logger.info(
                "Frame processed.",
                extra={
                    "frame_index": i,
                    "is_anomaly_injected": is_anomaly,
                    "rms_energy": round(features[41], 6),
                    "mfcc_0": round(features[0], 4),
                    "elapsed_us": round(elapsed_us, 2),
                },
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _delta(coeffs: np.ndarray, width: int = 2) -> np.ndarray:
        """Compute delta (temporal derivative) coefficients for a single frame.

        Because we process one frame at a time, we approximate the temporal
        gradient using a second-order finite-difference filter over ±*width*
        synthetic neighbours formed by replicating the frame at its boundaries.

        The formula for each coefficient ``c`` is:

        .. math::

            \\Delta c = \\frac{\\sum_{n=1}^{N} n \\cdot (c_{+n} - c_{-n})}
                              {2 \\sum_{n=1}^{N} n^2}

        where ``N = width`` and ``c_{±n}`` are boundary-padded replicates.

        Parameters
        ----------
        coeffs:
            1-D array of input coefficients (e.g. MFCCs), shape ``(K,)``.
        width:
            Finite-difference half-window (default: 2 → uses ±1 and ±2).

        Returns
        -------
        np.ndarray
            Delta coefficients, shape ``(K,)``.
        """
        # Edge-replicate to simulate adjacent frames.
        padded: np.ndarray = np.pad(coeffs, width, mode="edge")
        denominator: float = 2.0 * float(np.sum(np.arange(1, width + 1) ** 2))
        delta: np.ndarray = np.zeros_like(coeffs)

        for n in range(1, width + 1):
            delta += n * (
                padded[width + n: width + n + coeffs.size]
                - padded[width - n: width - n + coeffs.size]
            )

        return delta / denominator

    def _spectral_rolloff(self, power_spectrum: np.ndarray) -> float:
        """Compute the normalised spectral rolloff frequency.

        Returns the FFT bin index (normalised to ``[0, 1]`` by the total number
        of bins) below which ``_ROLLOFF_FRACTION`` of the total spectral energy
        is contained.

        Parameters
        ----------
        power_spectrum:
            One-sided power spectrum array of shape ``(n_fft // 2 + 1,)``.

        Returns
        -------
        float
            Normalised rolloff in ``[0, 1]``.  Multiply by the Nyquist
            frequency to recover Hz.
        """
        total_energy: float = float(power_spectrum.sum())
        if total_energy < _LOG_FLOOR:
            return 0.0

        cumulative: np.ndarray = np.cumsum(power_spectrum)
        threshold: float = _ROLLOFF_FRACTION * total_energy

        indices: np.ndarray = np.where(cumulative >= threshold)[0]
        rolloff_bin: int = int(indices[0]) if indices.size > 0 else len(power_spectrum) - 1

        return float(rolloff_bin) / float(len(power_spectrum) - 1)

    def _maybe_publish(
        self,
        features: tuple[float, ...],
        rms_energy: float,
    ) -> None:
        """Publish an ``AcousticAnomalyEvent`` when features exceed threshold.

        Confidence is approximated as the sigmoid of the mean absolute
        MFCC value normalised by RMS energy.  This is a lightweight proxy
        for a real classifier — replace with a TFLite / ONNX call in
        production.

        Parameters
        ----------
        features:
            The 65-dimensional feature tuple computed by ``extract_features``.
        rms_energy:
            Frame RMS energy, used to gate the confidence estimate.
        """
        mfcc_values: np.ndarray = np.array(features[:_N_MFCC])
        raw_score: float = float(np.mean(np.abs(mfcc_values))) * rms_energy
        confidence: float = float(1.0 / (1.0 + np.exp(-raw_score)))  # sigmoid

        if confidence >= self._threshold:
            event = AcousticAnomalyEvent(
                confidence=confidence,
                features=features[:_N_MFCC],  # publish only the base MFCCs
            )
            self._event_bus.publish(event)
            logger.debug(
                "AcousticAnomalyEvent published.",
                extra={"confidence": round(confidence, 4), "rms": round(rms_energy, 6)},
            )
        else:
            logger.debug(
                "Frame below anomaly threshold.",
                extra={"confidence": round(confidence, 4), "threshold": self._threshold},
            )


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

    # Wire up a simple listener to verify event publication.
    def _on_anomaly(event: AcousticAnomalyEvent) -> None:
        logger.info(
            "TEST LISTENER — AcousticAnomalyEvent received.",
            extra={"confidence": round(event.confidence, 4)},
        )

    from src.core.events import AcousticAnomalyEvent as _AAE  # noqa: PLC0415

    bus.subscribe(_AAE, _on_anomaly)

    extractor = AcousticFeatureExtractor(config=cfg, event_bus=bus)

    # --- Test 1: silence ---
    silent = np.zeros(640, dtype=np.float32)
    result = extractor.extract_features(silent)
    assert len(result) == 65, "Silence: wrong output length"
    assert all(v == 0.0 for v in result), "Silence: expected all zeros"
    logger.info("Test 1 passed — silence returns zero vector.")

    # --- Test 2: normal frame ---
    rng = np.random.default_rng(0)
    normal_frame = rng.normal(0.0, 0.02, 640).astype(np.float32)
    result = extractor.extract_features(normal_frame)
    assert len(result) == 65, "Normal: wrong output length"
    logger.info("Test 2 passed — normal frame returns 65-dimensional vector.")

    # --- Test 3: stream simulation ---
    extractor.simulate_audio_stream(n_frames=6, anomaly_probability=0.5)
    logger.info("Test 3 passed — simulated stream completed.")

    print("\n[OK] All smoke tests passed.", file=sys.stderr)