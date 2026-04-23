"""
Post-processing utilities for frame-level predictions.

Pipeline order (each stage is optional and independently configurable):

    raw frame probs  (from model sigmoid)
        │
        ▼
    1. smooth_probabilities()       ← probability-domain temporal filter
        │
        ▼
    2. hysteresis_threshold()       ← dual-threshold binarization
       OR simple threshold (>0.5)
        │
        ▼
    3. aggregate_frames_to_seconds()  ← frame→second binning
        │
        ▼
    4. enforce_minimum_bout()       ← absorb short flickers
        │
        ▼
    final second-level predictions
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


# ---------------------------------------------------------------------------
# Stage 1: Probability-domain temporal smoothing
# ---------------------------------------------------------------------------

def smooth_probabilities(
    frame_probs: np.ndarray,
    kernel_size: int = None,
    fps: int = 30,
    kernel_seconds: float = 1.0,
    method: str = "gaussian",
) -> np.ndarray:
    """
    Temporally smooth frame-level probabilities BEFORE thresholding.

    Operating in probability space preserves more information than smoothing
    binary labels. Transitions that hover near the decision boundary get
    stabilised without destroying sharp, confident state changes.

    Args:
        frame_probs:    [num_frames] array of P(immobile), range [0, 1].
        kernel_size:    Filter width in frames. If None, derived from
                        ``fps * kernel_seconds``.
        fps:            Frames per second (used only when kernel_size is None).
        kernel_seconds: Kernel width in seconds (default 1.0 s).
        method:         "gaussian" | "uniform" | "median".
                        - gaussian: smooth roll-off, good default.
                        - uniform:  equal-weight moving average.
                        - median:   non-linear, better at preserving edges
                                    but can create flat plateaus.

    Returns:
        smoothed: [num_frames] array, same shape, clipped to [0, 1].
    """
    if kernel_size is None:
        kernel_size = max(3, int(fps * kernel_seconds))

    if method == "gaussian":
        # sigma ~ kernel_size / 4  gives a window that contains ~95 % of the
        # Gaussian weight, similar effective width to a uniform filter.
        sigma = kernel_size / 4.0
        smoothed = gaussian_filter1d(frame_probs.astype(np.float64), sigma=sigma,
                                     mode="nearest")
    elif method == "uniform":
        smoothed = uniform_filter1d(frame_probs.astype(np.float64), size=kernel_size,
                                    mode="nearest")
    elif method == "median":
        # Median filter – manual implementation to avoid importing another
        # module.  Uses edge-padding like the scipy filters above.
        k = max(1, kernel_size | 1)          # ensure odd
        pad = k // 2
        padded = np.pad(frame_probs, (pad, pad), mode="edge")
        smoothed = np.empty_like(frame_probs, dtype=np.float64)
        for i in range(len(frame_probs)):
            smoothed[i] = np.median(padded[i : i + k])
    else:
        raise ValueError(f"Unknown smoothing method: {method!r}")

    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 2: Hysteresis thresholding
# ---------------------------------------------------------------------------

def hysteresis_threshold(
    frame_probs: np.ndarray,
    high: float = 0.6,
    low: float = 0.4,
) -> np.ndarray:
    """
    Binarise probabilities with a dual-threshold (Schmitt-trigger) rule.

    * A frame switches to *immobile* only when P(immobile) >= ``high``.
    * Once immobile, it stays immobile until P(immobile) drops below ``low``.
    * Starts in the *mobile* state.

    This dead-zone between ``low`` and ``high`` prevents the rapid on/off
    flicker that a single 0.5 threshold produces near decision boundaries.

    Args:
        frame_probs: [num_frames] P(immobile).
        high:        Threshold to enter immobility.
        low:         Threshold to exit immobility.

    Returns:
        predictions: [num_frames] int array, 0 = mobile, 1 = immobile.
    """
    if low >= high:
        raise ValueError(f"low ({low}) must be strictly less than high ({high})")

    predictions = np.zeros(len(frame_probs), dtype=np.int32)
    state = 0  # start mobile

    for i, p in enumerate(frame_probs):
        if state == 0 and p >= high:
            state = 1
        elif state == 1 and p < low:
            state = 0
        predictions[i] = state

    return predictions


# ---------------------------------------------------------------------------
# Stage 3: Frame → second aggregation  (existing logic, preserved)
# ---------------------------------------------------------------------------

def aggregate_frames_to_seconds(
    frame_predictions: np.ndarray,
    frame_probabilities: np.ndarray = None,
    fps: int = 60,
    agg_threshold: float = 0.5,
    confidence_threshold: float = None,
) -> np.ndarray:
    """
    Aggregate frame-level predictions to second-level with optional
    confidence filtering.

    Args:
        frame_predictions:  [num_frames] array of 0/1 predictions.
        frame_probabilities: [num_frames] array of P(immobile) (0–1).
        fps:                Frames per second.
        agg_threshold:      Fraction of immobile frames required to call the
                            second immobile (0.0 = any, 0.5 = majority,
                            1.0 = all).
        confidence_threshold: If set, revert immobility predictions to mobile
                              when mean P(immobile) in that second is below
                              this value.

    Returns:
        second_predictions: [num_seconds] array of 0/1 predictions.
    """
    num_frames = len(frame_predictions)
    num_seconds = num_frames // fps
    second_preds = np.zeros(num_seconds, dtype=np.int32)

    for sec in range(num_seconds):
        s = sec * fps
        e = (sec + 1) * fps
        frames = frame_predictions[s:e]

        immobile_ratio = frames.mean()

        if agg_threshold == 0.0:
            pred = 1 if immobile_ratio > 0 else 0
        else:
            pred = 1 if immobile_ratio >= agg_threshold else 0

        # Confidence gate: revert low-confidence immobility calls
        if (confidence_threshold is not None
                and frame_probabilities is not None
                and pred == 1):
            if frame_probabilities[s:e].mean() < confidence_threshold:
                pred = 0

        second_preds[sec] = pred

    return second_preds


# ---------------------------------------------------------------------------
# Stage 4: Minimum bout duration enforcement
# ---------------------------------------------------------------------------

def enforce_minimum_bout(
    predictions: np.ndarray,
    min_immobile: int = 2,
    min_mobile: int = 1,
) -> np.ndarray:
    """
    Remove bouts shorter than a minimum duration (in the units of the
    input array — typically seconds after aggregation).

    Short bouts are absorbed into the surrounding state.  This eliminates
    isolated 1-second immobility calls in an otherwise mobile stretch
    (and vice-versa), which are the main source of "irregularities"
    reported by downstream analysts.

    Args:
        predictions:   1-D int array of 0/1 labels (frame- or second-level).
        min_immobile:  Minimum consecutive immobile units to keep.
                       Shorter immobile bouts are flipped to mobile.
        min_mobile:    Minimum consecutive mobile units to keep.
                       Shorter mobile bouts are flipped to immobile.
                       Set to 1 (default) to only filter immobile flickers.

    Returns:
        cleaned: same shape/dtype, with short bouts absorbed.
    """
    if len(predictions) == 0:
        return predictions.copy()

    cleaned = predictions.copy()

    # Identify run-length encoded bouts: (value, start, length)
    bouts: list[tuple[int, int, int]] = []
    val = cleaned[0]
    start = 0
    for i in range(1, len(cleaned)):
        if cleaned[i] != val:
            bouts.append((val, start, i - start))
            val = cleaned[i]
            start = i
    bouts.append((val, start, len(cleaned) - start))

    # Remove short bouts (absorb into surrounding state)
    for val, start, length in bouts:
        if val == 1 and length < min_immobile:
            cleaned[start : start + length] = 0
        elif val == 0 and length < min_mobile:
            cleaned[start : start + length] = 1

    return cleaned


# ---------------------------------------------------------------------------
# Convenience: full post-processing pipeline
# ---------------------------------------------------------------------------

def postprocess_pipeline(
    frame_probs: np.ndarray,
    fps: int = 30,
    # Stage 1 – smoothing
    smooth: bool = True,
    smooth_method: str = "gaussian",
    smooth_kernel_seconds: float = 1.0,
    # Stage 2 – thresholding
    use_hysteresis: bool = True,
    hysteresis_high: float = 0.6,
    hysteresis_low: float = 0.4,
    decision_threshold: float = 0.5,
    # Stage 3 – aggregation
    agg_threshold: float = 0.5,
    confidence_threshold: float = None,
    # Stage 4 – bout filtering
    min_immobile_seconds: int = 2,
    min_mobile_seconds: int = 1,
) -> dict:
    """
    Run the full post-processing chain on raw frame-level probabilities.

    Returns a dict so callers can inspect intermediate results:
        {
            "smoothed_probs":      np.ndarray [num_frames],
            "frame_predictions":   np.ndarray [num_frames] int,
            "second_predictions":  np.ndarray [num_seconds] int,  (before bout filter)
            "final_predictions":   np.ndarray [num_seconds] int,  (after bout filter)
        }
    """
    # --- Stage 1: smooth ---
    if smooth:
        smoothed = smooth_probabilities(
            frame_probs, fps=fps,
            kernel_seconds=smooth_kernel_seconds,
            method=smooth_method,
        )
    else:
        smoothed = frame_probs.copy()

    # --- Stage 2: threshold ---
    if use_hysteresis:
        frame_preds = hysteresis_threshold(
            smoothed, high=hysteresis_high, low=hysteresis_low,
        )
    else:
        frame_preds = (smoothed >= decision_threshold).astype(np.int32)

    # --- Stage 3: aggregate ---
    second_preds = aggregate_frames_to_seconds(
        frame_predictions=frame_preds,
        frame_probabilities=smoothed,
        fps=fps,
        agg_threshold=agg_threshold,
        confidence_threshold=confidence_threshold,
    )

    # --- Stage 4: minimum bout ---
    final_preds = enforce_minimum_bout(
        second_preds,
        min_immobile=min_immobile_seconds,
        min_mobile=min_mobile_seconds,
    )

    return {
        "smoothed_probs": smoothed,
        "frame_predictions": frame_preds,
        "second_predictions": second_preds,
        "final_predictions": final_preds,
    }