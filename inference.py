"""
Inference script for TST classification.

Takes a video as input and outputs per-second predictions in CSV format.

Pipeline:
    Video → YOLO features → Transformer (frame probs) → Post-processing → CSV

Post-processing stages (all optional, all enabled by default):
    1. Probability smoothing  (--smooth / --no_smooth)
    2. Hysteresis threshold   (--use_hysteresis / --no_hysteresis)
    3. Frame → second agg.   (--agg_threshold)
    4. Min bout enforcement   (--min_immobile_seconds)

Usage:
    # Recommended (new pipeline with smoothing + hysteresis + bout filter):
    python inference.py --video path/to/video.mp4 --model best_model.pt \\
        --yolo_path best.pt --config config.json

    # Legacy behaviour (disable all new post-processing):
    python inference.py --video path/to/video.mp4 --model best_model.pt \\
        --yolo_path best.pt --config config.json \\
        --no_smooth --no_hysteresis --min_immobile_seconds 0
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model import YOLOFeatureExtractor, TSTModel
from postprocess import postprocess_pipeline, aggregate_frames_to_seconds


# ============================================================================
# Feature extraction
# ============================================================================

def extract_features_from_video(video_path: str, yolo_path: str, layer_name: str,
                                 imgsz: int, batch_size: int, device: str) -> np.ndarray:
    """
    Extract YOLO features from a video.

    Returns:
        features: [num_frames, feature_dim] numpy array
    """
    print(f"Extracting features from {video_path}...")

    extractor = YOLOFeatureExtractor(
        yolo_path=yolo_path,
        layer_name=layer_name,
        device=device
    )

    features = extractor.extract_video_features(
        video_path=video_path,
        imgsz=imgsz,
        batch_size=batch_size
    )

    return features


# ============================================================================
# Model inference  (produces raw frame probabilities — no thresholding)
# ============================================================================

def predict_frame_probabilities(
    features: np.ndarray,
    model: TSTModel,
    window_size: int,
    device: str,
) -> np.ndarray:
    """
    Run the Transformer over sliding windows and return averaged
    per-frame P(immobile).

    This function does NO thresholding or aggregation — it outputs the
    continuous probability signal that downstream post-processing operates on.

    Args:
        features:    [num_frames, feature_dim] numpy array.
        model:       Trained TSTModel.
        window_size: Window size in frames (must match training).
        device:      Torch device string.

    Returns:
        frame_probs: [num_frames] float32 array, P(immobile) per frame.
    """
    model.eval()
    num_frames = features.shape[0]

    # Accumulators for overlapping window averaging
    frame_probs = np.zeros(num_frames, dtype=np.float64)
    frame_counts = np.zeros(num_frames, dtype=np.int32)

    print(f"Generating frame-level probabilities...")
    print(f"  Total frames: {num_frames}")
    print(f"  Window size:  {window_size} frames")

    with torch.no_grad():
        for start_idx in tqdm(range(0, num_frames, window_size), desc="Predicting windows"):
            end_idx = min(start_idx + window_size, num_frames)
            window_features = features[start_idx:end_idx]

            # Pad last window if incomplete
            actual_length = window_features.shape[0]
            if actual_length < window_size:
                padding = np.zeros(
                    (window_size - actual_length, features.shape[1]),
                    dtype=np.float32
                )
                window_features = np.concatenate([window_features, padding], axis=0)

            # [1, T, C]
            window_tensor = (
                torch.from_numpy(window_features)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            # Model outputs [1, T, 1] binary logits
            logits = model(window_tensor)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()[0]  # [T]

            # Only accumulate valid (non-padded) frames
            valid_probs = probs[:actual_length]
            frame_probs[start_idx:end_idx] += valid_probs
            frame_counts[start_idx:end_idx] += 1

    # Average where windows overlapped
    frame_probs = frame_probs / np.maximum(frame_counts, 1)

    print(f"✓ Frame probabilities computed")
    print(f"  Mean P(immobile): {frame_probs.mean():.3f}")

    return frame_probs.astype(np.float32)


# ============================================================================
# Legacy wrapper (backward compat — calls old code path)
# ============================================================================

def predict_frames_with_aggregation(
    features: np.ndarray,
    model: TSTModel,
    window_size: int,
    fps: int,
    device: str,
    agg_threshold: float = 0.5,
    confidence_threshold: float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy inference path: raw threshold + aggregate, no smoothing.

    Retained for backward compatibility. New code should use
    ``predict_frame_probabilities`` + ``postprocess_pipeline``.
    """
    frame_probs = predict_frame_probabilities(features, model, window_size, device)

    # Hard threshold (old behaviour)
    frame_predictions = (frame_probs > 0.5).astype(np.int32)
    frame_confidences = np.where(
        frame_predictions == 1,
        frame_probs,
        1.0 - frame_probs
    )

    print(f"  Immobile frames: {(frame_predictions == 1).sum()} "
          f"({100.0 * (frame_predictions == 1).mean():.1f}%)")

    # Aggregate
    second_predictions = aggregate_frames_to_seconds(
        frame_predictions=frame_predictions,
        frame_probabilities=frame_probs,
        fps=fps,
        agg_threshold=agg_threshold,
        confidence_threshold=confidence_threshold,
    )

    num_seconds = len(second_predictions)
    num_frames = len(frame_probs)
    second_confidences = np.zeros(num_seconds, dtype=np.float32)
    for sec in range(num_seconds):
        s = sec * fps
        e = min((sec + 1) * fps, num_frames)
        second_confidences[sec] = frame_confidences[s:e].mean()

    return frame_predictions, frame_confidences, second_predictions, second_confidences


# ============================================================================
# CSV output
# ============================================================================

def save_predictions_csv(predictions: np.ndarray, confidences: np.ndarray,
                         output_path: str, video_name: str = None,
                         level: str = "second"):
    """
    Save predictions to CSV file.

    Args:
        level: "frame" or "second" — changes the index column name.
    """
    time_col = 'frame' if level == 'frame' else 'second'

    df = pd.DataFrame({
        time_col: range(len(predictions)),
        'prediction': predictions.astype(int),
        'confidence': confidences
    })

    with open(output_path, 'w') as f:
        if video_name:
            f.write(f"# Video: {video_name}\n")
        f.write(f"# Total {level}s: {len(predictions)}\n")
        f.write(f"# Mobile (0): {(predictions == 0).sum()}\n")
        f.write(f"# Immobile (1): {(predictions == 1).sum()}\n")
        f.write("# prediction: 0=mobile, 1=immobile\n")
        f.write("# confidence: probability of predicted class\n")
        df.to_csv(f, index=False)

    print(f"✓ Saved {level}-level predictions to {output_path}")
    print(f"  Mobile: {(predictions == 0).sum()} ({100.0 * (predictions == 0).mean():.1f}%)")
    print(f"  Immobile: {(predictions == 1).sum()} ({100.0 * (predictions == 1).mean():.1f}%)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TST Inference — Predict immobility from video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required paths ---
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video (.mp4)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--yolo_path", type=str, required=True,
                        help="Path to YOLO model used for feature extraction")

    # --- Output ---
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <video_name>_predictions.csv)")
    parser.add_argument("--save_frame_predictions", action="store_true",
                        help="Also save frame-level predictions CSV")

    # --- YOLO settings (must match training) ---
    parser.add_argument("--layer_name", type=str, default="model.18",
                        help="YOLO layer for feature extraction")
    parser.add_argument("--imgsz", type=int, default=1088,
                        help="YOLO input size")

    # --- Model architecture (must match training) ---
    parser.add_argument("--window_size", type=int, default=120,
                        help="Temporal window size in FRAMES")
    parser.add_argument("--window_seconds", type=float, default=None,
                        help="Alternative: window in seconds (converted via fps)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frames per second")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Model hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers")

    # --- Post-processing: Stage 1 — Probability smoothing ---
    parser.add_argument("--smooth", action="store_true", default=True,
                        help="Enable probability-domain temporal smoothing (default: on)")
    parser.add_argument("--no_smooth", action="store_true",
                        help="Disable probability smoothing")
    parser.add_argument("--smooth_method", type=str, default="gaussian",
                        choices=["gaussian", "uniform", "median"],
                        help="Smoothing kernel type")
    parser.add_argument("--smooth_kernel_seconds", type=float, default=1.0,
                        help="Smoothing kernel width in seconds")

    # --- Post-processing: Stage 2 — Thresholding ---
    parser.add_argument("--use_hysteresis", action="store_true", default=True,
                        help="Use dual-threshold hysteresis (default: on)")
    parser.add_argument("--no_hysteresis", action="store_true",
                        help="Disable hysteresis; use single decision_threshold")
    parser.add_argument("--hysteresis_high", type=float, default=0.6,
                        help="P(immobile) threshold to ENTER immobility")
    parser.add_argument("--hysteresis_low", type=float, default=0.4,
                        help="P(immobile) threshold to EXIT immobility")
    parser.add_argument("--decision_threshold", type=float, default=0.5,
                        help="Single threshold when hysteresis is off")

    # --- Post-processing: Stage 3 — Frame→second aggregation ---
    parser.add_argument("--agg_threshold", type=float, default=0.5,
                        help="Aggregation threshold (0.0=any, 0.5=majority, 1.0=all)")
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                        help="Min avg P(immobile) per second to keep immobility call. "
                             "Set to 0 to disable.")

    # --- Post-processing: Stage 4 — Minimum bout duration ---
    parser.add_argument("--min_immobile_seconds", type=int, default=2,
                        help="Minimum consecutive immobile seconds to keep (shorter → mobile). "
                             "Set to 0 to disable.")
    parser.add_argument("--min_mobile_seconds", type=int, default=1,
                        help="Minimum consecutive mobile seconds to keep (shorter → immobile). "
                             "Set to 1 to effectively disable.")

    # --- Processing ---
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    # --- Config override ---
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json from training (overrides model arch params)")

    # --- Debug ---
    parser.add_argument("--debug", action="store_true",
                        help="Enable debugpy remote debugging")
    parser.add_argument("--debug_port", type=int, default=5678,
                        help="Port for debugpy")

    args = parser.parse_args()

    # ---- Debugpy ----
    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", args.debug_port))
        print(f"🐛 Waiting for debugger to attach on port {args.debug_port}...")
        debugpy.wait_for_client()
        print("✓ Debugger attached!")

    # ---- Resolve --no_* flags ----
    if args.no_smooth:
        args.smooth = False
    if args.no_hysteresis:
        args.use_hysteresis = False

    # ---- Handle window_seconds ----
    if args.window_seconds is not None:
        args.window_size = int(args.window_seconds * args.fps)
        print(f"Converting window_seconds={args.window_seconds}s "
              f"to window_size={args.window_size} frames")

    # ---- Validate paths ----
    if not Path(args.video).exists():
        raise ValueError(f"Video not found: {args.video}")
    if not Path(args.model).exists():
        raise ValueError(f"Model checkpoint not found: {args.model}")
    if not Path(args.yolo_path).exists():
        raise ValueError(f"YOLO model not found: {args.yolo_path}")

    # ---- Load config (overrides model-architecture params only) ----
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Only override model / YOLO architecture keys, not post-processing
        arch_keys = {
            'window_size', 'fps', 'hidden_dim', 'num_heads', 'num_layers',
            'layer_name', 'imgsz', 'yolo_path',
        }
        for key, value in config.items():
            if key in arch_keys and hasattr(args, key):
                setattr(args, key, value)
        print(f"✓ Loaded architecture config from {args.config}")

    # ---- Output path ----
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"{video_name}_predictions.csv"

    # ---- Device ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==================================================================
    # STEP 1: Feature extraction
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 1: FEATURE EXTRACTION")
    print("=" * 60)

    features = extract_features_from_video(
        video_path=args.video,
        yolo_path=args.yolo_path,
        layer_name=args.layer_name,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
    )
    feature_dim = features.shape[1]
    print(f"✓ Features extracted: {features.shape}")

    # ==================================================================
    # STEP 2: Load model
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 2: LOAD MODEL")
    print("=" * 60)

    model = TSTModel(
        yolo_feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_length=args.window_size,
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Model loaded from {args.model}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'val_accuracy' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_accuracy']:.2f}%")

    # ==================================================================
    # STEP 3: Frame-level inference (raw probabilities)
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 3: INFERENCE")
    print("=" * 60)

    frame_probs = predict_frame_probabilities(
        features=features,
        model=model,
        window_size=args.window_size,
        device=str(device),
    )

    # ==================================================================
    # STEP 4: Post-processing pipeline
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 4: POST-PROCESSING")
    print("=" * 60)

    # Resolve confidence_threshold=0 → disabled
    conf_thresh = args.confidence_threshold if args.confidence_threshold > 0 else None

    pp = postprocess_pipeline(
        frame_probs=frame_probs,
        fps=args.fps,
        # Stage 1
        smooth=args.smooth,
        smooth_method=args.smooth_method,
        smooth_kernel_seconds=args.smooth_kernel_seconds,
        # Stage 2
        use_hysteresis=args.use_hysteresis,
        hysteresis_high=args.hysteresis_high,
        hysteresis_low=args.hysteresis_low,
        decision_threshold=args.decision_threshold,
        # Stage 3
        agg_threshold=args.agg_threshold,
        confidence_threshold=conf_thresh,
        # Stage 4
        min_immobile_seconds=max(0, args.min_immobile_seconds),
        min_mobile_seconds=max(1, args.min_mobile_seconds),
    )

    frame_predictions = pp["frame_predictions"]
    second_predictions_raw = pp["second_predictions"]
    final_predictions = pp["final_predictions"]
    smoothed_probs = pp["smoothed_probs"]

    # Build per-frame and per-second confidence arrays
    frame_confidences = np.where(
        frame_predictions == 1,
        smoothed_probs,
        1.0 - smoothed_probs,
    )

    num_seconds = len(final_predictions)
    num_frames = len(frame_probs)
    second_confidences = np.zeros(num_seconds, dtype=np.float32)
    for sec in range(num_seconds):
        s = sec * args.fps
        e = min((sec + 1) * args.fps, num_frames)
        second_confidences[sec] = frame_confidences[s:e].mean()

    # Summary
    print(f"  Smoothing:         {'ON (' + args.smooth_method + ', ' + str(args.smooth_kernel_seconds) + 's)' if args.smooth else 'OFF'}")
    print(f"  Hysteresis:        {'ON (low=' + str(args.hysteresis_low) + ', high=' + str(args.hysteresis_high) + ')' if args.use_hysteresis else 'OFF (threshold=' + str(args.decision_threshold) + ')'}")
    print(f"  Aggregation:       threshold={args.agg_threshold}")
    if conf_thresh:
        print(f"  Confidence gate:   {conf_thresh}")
    print(f"  Min bout filter:   immobile≥{args.min_immobile_seconds}s, mobile≥{args.min_mobile_seconds}s")
    print()
    print(f"  Frame predictions: {(frame_predictions == 1).sum()} immobile / {len(frame_predictions)} total "
          f"({100.0 * (frame_predictions == 1).mean():.1f}%)")
    print(f"  Second preds (raw):  {(second_predictions_raw == 1).sum()} immobile / {num_seconds} total")
    print(f"  Second preds (final): {(final_predictions == 1).sum()} immobile / {num_seconds} total")

    bout_diff = int((second_predictions_raw == 1).sum()) - int((final_predictions == 1).sum())
    if bout_diff != 0:
        print(f"  Bout filter removed {abs(bout_diff)} immobile second(s)")

    # ==================================================================
    # STEP 5: Save results
    # ==================================================================
    print("\n" + "=" * 60)
    print("STEP 5: SAVE RESULTS")
    print("=" * 60)

    video_name = Path(args.video).name

    # Primary output: second-level final predictions
    save_predictions_csv(
        final_predictions, second_confidences,
        args.output, video_name, level="second",
    )

    # Optional: frame-level predictions
    if args.save_frame_predictions:
        frame_output = args.output.replace('.csv', '_frames.csv')
        save_predictions_csv(
            frame_predictions, frame_confidences,
            frame_output, video_name, level="frame",
        )

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()