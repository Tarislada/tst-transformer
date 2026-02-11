"""
Inference script for TST classification.

Takes a video as input and outputs per-second predictions in CSV format.

Usage:
    python inference.py --video path/to/video.mp4 --model checkpoints/best_model.pt --output predictions.csv
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
from postprocess import aggregate_frames_to_seconds


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


def predict_per_second(features: np.ndarray, model: TSTModel, window_seconds: float,
                       fps: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate per-second predictions from frame features.
    
    Args:
        features: [num_frames, feature_dim] numpy array
        model: Trained TST model
        window_seconds: Temporal window size in seconds
        fps: Frames per second
        device: Device to run inference on
        
    Returns:
        predictions: [num_seconds] array of predictions (0=mobile, 1=immobile)
        confidences: [num_seconds] array of confidence scores (0-1)
    """
    model.eval()
    
    num_frames = features.shape[0]
    window_frames = int(window_seconds * fps)
    num_seconds = num_frames // fps
    
    predictions = []
    confidences = []
    
    print(f"Generating predictions for {num_seconds} seconds...")
    
    with torch.no_grad():
        for second in tqdm(range(num_seconds), desc="Predicting"):
            # Get frame range for this second
            # Use past context (causal prediction)
            end_frame = min((second + 1) * fps, num_frames)
            start_frame = max(0, end_frame - window_frames)
            
            # Extract window features
            window_features = features[start_frame:end_frame]
            
            # Pad if necessary (for first few seconds)
            if window_features.shape[0] < window_frames:
                padding = np.zeros((window_frames - window_features.shape[0], features.shape[1]))
                window_features = np.concatenate([padding, window_features], axis=0)
            
            # Convert to tensor and add batch dimension
            window_tensor = torch.from_numpy(window_features).unsqueeze(0).float().to(device)
            
            # Predict
            logits = model(window_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
            conf = probs[0, pred].item()
            
            predictions.append(pred)
            confidences.append(conf)
    
    return np.array(predictions), np.array(confidences)


def predict_frames_with_aggregation(
    features: np.ndarray, 
    model: TSTModel, 
    window_size: int,
    fps: int,
    device: str,
    aggregate_method: str = 'majority',
    confidence_threshold: float = None  # Add this parameter
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate frame-level predictions and aggregate to seconds.
    
    This is the CORRECT way to do inference:
    1. Extract sliding windows from features
    2. Model predicts per-frame binary logits [B, T, 1]
    3. Average overlapping frame predictions
    4. Aggregate frames to seconds using postprocess.py
    
    Args:
        features: [num_frames, feature_dim] numpy array
        model: Trained TST model
        window_size: Window size in FRAMES (e.g., 120 frames)
        fps: Video frames per second
        device: Device to run inference on
        aggregate_method: Method to aggregate frames to seconds
        confidence_threshold: Minimum confidence to keep immobility predictions
        
    Returns:
        frame_predictions: [num_frames] array of 0/1 predictions
        frame_confidences: [num_frames] array of probabilities for predicted class
        second_predictions: [num_seconds] array of 0/1 predictions  
        second_confidences: [num_seconds] array of averaged frame confidences per second
    """
    model.eval()
    
    num_frames = features.shape[0]
    
    # Accumulate frame probabilities (for overlapping windows)
    frame_probs = np.zeros(num_frames, dtype=np.float32)
    frame_counts = np.zeros(num_frames, dtype=np.int32)
    
    print(f"Generating frame-level predictions...")
    print(f"  Total frames: {num_frames}")
    print(f"  Window size: {window_size} frames")
    print(f"  FPS: {fps}")
    
    with torch.no_grad():
        # Sliding window over all frames (NO stride - full coverage)
        for start_idx in tqdm(range(0, num_frames, window_size), desc="Predicting windows"):
            end_idx = min(start_idx + window_size, num_frames)
            window_features = features[start_idx:end_idx]
            
            # Pad if last window is incomplete
            actual_length = window_features.shape[0]
            if actual_length < window_size:
                padding = np.zeros((window_size - actual_length, features.shape[1]), dtype=np.float32)
                window_features = np.concatenate([window_features, padding], axis=0)
            
            # Convert to tensor [1, T, C]
            window_tensor = torch.from_numpy(window_features).unsqueeze(0).float().to(device)
            
            # Predict: model outputs [1, T, 1] binary logits
            logits = model(window_tensor)  # [1, window_size, 1]
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()[0]  # [window_size]
            
            # Accumulate only the valid frames (not padding)
            valid_probs = probs[:actual_length]
            frame_probs[start_idx:end_idx] += valid_probs
            frame_counts[start_idx:end_idx] += 1
    
    # Average overlapping predictions
    frame_probs = frame_probs / np.maximum(frame_counts, 1)
    
    # Threshold to get binary predictions
    frame_predictions = (frame_probs > 0.5).astype(int)
    
    # Confidence = probability of predicted class
    frame_confidences = np.where(
        frame_predictions == 1, 
        frame_probs,           # If pred=1, confidence = p(immobile)
        1.0 - frame_probs      # If pred=0, confidence = p(mobile)
    )
    
    print(f"✓ Frame predictions complete")
    print(f"  Immobile frames: {(frame_predictions == 1).sum()} ({100.0 * (frame_predictions == 1).mean():.1f}%)")
    
    # Aggregate to seconds with confidence filtering
    print(f"\nAggregating to seconds (method={aggregate_method})...")
    if confidence_threshold is not None:
        print(f"  Applying confidence threshold: {confidence_threshold}")
    
    second_predictions = aggregate_frames_to_seconds(
        frame_predictions=frame_predictions,
        frame_probabilities=frame_probs,  # Pass the probabilities
        fps=fps,
        method=aggregate_method,
        confidence_threshold=confidence_threshold  # Add this parameter
    )
    
    # Calculate per-second confidence as mean of frame confidences
    num_seconds = len(second_predictions)
    second_confidences = np.zeros(num_seconds, dtype=np.float32)
    
    for sec in range(num_seconds):
        start_frame = sec * fps
        end_frame = min((sec + 1) * fps, num_frames)
        second_confidences[sec] = frame_confidences[start_frame:end_frame].mean()
    
    print(f"✓ Second-level aggregation complete")
    print(f"  Total seconds: {num_seconds}")
    print(f"  Immobile seconds: {(second_predictions == 1).sum()} ({100.0 * (second_predictions == 1).mean():.1f}%)")
    
    return frame_predictions, frame_confidences, second_predictions, second_confidences


def save_predictions_csv(predictions: np.ndarray, confidences: np.ndarray, 
                         output_path: str, video_name: str = None, level: str = "second"):
    """
    Save predictions to CSV file.
    
    Args:
        level: "frame" or "second" - changes the index column name
    """
    time_col = 'frame' if level == 'frame' else 'second'
    
    df = pd.DataFrame({
        time_col: range(len(predictions)),
        'prediction': predictions.astype(int),
        'confidence': confidences
    })
    
    # Add metadata as comments
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


def main():
    parser = argparse.ArgumentParser(description="TST Inference - Predict immobility from video")
    
    # Required arguments
    parser.add_argument("--video", type=str, required=True,
                        help="Path to input video (.mp4)")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--yolo_path", type=str, required=True,
                        help="Path to YOLO model used for feature extraction")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <video_name>_predictions.csv)")
    parser.add_argument("--save_frame_predictions", action="store_true",
                        help="Also save frame-level predictions")
    
    # YOLO settings (should match training)
    parser.add_argument("--layer_name", type=str, default="model.18",
                        help="YOLO layer for feature extraction (must match training)")
    parser.add_argument("--imgsz", type=int, default=1088,
                        help="YOLO input size (must match YOLO training)")
    
    # Model settings (should match training)
    parser.add_argument("--window_size", type=int, default=120,
                        help="Temporal window size in FRAMES (must match training, e.g., 120)")
    parser.add_argument("--window_seconds", type=float, default=None,
                        help="Alternative: specify window in seconds (converted to frames using fps)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frames per second")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Model hidden dimension (must match training)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads (must match training)")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of transformer layers (must match training)")
    
    # Aggregation settings
    parser.add_argument("--aggregate_method", type=str, default='majority',
                        choices=['majority', 'any', 'all', 'mean_threshold'],
                        help="Method to aggregate frame predictions to seconds")
    parser.add_argument("--confidence_threshold", type=float, default=0.65,
                        help="Minimum average confidence per second to keep immobility predictions (e.g., 0.6). "
                             "Low-confidence immobility predictions are reverted to mobile.")
    
    # Processing settings
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    # Optional: Load config from training
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json from training (overrides individual params)")
    
    args = parser.parse_args()
    
    # Handle window_seconds option
    if args.window_seconds is not None:
        args.window_size = int(args.window_seconds * args.fps)
        print(f"Converting window_seconds={args.window_seconds}s to window_size={args.window_size} frames")
    
    # Validate paths
    if not Path(args.video).exists():
        raise ValueError(f"Video not found: {args.video}")
    
    if not Path(args.model).exists():
        raise ValueError(f"Model checkpoint not found: {args.model}")
    
    if not Path(args.yolo_path).exists():
        raise ValueError(f"YOLO model not found: {args.yolo_path}")
    
    # Load config if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Override args with config values
        for key, value in config.items():
            if hasattr(args, key) and key not in ['video', 'model', 'output', 'config', 'save_frame_predictions']:
                setattr(args, key, value)
        print(f"✓ Loaded config from {args.config}")
    
    # Generate output path if not provided
    if args.output is None:
        video_name = Path(args.video).stem
        args.output = f"{video_name}_predictions.csv"
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Extract features from video
    print("\n" + "=" * 60)
    print("STEP 1: FEATURE EXTRACTION")
    print("=" * 60)
    
    features = extract_features_from_video(
        video_path=args.video,
        yolo_path=args.yolo_path,
        layer_name=args.layer_name,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device
    )
    
    feature_dim = features.shape[1]
    print(f"✓ Features extracted: {features.shape}")
    
    # Step 2: Load trained model
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
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded from {args.model}")
    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch']} epochs")
    if 'val_accuracy' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # Step 3: Generate predictions
    print("\n" + "=" * 60)
    print("STEP 3: INFERENCE")
    print("=" * 60)
    
    frame_preds, frame_confs, second_preds, second_confs = predict_frames_with_aggregation(
        features=features,
        model=model,
        window_size=args.window_size,
        fps=args.fps,
        device=device,
        aggregate_method=args.aggregate_method,
        confidence_threshold=args.confidence_threshold  # Add this parameter
    )
    
    # Step 4: Save results
    print("\n" + "=" * 60)
    print("STEP 4: SAVE RESULTS")
    print("=" * 60)
    
    video_name = Path(args.video).name
    
    # Save second-level predictions (primary output)
    save_predictions_csv(second_preds, second_confs, args.output, video_name, level="second")
    
    # Optionally save frame-level predictions
    if args.save_frame_predictions:
        frame_output = args.output.replace('.csv', '_frames.csv')
        save_predictions_csv(frame_preds, frame_confs, frame_output, video_name, level="frame")
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
