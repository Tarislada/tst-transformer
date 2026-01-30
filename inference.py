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


def save_predictions_csv(predictions: np.ndarray, confidences: np.ndarray, 
                         output_path: str, video_name: str = None):
    """
    Save predictions to CSV file.
    
    Format:
        second,prediction,confidence
        0,0,0.95
        1,0,0.87
        2,1,0.92
    """
    df = pd.DataFrame({
        'second': range(len(predictions)),
        'prediction': predictions.astype(int),
        'confidence': confidences
    })
    
    # Add metadata as comments
    with open(output_path, 'w') as f:
        if video_name:
            f.write(f"# Video: {video_name}\n")
        f.write(f"# Total seconds: {len(predictions)}\n")
        f.write(f"# Mobile (0): {(predictions == 0).sum()}\n")
        f.write(f"# Immobile (1): {(predictions == 1).sum()}\n")
        f.write("# prediction: 0=mobile, 1=immobile\n")
        f.write("# confidence: probability of predicted class\n")
        df.to_csv(f, index=False)
    
    print(f"✓ Saved predictions to {output_path}")
    print(f"  Total seconds: {len(predictions)}")
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
    
    # YOLO settings (should match training)
    parser.add_argument("--layer_name", type=str, default="model.18",
                        help="YOLO layer for feature extraction (must match training)")
    parser.add_argument("--imgsz", type=int, default=1080,
                        help="YOLO input size (must match YOLO training)")
    
    # Model settings (should match training)
    parser.add_argument("--window_seconds", type=float, default=2.0,
                        help="Temporal window size (must match training)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video frames per second")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Model hidden dimension (must match training)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads (must match training)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers (must match training)")
    
    # Processing settings
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    # Optional: Load config from training
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json from training (overrides individual params)")
    
    args = parser.parse_args()
    
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
            if hasattr(args, key) and key not in ['video', 'model', 'output', 'config']:
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
    
    predictions, confidences = predict_per_second(
        features=features,
        model=model,
        window_seconds=args.window_seconds,
        fps=args.fps,
        device=device
    )
    
    # Step 4: Save results
    print("\n" + "=" * 60)
    print("STEP 4: SAVE RESULTS")
    print("=" * 60)
    
    video_name = Path(args.video).name
    save_predictions_csv(predictions, confidences, args.output, video_name)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
