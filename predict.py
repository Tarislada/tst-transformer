"""
Inference script for TST Transformer with postprocessing.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from model import TSTModel
from postprocess import aggregate_frames_to_seconds


def predict_video(
    model: torch.nn.Module,
    features: torch.Tensor,
    window_size: int,
    stride: int,
    device: str = 'cuda'
) -> np.ndarray:
    """Predict all frames using sliding windows."""
    model.eval()
    num_frames = features.shape[0]
    
    # Binary predictions - only track probability of class 1
    probs = np.zeros(num_frames)
    counts = np.zeros(num_frames)
    
    with torch.no_grad():
        for start in range(0, num_frames - window_size + 1, stride):
            window = features[start:start + window_size].unsqueeze(0).to(device)
            
            logits = model(window)  # [1, T, 1]
            window_probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()[0]  # [T]
            
            probs[start:start + window_size] += window_probs
            counts[start:start + window_size] += 1
    
    probs = probs / counts
    frame_predictions = (probs > 0.5).astype(int)
    
    return frame_predictions


def main():
    parser = argparse.ArgumentParser(description="Predict TST labels for videos")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with feature .pt files")
    parser.add_argument("--output_dir", type=str, default="../predictions",
                        help="Output directory for predictions")
    parser.add_argument("--window_size", type=int, default=120,
                        help="Window size (must match training)")
    parser.add_argument("--stride", type=int, default=60,
                        help="Stride (must match training)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Video FPS for second-level aggregation")
    parser.add_argument("--aggregate_method", type=str, default='majority',
                        choices=['majority', 'any', 'all', 'mean_threshold'],
                        help="Method to aggregate frames to seconds")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get feature dimension from first feature file
    feature_files = list(Path(args.feature_dir).glob("*.pt"))
    sample_features = torch.load(feature_files[0], weights_only=True)
    if isinstance(sample_features, dict):
        sample_features = sample_features['features']
    feature_dim = sample_features.shape[1]
    
    # Create model
    model = TSTModel(yolo_feature_dim=feature_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']}, val_acc={checkpoint['val_accuracy']:.2f}%)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    for feat_path in feature_files:
        print(f"\nProcessing {feat_path.name}...")
        
        # Load features
        features = torch.load(feat_path, weights_only=True)
        if isinstance(features, dict):
            features = features['features']
        
        # Predict frame-level
        frame_preds = predict_video(
            model, features, args.window_size, args.stride, args.device
        )
        
        # Aggregate to seconds
        second_preds = aggregate_frames_to_seconds(
            frame_preds, args.fps, method=args.aggregate_method
        )
        
        # Save frame-level predictions
        frame_csv = output_dir / f"{feat_path.stem.replace('_features', '')}_frame_predictions.csv"
        pd.DataFrame({
            'frame': range(len(frame_preds)),
            'prediction': frame_preds
        }).to_csv(frame_csv, index=False)
        
        # Save second-level predictions
        second_csv = output_dir / f"{feat_path.stem.replace('_features', '')}_second_predictions.csv"
        pd.DataFrame({
            'second': range(len(second_preds)),
            'prediction': second_preds
        }).to_csv(second_csv, index=False)
        
        print(f"  ✓ Saved to {frame_csv} and {second_csv}")
        print(f"    Frames: {len(frame_preds)} | Seconds: {len(second_preds)}")
        print(f"    Immobile frames: {(frame_preds == 1).sum()} ({100*(frame_preds == 1).mean():.1f}%)")
        print(f"    Immobile seconds: {(second_preds == 1).sum()} ({100*(second_preds == 1).mean():.1f}%)")
    
    print(f"\n✓ All predictions saved to {output_dir}")


if __name__ == "__main__":
    main()