"""
Preprocessing: Extract YOLO features from videos and save to disk.

This script processes all videos and extracts features from the YOLO model's neck layer.
Features are cached to .pt files for fast training.
"""

import os
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model import YOLOFeatureExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract YOLO features from TST videos")
    
    # Input/output paths
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing .mp4 videos")
    parser.add_argument("--output_dir", type=str, default="features",
                        help="Directory to save extracted features")
    
    # YOLO settings
    parser.add_argument("--yolo_path", type=str, required=True,
                        help="Path to trained YOLO model (.pt file)")
    parser.add_argument("--layer_name", type=str, default="model.18",
                        help="YOLO layer to extract features from (e.g., model.15, model.18, model.21)")
    parser.add_argument("--imgsz", type=int, default=1080,
                        help="Input image size for YOLO (should match training size)")
    
    # Processing settings
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing frames")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos that already have extracted features")
    
    args = parser.parse_args()
    
    # Validate paths
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise ValueError(f"Video directory not found: {video_dir}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(args.yolo_path).exists():
        raise ValueError(f"YOLO model not found: {args.yolo_path}")
    
    # Find all videos
    video_files = sorted(video_dir.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 videos found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Initialize YOLO feature extractor
    extractor = YOLOFeatureExtractor(
        yolo_path=args.yolo_path,
        layer_name=args.layer_name,
        device=args.device
    )
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = video_path.stem
        output_path = output_dir / f"{video_name}_features.pt"
        
        # Skip if already exists
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        
        try:
            print(f"\nProcessing: {video_name}")
            
            # Extract features
            features = extractor.extract_video_features(
                str(video_path),
                imgsz=args.imgsz,
                batch_size=args.batch_size
            )
            
            # Save features
            torch.save({
                'features': torch.from_numpy(features),
                'video_name': video_name,
                'video_path': str(video_path),
                'layer_name': args.layer_name,
                'feature_dim': features.shape[1],
                'num_frames': features.shape[0],
            }, output_path)
            
            processed += 1
            print(f"✓ Saved to {output_path}")
            
        except Exception as e:
            failed += 1
            print(f"✗ Failed to process {video_name}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(video_files)}")
    print("=" * 60)


def inspect_yolo_layers(yolo_path: str):
    """
    Helper function to inspect available layers in YOLO model.
    Run this to decide which layer to use for feature extraction.
    """
    from ultralytics import YOLO
    
    print(f"Loading YOLO model from {yolo_path}...")
    yolo = YOLO(yolo_path)
    model = yolo.model
    
    print("\nAvailable layers:")
    print("=" * 60)
    
    def print_layers(module, prefix="", depth=0, max_depth=3):
        if depth > max_depth:
            return
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            print(f"{'  ' * depth}{full_name}: {child.__class__.__name__}")
            
            # If it's a Sequential/ModuleList, show numbered children
            if isinstance(child, (torch.nn.Sequential, torch.nn.ModuleList)):
                for i, subchild in enumerate(child):
                    subname = f"{full_name}.{i}"
                    print(f"{'  ' * (depth+1)}{subname}: {subchild.__class__.__name__}")
            else:
                print_layers(child, full_name, depth + 1, max_depth)
    
    print_layers(model, "model")
    print("=" * 60)
    print("\nCommon neck layers for YOLOv8/v11/v12 segmentation:")
    print("  - model.15: Early neck features (high resolution)")
    print("  - model.18: Mid neck features (balanced)")
    print("  - model.21: Late neck features (semantic)")
    print("\nRun feature extraction with different --layer_name values to compare.")


if __name__ == "__main__":
    # Check if running in inspection mode
    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        if len(sys.argv) < 3:
            print("Usage: python preprocess.py inspect <yolo_model_path>")
            sys.exit(1)
        inspect_yolo_layers(sys.argv[2])
    else:
        main()
