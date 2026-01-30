"""
Batch inference script - process multiple videos at once.

Usage:
    python batch_inference.py --video_dir videos/ --model checkpoints/best_model.pt --yolo_path best.pt
"""

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm
import json


def main():
    parser = argparse.ArgumentParser(description="Batch TST inference on multiple videos")
    
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing .mp4 videos")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--yolo_path", type=str, required=True,
                        help="Path to YOLO model")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="Directory to save prediction CSVs")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json from training")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip videos with existing predictions")
    
    args = parser.parse_args()
    
    # Find all videos
    video_dir = Path(args.video_dir)
    video_files = sorted(video_dir.glob("*.mp4"))
    
    if not video_files:
        print(f"No .mp4 videos found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = video_path.stem
        output_path = output_dir / f"{video_name}_predictions.csv"
        
        # Skip if exists
        if args.skip_existing and output_path.exists():
            skipped += 1
            continue
        
        # Build command
        cmd = [
            "python", "inference.py",
            "--video", str(video_path),
            "--model", args.model,
            "--yolo_path", args.yolo_path,
            "--output", str(output_path),
            "--device", args.device,
        ]
        
        if args.config:
            cmd.extend(["--config", args.config])
        
        # Run inference
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Failed: {video_name}")
            print(e.stderr)
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(video_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
