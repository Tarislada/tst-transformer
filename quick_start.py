#!/usr/bin/env python3
"""
Quick Start Helper Script

Guides you through the complete pipeline setup and execution.
"""

import sys
from pathlib import Path
import subprocess


def print_header(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n→ {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    response = input("\nRun this command? [y/n]: ")
    if response.lower() != 'y':
        print("  Skipped.")
        return False
    
    try:
        subprocess.run(cmd, check=True)
        print("  ✓ Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print_header("TST TRANSFORMER - QUICK START GUIDE")
    
    print("""
This script will guide you through:
1. Converting Excel labels to CSV
2. Extracting YOLO features from videos
3. Training the transformer model
4. Running inference on new videos

You can skip any step you've already completed.
""")
    
    # Get user inputs
    print("\nFirst, let's gather some information about your setup:\n")
    
    video_dir = input("Path to video directory (e.g., data/videos): ").strip()
    if not Path(video_dir).exists():
        print(f"✗ Directory not found: {video_dir}")
        return
    
    label_excel_dir = input("Path to Excel labels directory (e.g., data/labels_excel): ").strip()
    yolo_path = input("Path to YOLO model (e.g., models/best.pt): ").strip()
    if not Path(yolo_path).exists():
        print(f"✗ YOLO model not found: {yolo_path}")
        return
    
    # Step 1: Convert labels
    print_header("STEP 1: CONVERT LABELS")
    
    label_csv_dir = "data/labels_csv"
    cmd = [
        "python", "../convert_labels_from_excel.py", "convert-batch",
        "--excel-glob", f"{label_excel_dir}/*.xlsx",
        "--out-dir", label_csv_dir
    ]
    
    run_command(cmd, "Convert Excel labels to CSV format")
    
    # Step 2: Inspect YOLO layers (optional)
    print_header("STEP 2: INSPECT YOLO LAYERS (OPTIONAL)")
    
    response = input("\nDo you want to inspect available YOLO layers? [y/n]: ")
    if response.lower() == 'y':
        cmd = ["python", "preprocess.py", "inspect", yolo_path]
        subprocess.run(cmd)
        print("\nRecommended: model.18 (balanced semantic/spatial features)")
        layer_name = input("Which layer to use? [model.18]: ").strip() or "model.18"
    else:
        layer_name = "model.18"
    
    # Step 3: Extract features
    print_header("STEP 3: EXTRACT YOLO FEATURES")
    
    feature_dir = "data/features"
    cmd = [
        "python", "preprocess.py",
        "--video_dir", video_dir,
        "--output_dir", feature_dir,
        "--yolo_path", yolo_path,
        "--layer_name", layer_name,
        "--imgsz", "1080",
        "--batch_size", "8"
    ]
    
    print("\n⚠ WARNING: This will take a long time (3-4 hours for 71 videos)")
    run_command(cmd, "Extract YOLO features from all videos")
    
    # Step 4: Train model
    print_header("STEP 4: TRAIN TRANSFORMER MODEL")
    
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    cmd = [
        "python", "train.py",
        "--feature_dir", feature_dir,
        "--label_dir", label_csv_dir,
        "--output_dir", checkpoint_dir,
        "--log_dir", log_dir,
        "--batch_size", "32",
        "--lr", "1e-4",
        "--epochs", "50",
        "--hidden_dim", "256",
        "--num_layers", "2",
        "--num_heads", "4",
        "--window_seconds", "2.0",
        "--use_class_weights"
    ]
    
    print(f"\nYou can monitor training with: tensorboard --logdir {log_dir}")
    success = run_command(cmd, "Train the transformer classifier")
    
    if not success:
        print("\n✗ Training failed or was skipped.")
        return
    
    # Step 5: Inference
    print_header("STEP 5: RUN INFERENCE")
    
    print("\nYou can now run inference on new videos!")
    print("\nExample command:")
    print(f"""
python inference.py \\
    --video {video_dir}/KHC2_M1_D10.mp4 \\
    --model {checkpoint_dir}/best_model.pt \\
    --yolo_path {yolo_path} \\
    --output predictions.csv
""")
    
    response = input("\nRun inference on a test video? [y/n]: ")
    if response.lower() == 'y':
        test_video = input("Path to test video: ").strip()
        if Path(test_video).exists():
            cmd = [
                "python", "inference.py",
                "--video", test_video,
                "--model", f"{checkpoint_dir}/best_model.pt",
                "--yolo_path", yolo_path,
                "--output", "test_predictions.csv"
            ]
            run_command(cmd, "Run inference on test video")
    
    print_header("SETUP COMPLETE!")
    print("""
Next steps:
1. Check predictions.csv for results
2. Monitor training: tensorboard --logdir logs
3. Tune hyperparameters if needed (see README.md)
4. Run batch inference on multiple videos

For detailed documentation, see README.md
""")


if __name__ == "__main__":
    main()
