# TST Transformer Workflow - Complete Guide

## Overview

This document provides a complete walkthrough of the TST classification pipeline, from raw data to final predictions.

## System Architecture

```
┌─────────────────┐
│   Raw Videos    │ (.mp4)
│  + Excel Labels │ (.xlsx)
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ 1. Label Conversion     │
│    Excel → CSV          │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 2. Feature Extraction   │
│    Video → YOLO → .pt   │
│    (Offline, cached)    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 3. Model Training       │
│    Features + Labels    │
│    → Trained Model      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ 4. Inference            │
│    New Video → CSV      │
└─────────────────────────┘
```

## Detailed Workflow

### Phase 1: Data Preparation

#### 1.1 Organize Your Data

```
project/
├── data/
│   ├── videos/                    # Raw .mp4 videos
│   │   ├── KHC2_M1_D10.mp4
│   │   ├── KHC2_M1_D11.mp4
│   │   └── ...
│   ├── labels_excel/             # Excel label files
│   │   ├── Filtered_TST_Manual_Scoring_D10_M1.xlsx
│   │   ├── Filtered_TST_Manual_Scoring_D11_M1.xlsx
│   │   └── ...
│   ├── labels_csv/               # [Generated] Converted labels
│   ├── features/                 # [Generated] Extracted features
│   └── predictions/              # [Generated] Final outputs
├── models/
│   └── best.pt                   # Your trained YOLO model
└── tst_transformer/              # Code directory
    ├── model.py
    ├── dataset.py
    ├── preprocess.py
    ├── train.py
    ├── inference.py
    └── ...
```

#### 1.2 Label Format Requirements

Your Excel files should have a "Filtered Data" sheet with:
- **Column "Second"**: Integer seconds (1, 2, 3, ...)
- **Column "Mobility Status"**: Text ("Mobile" or "Immobile")
- OR **Column "Mobility Status_num"**: Numeric (0=Mobile, 1=Immobile)

After conversion, CSV format will be:
```csv
second,label
0,0      # Mobile
1,0      # Mobile
2,1      # Immobile
...
```

### Phase 2: Preprocessing

#### 2.1 Convert Labels

```bash
cd tst_transformer

python convert_labels_from_excel.py convert-batch \
    --excel-glob "../data/labels_excel/*.xlsx" \
    --out-dir "../data/labels_csv"
```

**Expected output:**
- Creates one `.labels.csv` file per Excel file
- Files named: `D10_M1.labels.csv`, `D11_M1.labels.csv`, etc.
- Console shows: Mobile/Immobile counts per file

#### 2.2 Inspect YOLO Layers (Do this once!)

```bash
python preprocess.py inspect /path/to/best.pt
```

**What to look for:**
- You'll see a tree of layers like:
  ```
  model.15: C2f
  model.18: C2f  ← Recommended
  model.21: C2f
  ```
- **Recommendation**: Start with `model.18`
- **Why?**: It's in the middle of the neck, balancing spatial detail and semantic understanding

**Testing different layers:**

If `model.18` doesn't work well, try others:

1. Extract features with different layers:
   ```bash
   python preprocess.py --layer_name model.15 ...
   python preprocess.py --layer_name model.18 ...
   python preprocess.py --layer_name model.21 ...
   ```

2. Train separate models on each

3. Compare validation accuracy

**Rule of thumb:**
- Earlier layers (15) → More spatial detail, less semantic
- Later layers (21) → More semantic, less spatial detail
- Middle layers (18) → Balanced ← **Start here**

#### 2.3 Extract YOLO Features

```bash
python preprocess.py \
    --video_dir ../data/videos \
    --output_dir ../data/features \
    --yolo_path /home/tarislada/Documents/Extra_python_projects/Natalie/tst_transformer/model/best.pt \
    --layer_name model.18 \
    --imgsz 1088 \
    --batch_size 8 \
    --device cuda \
    --skip_existing
```

**Time estimate:**
- ~5-10 seconds per 3-minute video
- For 71 videos: **~10-15 minutes total**

**What this does:**
1. Loads YOLO model and freezes weights
2. For each video:
   - Reads all frames
   - Runs YOLO segmentation
   - Extracts features from the specified neck layer
   - Pools features using the largest detected mask
   - Saves to `.pt` file
3. Each `.pt` file contains:
   - `features`: [num_frames, feature_dim] tensor
   - Metadata: video name, layer name, etc.

**Troubleshooting:**
- **OOM error**: Reduce `--batch_size` to 4 or 2
- **No mask detected**: Check your YOLO model is working
- **Wrong size error**: Ensure `--imgsz` matches YOLO training

### Phase 3: Training

#### 3.1 Train the Model

```bash
python train.py \
    --feature_dir ../data/features \
    --label_dir ../data/labels_csv \
    --output_dir ../checkpoints \
    --log_dir ../logs \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --hidden_dim 256 \
    --num_layers 2 \
    --num_heads 4 \
    --window_seconds 2.0 \
    --fps 30 \
    --use_class_weights \
    --patience 10 \
    --device cuda
```

**What happens during training:**

1. **Data Loading:**
   - Finds matching feature/label pairs
   - Splits into train/val (80/20 by default)
   - Creates temporal windows (2-second sequences)

2. **Model Creation:**
   - Transformer with specified architecture
   - ~500K-1M trainable parameters
   - YOLO features are frozen (pre-extracted)

3. **Training Loop:**
   - Each epoch processes all training samples
   - Validates on held-out videos
   - Saves checkpoint if validation improves
   - Early stopping if no improvement for 10 epochs

4. **Outputs:**
   - `checkpoints/best_model.pt`: Best model by validation accuracy
   - `checkpoints/checkpoint_epochN.pt`: Periodic checkpoints
   - `checkpoints/config.json`: Configuration for inference
   - `logs/`: TensorBoard logs

#### 3.2 Monitor Training

In a separate terminal:
```bash
tensorboard --logdir logs
```

Then open `http://localhost:6006` in your browser.

**What to watch:**
- **Train/Val Loss**: Should decrease steadily
- **Train/Val Accuracy**: Should increase
- **Val/Mobile_Acc** and **Val/Immobile_Acc**: Check class-specific performance
- **LR**: Learning rate (will decrease if validation plateaus)

**Expected results:**
- Training time: ~2-3 minutes per epoch (71 videos, batch=32)
- Total time: ~1-2 hours
- Target accuracy: 85-95% (depends on data quality)

**If training fails:**
1. Check feature/label file names match
2. Verify labels are in correct format
3. Try lower learning rate: `--lr 5e-5`
4. Reduce model size: `--hidden_dim 128 --num_layers 1`

### Phase 4: Inference

#### 4.1 Single Video Inference

```bash
python inference.py \
    --video ../data/videos/KHC2_M1_D10.mp4 \
    --model ../checkpoints/best_model.pt \
    --yolo_path /path/to/best.pt \
    --output predictions.csv \
    --config ../checkpoints/config.json \
    --device cuda
```

**Process:**
1. Extracts YOLO features from video (same as training)
2. Loads trained model
3. Generates per-second predictions
4. Saves to CSV with confidence scores

**Output format:**
```csv
# Video: KHC2_M1_D10.mp4
# Total seconds: 180
# Mobile (0): 134
# Immobile (1): 46
second,prediction,confidence
0,0,0.953
1,0,0.876
2,1,0.921
...
```

#### 4.2 Batch Inference

Process multiple videos at once:

```bash
python batch_inference.py \
    --video_dir ../data/videos \
    --model ../checkpoints/best_model.pt \
    --yolo_path /path/to/best.pt \
    --output_dir ../data/predictions \
    --config ../checkpoints/config.json \
    --skip_existing
```

### Phase 5: Debugging

#### 5.1 VS Code Debugging

To debug training:

1. Set breakpoints in code (e.g., in `train.py` or `model.py`)

2. Add `--debug` flag:
   ```bash
   python train.py --debug [other args...]
   ```

3. In VS Code:
   - Press F5 or Run → Start Debugging
   - Select "Python: Attach"
   - Connect to localhost:5678

4. Program will pause at breakpoints

#### 5.2 Common Issues

**Problem: "No matching feature/label pairs found"**
- Solution: Check file naming
  - Features: `KHC2_M1_D10_features.pt`
  - Labels: `D10_M1.labels.csv`
- The matcher looks for M/D tokens and phase (pre/post)

**Problem: "CUDA out of memory"**
- Solution 1: Reduce batch size: `--batch_size 16` or `8`
- Solution 2: Reduce model size: `--hidden_dim 128`
- Solution 3: Reduce window size: `--window_seconds 1.0`

**Problem: "Poor accuracy"**
- Solution 1: Try different YOLO layers
- Solution 2: Increase window size: `--window_seconds 3.0`
- Solution 3: Enable class weights: `--use_class_weights`
- Solution 4: Check label alignment with videos

**Problem: "Model not learning"**
- Solution 1: Lower LR: `--lr 5e-5` or `1e-5`
- Solution 2: Check loss curve in TensorBoard
- Solution 3: Verify labels are correct

## Hyperparameter Guidelines

### Model Architecture

| Parameter | Recommended | Range | Effect |
|-----------|-------------|-------|---------|
| `hidden_dim` | 256 | 128-512 | Larger = more capacity, more memory |
| `num_layers` | 2 | 1-3 | More layers = more temporal modeling |
| `num_heads` | 4 | 2-8 | More heads = more attention patterns |
| `window_seconds` | 2.0 | 1.0-5.0 | Larger = more context, more computation |

### Training

| Parameter | Recommended | Range | Effect |
|-----------|-------------|-------|---------|
| `lr` | 1e-4 | 1e-5 to 1e-3 | Lower = slower but more stable |
| `batch_size` | 32 | 8-64 | Larger = faster, needs more memory |
| `dropout` | 0.1 | 0.0-0.3 | Higher = more regularization |

### Best Practices

1. **Start small, scale up:**
   - Begin with: `hidden_dim=128, num_layers=1, batch_size=16`
   - If working well, increase model capacity

2. **Use class weights:**
   - Always enable `--use_class_weights` for imbalanced data

3. **Monitor validation:**
   - Focus on validation accuracy, not training
   - Stop if validation stops improving

4. **Test incrementally:**
   - Extract features once, experiment with model architectures
   - Don't re-extract features unless changing YOLO layer

## Performance Benchmarks

### Hardware: 8GB VRAM GPU

| Task | Time | Memory |
|------|------|--------|
| Feature extraction (1 video) | ~5-10s | 3-4 GB |
| Training (1 epoch, 71 videos) | ~2-3 min | 4-6 GB |
| Inference (1 video) | ~5-10s | 2-3 GB |

### Expected Accuracy

- **Good performance**: 85-92% overall accuracy
- **Excellent performance**: 93-97% overall accuracy
- **Per-class**: Should be balanced (both >80%)

If significantly worse:
1. Check label quality
2. Try different YOLO layers
3. Increase temporal window
4. Add more training data

## Next Steps

After successful training:

1. **Evaluate on test set:**
   - Hold out some videos not used in training
   - Run inference and compare to ground truth

2. **Optimize for deployment:**
   - Export to ONNX for faster inference
   - Quantize model if needed

3. **Improve model:**
   - Collect more training data
   - Try data augmentation
   - Ensemble multiple models

## Support

For issues:
1. Check this guide
2. Check README.md
3. Inspect TensorBoard logs
4. Use VS Code debugging

---

**Remember:** The key to success is starting simple and iterating!
