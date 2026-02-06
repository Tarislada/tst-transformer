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
┌──────────────────────────┐
│ 1. Label Conversion      │
│    Excel → CSV           │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ 2. Feature Extraction    │
│    Video → YOLO → .pt    │
│    (Offline, cached)     │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ 3. Model Training        │
│    Features + Labels     │
│    → Trained Model       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ 4. Inference             │
│    Video → Frame Preds   │
│    → Aggregate → Seconds │
└──────────────────────────┘
```

## Key Concept: Frame-Level Prediction

**Important**: The model predicts **per-frame**, not per-second directly. This is more accurate because:

1. **Model sees individual frames**: Predicts mobility for each frame in sliding windows
2. **Overlapping predictions averaged**: Multiple windows may cover the same frame
3. **Aggregation to seconds**: Uses configurable methods (majority vote, any, all, etc.)

This approach is **more flexible** than direct per-second prediction and allows experimenting with different aggregation strategies.

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
│       ├── frames/               # Frame-level predictions (optional)
│       └── seconds/              # Second-level predictions (default)
├── models/
│   └── best.pt                   # Your trained YOLO model
├── checkpoints/                  # [Generated] Trained models
│   ├── best_model.pt
│   └── config.json
└── tst_transformer/              # Code directory
    ├── model.py
    ├── dataset.py
    ├── preprocess.py
    ├── train.py
    ├── inference.py
    ├── postprocess.py
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
    --yolo_path ../models/best.pt \
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
    --hidden_dim 512 \
    --num_layers 4 \
    --num_heads 4 \
    --window_size 120 \
    --stride 60 \
    --use_class_weights \
    --patience 10 \
    --device cuda
```

**Important parameters:**
- `--window_size 120`: Window size in **frames** (e.g., 120 frames = 2 seconds at 60fps)
- `--stride 60`: Stride in frames during training (e.g., 60 = 50% overlap)
- **Note**: Validation uses NO stride (full coverage without overlap)

**What happens during training:**

1. **Data Loading:**
   - Finds matching feature/label pairs
   - Splits into train/val (80/20 by default)
   - Creates temporal windows with stride during training
   - **No stride** during validation (clean non-overlapping windows)

2. **Model Creation:**
   - Transformer with specified architecture
   - ~500K-2M trainable parameters (depending on hidden_dim)
   - YOLO features are frozen (pre-extracted)

3. **Training Loop:**
   - Each epoch processes all training samples
   - Validates on held-out videos
   - **Per-frame predictions** in both train and validation
   - Model outputs `[B, T, 1]` binary logits per frame
   - Loss calculated on flattened `[B*T]` predictions
   - Saves checkpoint if validation improves
   - Early stopping if no improvement for 10 epochs

4. **Outputs:**
   - `checkpoints/best_model.pt`: Best model by validation F1 score
   - `checkpoints/latest_model.pt`: Most recent checkpoint
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
- **Val/Mobile_Acc** and **Val/Immobile_Acc**: Check class-specific performance (both should be >80%)
- **Val/F1_Macro**: Overall F1 score (balanced metric)
- **Val/F1_Immobile**: F1 for immobile class (this is what model optimizes)
- **LR**: Learning rate (decreases with cosine annealing)

**Expected results:**
- Training time: ~2-3 minutes per epoch (71 videos, batch=32)
- Total time: ~1-2 hours (depending on early stopping)
- Target metrics:
  - Overall accuracy: 85-95%
  - Per-class accuracy: Both >80%
  - F1 scores: >0.80

**If training fails:**
1. Check feature/label file names match
2. Verify labels are in correct format (0/1 per second)
3. Try lower learning rate: `--lr 5e-5`
4. Reduce model size: `--hidden_dim 256 --num_layers 2`
5. Check class balance: enable `--use_class_weights`

### Phase 4: Inference

#### 4.1 Understanding the Inference Pipeline

The new inference pipeline works as follows:

```
1. Video → YOLO Features [T, feature_dim]
                ↓
2. Sliding Windows (size=120, NO stride)
                ↓
3. Model predicts per-frame [T, 1] logits
                ↓
4. Sigmoid → probabilities [T]
                ↓
5. Threshold (>0.5) → binary predictions [T]
                ↓
6. Aggregate frames to seconds (majority/any/all)
                ↓
7. Output: Second-level predictions + confidences
```

**Key differences from training:**
- ✅ **No stride**: Windows don't overlap during inference
- ✅ **Frame-level predictions**: Full temporal resolution
- ✅ **Flexible aggregation**: Post-hoc method selection
- ✅ **Both outputs**: Frame + second level for debugging

#### 4.2 Single Video Inference

**Using config file (recommended):**
```bash
python inference.py \
    --video ../data/videos/KHC2_M1_D10.mp4 \
    --model ../checkpoints/best_model.pt \
    --yolo_path ../models/best.pt \
    --config ../checkpoints/config.json \
    --output predictions.csv \
    --save_frame_predictions \
    --aggregate_method majority \
    --device cuda
```

**Without config (specify all parameters):**
```bash
python inference.py \
    --video ../data/videos/KHC2_M1_D10.mp4 \
    --model ../checkpoints/best_model.pt \
    --yolo_path ../models/best.pt \
    --window_size 120 \
    --fps 30 \
    --hidden_dim 512 \
    --num_layers 4 \
    --num_heads 4 \
    --layer_name model.18 \
    --imgsz 1088 \
    --aggregate_method majority \
    --save_frame_predictions
```

**Using window in seconds (alternative):**
```bash
python inference.py \
    --video ../data/videos/KHC2_M1_D10.mp4 \
    --model ../checkpoints/best_model.pt \
    --yolo_path ../models/best.pt \
    --window_seconds 2.0 \
    --fps 30 \
    --config ../checkpoints/config.json \
    --aggregate_method majority
```

**Process:**
1. **Feature Extraction**: Extracts YOLO features from video (same as training)
2. **Model Loading**: Loads trained model with correct architecture
3. **Frame Prediction**: Generates per-frame predictions using sliding windows
4. **Aggregation**: Aggregates frame predictions to seconds
5. **Saving**: Saves both frame and second-level outputs

**Output files:**
- `predictions.csv`: Second-level predictions (always created)
- `predictions_frames.csv`: Frame-level predictions (if `--save_frame_predictions`)

#### 4.3 Aggregation Methods

Choose how frame predictions aggregate to seconds:

| Method | Formula | Use Case | Example |
|--------|---------|----------|---------|
| `majority` | `1 if mean(frames) > 0.5` | **Default**, balanced approach | 31/60 immobile frames → immobile |
| `any` | `1 if any(frames) == 1` | Conservative: catch all immobility | 1/60 immobile frames → immobile |
| `all` | `1 if all(frames) == 1` | Strict: only clear immobility | 60/60 immobile frames → immobile |
| `mean_threshold` | `1 if mean(frames) > 0.7` | Adjustable threshold (70%) | 43/60 immobile frames → immobile |

**Choosing the right method:**

1. **Start with `majority`** (default) - most balanced
2. **Use `any`** if you want to catch brief immobility events
3. **Use `all`** if you only care about sustained immobility
4. **Use `mean_threshold`** if you want fine-grained control

**Experiment with methods:**
```bash
# Try different aggregation methods on same video
for method in majority any all mean_threshold; do
    python inference.py \
        --video test.mp4 \
        --model best_model.pt \
        --yolo_path yolo.pt \
        --config config.json \
        --aggregate_method $method \
        --output predictions_$method.csv \
        --save_frame_predictions
done
```

Then compare which method best matches your ground truth labels.

#### 4.4 Output Format

**Second-level predictions** (`predictions.csv`):
```csv
# Video: KHC2_M1_D10.mp4
# Total seconds: 180
# Mobile (0): 134
# Immobile (1): 46
# prediction: 0=mobile, 1=immobile
# confidence: probability of predicted class
second,prediction,confidence
0,0,0.953
1,0,0.876
2,1,0.921
...
```

**Frame-level predictions** (`predictions_frames.csv`):
```csv
# Video: KHC2_M1_D10.mp4
# Total frames: 5400
# Mobile (0): 4020
# Immobile (1): 1380
# prediction: 0=mobile, 1=immobile
# confidence: probability of predicted class
frame,prediction,confidence
0,0,0.95
1,0,0.94
2,1,0.89
...
```

**Confidence interpretation:**
- If prediction=0 (mobile): confidence = P(mobile) = 1 - P(immobile)
- If prediction=1 (immobile): confidence = P(immobile)
- Higher confidence = more certain prediction

#### 4.5 Batch Inference

Process multiple videos:

```bash
# Create batch script
for video in ../data/videos/*.mp4; do
    basename=$(basename "$video" .mp4)
    python inference.py \
        --video "$video" \
        --model ../checkpoints/best_model.pt \
        --yolo_path ../models/best.pt \
        --config ../checkpoints/config.json \
        --output "../data/predictions/${basename}_predictions.csv" \
        --aggregate_method majority \
        --save_frame_predictions
done
```

Or create a dedicated batch script:

```python
# batch_inference.py
import subprocess
from pathlib import Path

video_dir = Path("../data/videos")
output_dir = Path("../data/predictions")
output_dir.mkdir(exist_ok=True)

for video in video_dir.glob("*.mp4"):
    cmd = [
        "python", "inference.py",
        "--video", str(video),
        "--model", "../checkpoints/best_model.pt",
        "--yolo_path", "../models/best.pt",
        "--config", "../checkpoints/config.json",
        "--output", str(output_dir / f"{video.stem}_predictions.csv"),
        "--aggregate_method", "majority",
        "--save_frame_predictions"
    ]
    subprocess.run(cmd)
```

### Phase 5: Evaluation & Debugging

#### 5.1 Comparing Predictions to Ground Truth

If you have ground truth labels:

```python
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions and ground truth
preds = pd.read_csv('predictions.csv', comment='#')
truth = pd.read_csv('ground_truth.labels.csv')

# Align by second
merged = preds.merge(truth, left_on='second', right_on='second')

# Calculate metrics
acc = accuracy_score(merged['label'], merged['prediction'])
f1 = f1_score(merged['label'], merged['prediction'])
f1_mobile = f1_score(merged['label'], merged['prediction'], pos_label=0)
f1_immobile = f1_score(merged['label'], merged['prediction'], pos_label=1)

print(f"Accuracy: {acc:.2%}")
print(f"F1 Score (macro): {f1:.3f}")
print(f"F1 Mobile: {f1_mobile:.3f}")
print(f"F1 Immobile: {f1_immobile:.3f}")

print("\nClassification Report:")
print(classification_report(merged['label'], merged['prediction'], 
                            target_names=['Mobile', 'Immobile']))

# Confusion matrix
cm = confusion_matrix(merged['label'], merged['prediction'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Mobile', 'Immobile'],
            yticklabels=['Mobile', 'Immobile'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

#### 5.2 Visualizing Frame vs Second Predictions

```python
import matplotlib.pyplot as plt
import pandas as pd

# Load both levels
frames = pd.read_csv('predictions_frames.csv', comment='#')
seconds = pd.read_csv('predictions.csv', comment='#')

# Convert frame numbers to seconds
frames['second'] = frames['frame'] / 30  # Assuming 30 fps

fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

# Frame-level predictions
axes[0].plot(frames['second'], frames['prediction'], alpha=0.3, label='Frame predictions')
axes[0].set_ylabel('Prediction')
axes[0].set_title('Frame-Level Predictions')
axes[0].set_ylim([-0.1, 1.1])
axes[0].legend()

# Frame-level confidence
axes[1].plot(frames['second'], frames['confidence'], alpha=0.5, color='orange')
axes[1].set_ylabel('Confidence')
axes[1].set_title('Frame-Level Confidence')
axes[1].set_ylim([0, 1])

# Second-level predictions
axes[2].step(seconds['second'], seconds['prediction'], where='mid', linewidth=2)
axes[2].fill_between(seconds['second'], 0, seconds['prediction'], alpha=0.3, step='mid')
axes[2].set_ylabel('Prediction')
axes[2].set_xlabel('Time (seconds)')
axes[2].set_title('Second-Level Predictions (aggregated)')
axes[2].set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=150)
```

#### 5.3 Comparing Different Aggregation Methods

```python
import pandas as pd
import matplotlib.pyplot as plt

methods = ['majority', 'any', 'all', 'mean_threshold']
ground_truth = pd.read_csv('ground_truth.labels.csv')

fig, axes = plt.subplots(len(methods) + 1, 1, figsize=(15, 10), sharex=True)

# Ground truth
axes[0].step(ground_truth['second'], ground_truth['label'], where='mid', label='Ground Truth')
axes[0].set_ylabel('GT')
axes[0].set_title('Ground Truth')
axes[0].legend()

# Each aggregation method
for i, method in enumerate(methods):
    preds = pd.read_csv(f'predictions_{method}.csv', comment='#')
    acc = accuracy_score(ground_truth['label'][:len(preds)], preds['prediction'])
    
    axes[i+1].step(preds['second'], preds['prediction'], where='mid', label=f'{method} (acc={acc:.2%})')
    axes[i+1].set_ylabel(method)
    axes[i+1].legend()

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('aggregation_comparison.png', dpi=150)
```

#### 5.4 VS Code Debugging

To debug any script:

1. Add `--debug` flag:
   ```bash
   python train.py --debug [other args...]
   # or
   python inference.py --debug [other args...]
   ```

2. Set breakpoints in VS Code (click left of line numbers)

3. In VS Code:
   - Press **F5** or Run → Start Debugging
   - Select **"Python: Attach"**
   - Connect to **localhost:5678**

4. Program will pause at breakpoints - inspect variables, step through code

**Useful debug points:**
- `train.py`: Inside training loop to inspect batch data
- `inference.py`: Inside prediction loop to check frame predictions
- `model.py`: In forward pass to see intermediate outputs
- `dataset.py`: In `__getitem__` to verify data loading

#### 5.5 Common Issues & Solutions

**Problem: "Different window_size between training and inference"**
```
Error: Model expects 120 frames but got 60
```
- **Solution**: Use `--config` flag to auto-load training parameters
- Or manually match: `--window_size 120` (same as training)

**Problem: "Frame predictions look noisy/jumpy"**
```
Frame predictions alternate rapidly: 0,1,0,1,0,1...
```
- **Solution 1**: Try `any` or `mean_threshold` aggregation (more tolerant)
- **Solution 2**: Increase training `window_size` for more temporal context
- **Solution 3**: Check if model is well-trained (validation accuracy >85%)

**Problem: "Second predictions don't match ground truth"**
```
Predictions offset by 1-2 seconds or systematically wrong
```
- **Solution 1**: Try different `aggregate_method` (majority/any/all)
- **Solution 2**: Check if ground truth aligns with video (no offset?)
- **Solution 3**: Visualize frame predictions to see if model is learning patterns
- **Solution 4**: Verify FPS is correct (`--fps 30` or `60`)

**Problem: "CUDA out of memory during inference"**
```
RuntimeError: CUDA out of memory
```
- **Solution 1**: Reduce `--batch_size` during feature extraction (default 8 → 4)
- **Solution 2**: Use CPU for inference: `--device cpu` (slower but works)
- **Solution 3**: Close other GPU applications

**Problem: "Inference too slow"**
```
Taking minutes per video
```
- **Solution**: Features are cached after first extraction
- Only first run extracts features (~5-10s per video)
- Subsequent runs with same video are <1s
- If re-extracting: check `--skip_existing` flag

**Problem: "Model predicts all mobile or all immobile"**
```
All predictions are 0 or all are 1
```
- **Solution 1**: Check training converged (look at TensorBoard)
- **Solution 2**: Try lower `--decision_threshold` (e.g., 0.3 for more immobile)
- **Solution 3**: Retrain with `--use_class_weights`
- **Solution 4**: Verify labels are correct

**Problem: "High training accuracy but poor validation"**
```
Train: 95%, Val: 65%
```
- **Solution 1**: Overfitting - increase `--dropout` (0.1 → 0.3)
- **Solution 2**: Reduce model capacity: `--hidden_dim 256` (from 512)
- **Solution 3**: Add more training data or validation videos
- **Solution 4**: Check validation set is representative

## Hyperparameter Guidelines

### Model Architecture

| Parameter | Recommended | Range | Effect | When to Change |
|-----------|-------------|-------|---------|----------------|
| `hidden_dim` | 512 | 128-1024 | Model capacity | Increase if underfitting, decrease if overfitting/OOM |
| `num_layers` | 4 | 1-6 | Temporal depth | More layers = more context but slower |
| `num_heads` | 4 | 2-8 | Attention patterns | Usually keep at 4 |
| `window_size` | 120 | 60-240 | Temporal context | Larger = more context but more memory |
| `stride` | 60 | 30-120 | Training overlap | 50% of window_size is good |
| `dropout` | 0.1 | 0.0-0.3 | Regularization | Increase if overfitting |

### Training

| Parameter | Recommended | Range | Effect | When to Change |
|-----------|-------------|-------|---------|----------------|
| `lr` | 1e-4 | 1e-5 to 1e-3 | Learning speed | Lower if loss unstable, higher if slow |
| `batch_size` | 32 | 8-64 | Training speed | Decrease if OOM, increase if under-utilizing GPU |
| `epochs` | 50 | 20-100 | Training time | Let early stopping decide |
| `patience` | 10 | 5-20 | Early stopping | Increase for more training time |
| `use_class_weights` | True | - | Class balance | Always enable for imbalanced data |

### Inference

| Parameter | Recommended | Options | Effect | When to Change |
|-----------|-------------|---------|---------|----------------|
| `aggregate_method` | `majority` | see table | Frame→second | Experiment to match ground truth |
| `decision_threshold` | 0.5 | 0.1-0.9 | Prediction threshold | Lower = more sensitive to immobility |
| `save_frame_predictions` | True | True/False | Debug output | Disable for production to save disk |

### Best Practices

1. **Start small, scale up:**
   - Begin: `hidden_dim=256, num_layers=2, batch_size=16`
   - If working well: increase to `hidden_dim=512, num_layers=4`
   - Monitor GPU memory and training time

2. **Use class weights:**
   - Always enable `--use_class_weights` for imbalanced data
   - Helps model learn minority class

3. **Monitor validation, not training:**
   - Focus on validation metrics (accuracy, F1)
   - Training metrics can be misleading (overfitting)
   - Use early stopping to prevent overtraining

4. **Extract features once:**
   - Feature extraction is slowest step (~10-15 min)
   - Experiment with model architectures without re-extracting
   - Only re-extract if changing `--layer_name` or `--imgsz`

5. **Experiment with aggregation:**
   - Try all methods on held-out videos
   - Choose method that best matches your definition of "immobile"
   - `majority` is usually best for balanced datasets

6. **Use config files:**
   - Training saves `config.json` automatically
   - Use `--config` during inference to avoid mismatches
   - Prevents hard-to-debug parameter errors

## Performance Benchmarks

### Hardware: 8GB VRAM GPU (RTX 3070 / similar)

| Task | Time | Memory | Notes |
|------|------|--------|-------|
| Feature extraction (1 video, 3 min) | ~5-10s | 3-4 GB | Cached after first run |
| Training (1 epoch, 71 videos) | ~2-3 min | 4-6 GB | Depends on batch_size |
| Training (full, 50 epochs) | ~1-2 hours | 4-6 GB | With early stopping |
| Inference (1 video, first run) | ~5-10s | 2-3 GB | Includes feature extraction |
| Inference (1 video, cached) | <1s | 1 GB | Features already extracted |

### Expected Accuracy

| Metric | Good | Excellent | Notes |
|--------|------|-----------|-------|
| Overall accuracy | 85-92% | 93-97% | Should be balanced |
| Mobile accuracy | >80% | >90% | Class 0 |
| Immobile accuracy | >80% | >90% | Class 1 |
| F1 score (macro) | >0.80 | >0.90 | Balanced metric |

**If significantly worse:**
1. Check label quality and alignment with videos
2. Try different YOLO layer (`--layer_name`)
3. Increase temporal window (`--window_size`)
4. Experiment with aggregation methods
5. Collect more or better training data
6. Verify video quality (resolution, lighting, etc.)

## Advanced Topics

### 5.6 Custom Aggregation Methods

Add your own method to `postprocess.py`:

```python
# filepath: postprocess.py

def aggregate_frames_to_seconds(
    frame_predictions: np.ndarray,
    fps: int = 60,
    method: str = 'majority'
) -> np.ndarray:
    # ...existing code...
    
    elif method == 'weighted':
        # Your custom logic: weight frames by confidence
        # Requires passing frame confidences as well
        weighted_sum = (frames_in_second * confidences).sum()
        pred = 1 if weighted_sum > threshold else 0
    
    elif method == 'temporal_smoothing':
        # Smooth with temporal filter before aggregating
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(frames_in_second, size=5)
        pred = 1 if smoothed.mean() > 0.5 else 0
    
    # ...existing code...
```

Then use: `--aggregate_method weighted`

### 5.7 Adjusting Decision Threshold

The model outputs probabilities, which are converted to binary predictions using a threshold (default 0.5):

```python
# In inference.py
predictions = (probs > 0.5).astype(int)
```

To find optimal threshold:

1. Run inference with `--save_frame_predictions`
2. Load frame probabilities
3. Try different thresholds:

```python
import numpy as np
from sklearn.metrics import f1_score

# Load frame predictions
frame_probs = ...  # Load sigmoid probabilities
ground_truth = ...

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (frame_probs > threshold).astype(int)
    f1 = f1_score(ground_truth, preds)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
```

Then use in inference: `--decision_threshold 0.35`

### 5.8 Export to ONNX

For faster deployment or non-Python environments:

```python
import torch
from model import TSTModel

# Load model
model = TSTModel(
    yolo_feature_dim=512,
    hidden_dim=512,
    num_heads=4,
    num_layers=4,
    max_seq_length=120
)
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Export
dummy_input = torch.randn(1, 120, 512)
torch.onnx.export(
    model, 
    dummy_input, 
    'model.onnx',
    input_names=['features'],
    output_names=['logits'],
    dynamic_axes={'features': {0: 'batch', 1: 'sequence'}}
)
```

Then use with ONNX Runtime (faster inference):
```python
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
logits = session.run(['logits'], {'features': features_np})[0]
```

### 5.9 Ensemble Models

Train multiple models and average predictions for better robustness:

```bash
# Train with different seeds
for seed in 42 43 44 45 46; do
    python train.py \
        --feature_dir ../data/features \
        --label_dir ../data/labels_csv \
        --output_dir ../checkpoints/seed_$seed \
        --seed $seed \
        [other args...]
done
```

Then average predictions:

```python
import torch
import numpy as np

models = []
for seed in [42, 43, 44, 45, 46]:
    model = TSTModel(...)
    checkpoint = torch.load(f'checkpoints/seed_{seed}/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models.append(model)

# Ensemble prediction
def ensemble_predict(models, features):
    probs = []
    for model in models:
        with torch.no_grad():
            logits = model(features)
            prob = torch.sigmoid(logits)
            probs.append(prob)
    
    avg_prob = torch.stack(probs).mean(dim=0)
    return (avg_prob > 0.5).long()
```

## Next Steps

After successful training and inference:

1. **Evaluate on held-out test set:**
   - Videos not used in training or validation
   - Calculate metrics: accuracy, F1, confusion matrix
   - Compare different aggregation methods

2. **Optimize aggregation:**
   - Try all methods (majority, any, all, mean_threshold)
   - Compare to ground truth
   - Choose best for your specific use case

3. **Fine-tune threshold:**
   - Experiment with `--decision_threshold`
   - Lower threshold = more sensitive to immobility
   - Higher threshold = more conservative

4. **Deploy model:**
   - Export to ONNX for production
   - Create REST API for batch processing
   - Integrate into analysis pipeline

5. **Improve model:**
   - Collect more training data (especially edge cases)
   - Try data augmentation (temporal jittering)
   - Experiment with architecture changes
   - Ensemble multiple models

## Troubleshooting Checklist

Before opening an issue, verify:

- [ ] File naming matches conventions (`*_features.pt`, `*.labels.csv`)
- [ ] Labels are in correct format (0=mobile, 1=immobile)
- [ ] Features extracted with same YOLO layer as training
- [ ] Training parameters match inference (use `--config`)
- [ ] FPS is correct (`--fps 30` or `60`)
- [ ] Sufficient GPU memory (8GB+ recommended)
- [ ] PyTorch and CUDA versions compatible
- [ ] Video resolution matches YOLO training (`--imgsz`)
- [ ] Ground truth labels align with video timing
- [ ] Model has converged during training (check TensorBoard)

## FAQ

**Q: Should I use stride during inference?**  
A: No! Inference uses non-overlapping windows (`stride = window_size`) for clean predictions. Overlapping windows during inference can cause inconsistent frame predictions.

**Q: What's the difference between window_size and window_seconds?**  
A: `--window_size` is in frames (e.g., 120 frames), `--window_seconds` is in seconds (e.g., 2.0s). They're converted using FPS: `window_size = window_seconds * fps`.

**Q: Why does the model predict per-frame instead of per-second?**  
A: Frame-level prediction is more accurate and flexible. You can aggregate frames to seconds using different methods (majority, any, all) without retraining.

**Q: How do I choose the aggregation method?**  
A: Start with `majority` (default). If you want to catch brief immobility, use `any`. If you only care about sustained immobility, use `all`. Experiment on validation videos.

**Q: Can I use this on different FPS videos?**  
A: Yes, but specify `--fps` during both training and inference. Window size in frames remains constant, but represents different durations.

**Q: How much data do I need?**  
A: Minimum ~20-30 videos, recommended 50+. More diverse data (different mice, lighting, angles) improves generalization.

**Q: Can I transfer models between different YOLO checkpoints?**  
A: Only if feature dimensions match. Different YOLO architectures or layers have different dimensions. Safest to retrain if changing YOLO model.

---

**Remember:** The key to success is starting simple and iterating!

1. Start with default parameters
2. Train until convergence
3. Evaluate on validation set
4. Experiment with aggregation methods
5. Fine-tune based on results
6. Deploy and collect feedback
7. Iterate!
