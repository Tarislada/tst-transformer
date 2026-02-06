# TST Transformer - Tail Suspension Test Classifier

Deep learning pipeline for automated classification of mouse mobility in Tail Suspension Test (TST) videos using YOLO pose features and Transformer models.

## Overview

This project provides an end-to-end solution for:
1. **Feature Extraction**: Extract pose features from videos using YOLO segmentation
2. **Training**: Train a Transformer classifier on temporal sequences
3. **Inference**: Generate per-second mobility predictions for new videos

### Key Features

- **Frame-level predictions** aggregated to seconds using flexible methods
- **Temporal modeling** with Transformer architecture
- **Pre-extracted features** for fast experimentation
- **Class balancing** for imbalanced datasets
- **TensorBoard logging** for training monitoring
- **VS Code debugging** support

## Architecture

```
Raw Video (.mp4)
    ↓
YOLO Feature Extractor (frozen)
    ↓
Frame Features [T, feature_dim]
    ↓
Transformer Model
    ↓
Per-Frame Predictions [T, 1]
    ↓
Aggregation (majority/any/all)
    ↓
Per-Second Predictions
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Ultralytics YOLO model
- CUDA-capable GPU (8GB+ VRAM recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/Tarislada/tst-transformer.git
cd tst-transformer

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Dependencies

Core:
- `torch>=2.0.0`
- `ultralytics>=8.0.0` (YOLO)
- `opencv-python>=4.8.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `tqdm>=4.65.0`

Optional:
- `tensorboard>=2.13.0` (training visualization)
- `debugpy>=1.6.0` (VS Code debugging)

## Quick Start

### 1. Prepare Data

Organize your data:
```
data/
├── videos/              # Raw .mp4 videos
├── labels_excel/        # Excel files with labels
├── labels_csv/          # [Generated] Converted CSV labels
├── features/            # [Generated] Extracted YOLO features
└── predictions/         # [Generated] Inference outputs
```

Convert labels from Excel:
```bash
python tst_transformer/convert_labels_from_excel.py convert-batch \
    --excel-glob "data/labels_excel/*.xlsx" \
    --out-dir "data/labels_csv"
```

### 2. Extract Features

```bash
cd tst_transformer
python preprocess.py inspect /path/to/best.pt
```

This will show available layers. Common choices:
- `model.15`: Early neck (high resolution)
- `model.18`: Mid neck (balanced) ← **recommended**
- `model.21`: Late neck (more semantic)

Then extract features from all videos:

```bash
python preprocess.py \
    --video_dir ../data/videos \
    --output_dir ../data/features \
    --yolo_path /path/to/yolomodel.pt \
    --layer_name model.18 \
    --imgsz 1080 \
    --batch_size 8 \
    --device cuda
```

**Time**: ~5-10 seconds per 3-minute video

### 3. Train Model

```bash
python tst_transformer/train.py \
    --feature_dir data/features \
    --label_dir data/labels_csv \
    --output_dir checkpoints \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --window_size 120 \
    --fps 60 \
    --hidden_dim 512 \
    --num_layers 4 \
    --num_heads 4 \
    --use_class_weights \
    --patience 10 \
    --device cuda
```

**Time**: ~2-3 minutes per epoch

Monitor training:
```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

### 4. Run Inference

**Single video:**
```bash
python tst_transformer/inference.py \
    --video data/videos/test_video.mp4 \
    --model checkpoints/best_model.pt \
    --yolo_path path/to/yolo11n-pose.pt \
    --output predictions.csv \
    --save_frame_predictions \
    --config checkpoints/config.json
```

Or batch process multiple videos:

```bash
python batch_inference.py \
    --video_dir ../data/videos \
    --model ../checkpoints/best_model.pt \
    --yolo_path /path/to/yolomodel.pt \
    --output_dir ../predictions \
    --config ../checkpoints/config.json
```

## Debugging with VS Code

To debug training with VS Code:

1. Add breakpoints in `train.py` or `model.py`
2. Run with `--debug` flag:
   ```bash
   python train.py --debug --feature_dir ... [other args]
   ```
3. In VS Code, attach to the debugger (F5 or Run → Attach to Process → localhost:5678)

VS Code `launch.json` example:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            }
        }
    ]
}
```

## Hyperparameter Tuning

Key hyperparameters to tune:

### Model Architecture
- `--hidden_dim`: Transformer hidden size (128, 256, 512)
- `--num_layers`: Number of transformer layers (1, 2, 3)
- `--num_heads`: Attention heads (4, 8)
- `--window_seconds`: Temporal context window (1.0, 2.0, 3.0)

### Training
- `--lr`: Learning rate (1e-5 to 1e-3)
- `--batch_size`: Batch size (16, 32, 64)
- `--use_class_weights`: Enable for imbalanced datasets

### Data
- `--train_split`: Train/val split (0.7, 0.8, 0.9)
- Try different YOLO `--layer_name` choices

## Output Format

Prediction CSV format:
```csv
# Video: test_video.mp4
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

**Frame-level predictions** (`predictions_frames.csv`, optional):
```csv
# Video: test_video.mp4
# Total frames: 10800
# Mobile (0): 8040
# Immobile (1): 2760
frame,prediction,confidence
0,0,0.95
1,0,0.94
2,1,0.89
...
```

## Model Architecture

### Transformer Classifier

```python
TSTModel(
    yolo_feature_dim=512,    # From YOLO neck layer
    hidden_dim=512,          # Transformer hidden size
    num_heads=4,             # Multi-head attention
    num_layers=4,            # Transformer encoder layers
    num_classes=2,           # Binary: mobile/immobile
    dropout=0.1,
    max_seq_length=120       # Window size in frames
)
```

**Input**: `[batch, T, yolo_feature_dim]`  
**Output**: `[batch, T, 1]` binary logits per frame

### Training Details

- **Loss**: Focal Loss (handles class imbalance)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience of 10 epochs
- **Validation**: 20% of videos held out

## Aggregation Methods

The model predicts per-frame, then aggregates to seconds:

| Method | Description | Use Case |
|--------|-------------|----------|
| `majority` | Label=1 if >50% frames immobile | **Default**, balanced |
| `any` | Label=1 if ANY frame immobile | Conservative (catch all immobility) |
| `all` | Label=1 if ALL frames immobile | Strict (only clear immobility) |
| `mean_threshold` | Label=1 if >70% frames immobile | Adjustable threshold |

Specify with: `--aggregate_method majority`

## Hyperparameters

### Recommended Starting Point

```bash
--window_size 120        # 2 seconds at 60fps
--hidden_dim 512         # Model capacity
--num_layers 4           # Temporal depth
--num_heads 4            # Attention heads
--batch_size 32          # GPU memory permitting
--lr 1e-4               # Learning rate
```

### Tuning Guidelines

**If model underfits (low accuracy):**
- Increase `hidden_dim` (256 → 512)
- Increase `num_layers` (2 → 4)
- Increase `window_size` (60 → 120)

**If model overfits (train >> val accuracy):**
- Increase `dropout` (0.1 → 0.2)
- Reduce `hidden_dim` (512 → 256)
- Add more training data

**If OOM errors:**
- Reduce `batch_size` (32 → 16 → 8)
- Reduce `hidden_dim` (512 → 256)
- Reduce `window_size` (120 → 60)

## Debugging

### VS Code Integration

Add `--debug` flag to any script:
```bash
python train.py --debug [other args...]
```

Then in VS Code:
1. Set breakpoints
2. Press F5 → "Python: Attach"
3. Connect to localhost:5678

### Common Issues

**"No matching feature/label pairs found"**
- Check file naming conventions
- Features: `*_features.pt`
- Labels: `*.labels.csv`
- Must share mouse/day identifiers

**"CUDA out of memory"**
- Reduce `--batch_size`
- Reduce `--hidden_dim`
- Close other GPU applications

**"Poor validation accuracy"**
- Check label quality
- Try different `--layer_name` (model.15, model.18, model.21)
- Enable `--use_class_weights`
- Increase `--window_size`

## Project Structure

```
tst_transformer/
├── model.py                    # Transformer model + YOLO extractor
├── dataset.py                  # Data loading and windowing
├── train.py                    # Training script
├── inference.py                # Inference script (NEW: frame→second)
├── predict.py                  # Batch prediction
├── preprocess.py              # Feature extraction
├── postprocess.py             # Aggregation methods
├── convert_labels_from_excel.py  # Label conversion
└── WORKFLOW.md                # Detailed guide
```

## Performance

### Hardware: 8GB VRAM GPU

| Task | Time | Memory |
|------|------|--------|
| Feature extraction (1 video) | ~5-10s | 3-4 GB |
| Training (1 epoch, 71 videos) | ~2-3 min | 4-6 GB |
| Inference (1 video) | ~5-10s | 2-3 GB |

### Expected Accuracy

- **Good**: 85-92% overall
- **Excellent**: 93-97% overall
- Both classes should be >80%

## Citation

If you use this code, please cite:

```bibtex
@software{tst_transformer,
  title={TST Transformer: Automated Mobility Classification in Tail Suspension Test},
  author={Your Name},
  year={2024},
  url={https://github.com/Tarislada/tst-transformer}
}
```

## License


## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

- **Issues**: GitHub Issues
- **Documentation**: See `WORKFLOW.md` for detailed guide
- **Questions**: Open a discussion

---

**Quick Links:**
- [Detailed Workflow Guide](WORKFLOW.md)
- [Model Architecture Details](docs/architecture.md)
- [Training Tips](docs/training_tips.md)
