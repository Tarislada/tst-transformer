# TST Transformer Classifier

Deep learning system for Tail Suspension Test (TST) immobility classification using YOLO features + Transformer.

## Overview

This system uses a two-stage approach:
1. **YOLO feature extraction**: Extract spatial features from each video frame using a pre-trained segmentation model
2. **Transformer classification**: Temporal transformer model predicts per-second immobility labels

## Architecture

```
Video (.mp4) 
    ↓
YOLO Segmentation Model (frozen)
    ↓
Frame Features [T, C] 
    ↓
Temporal Windows [batch, window_frames, C]
    ↓
Transformer Encoder (2-3 layers)
    ↓
Classification Head
    ↓
Predictions (per-second) [0=mobile, 1=immobile]
```

## Installation

### Requirements
- Python 3.9+
- CUDA-capable GPU (8GB VRAM minimum)
- PyTorch 2.0+
- ultralytics (YOLO)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Prepare Your Data

Organize your data as follows:
```
data/
├── videos/                 # Raw videos
│   ├── KHC2_M1_D10.mp4
│   ├── KHC2_M2_D10.mp4
│   └── ...
├── labels_excel/          # Excel label files
│   ├── Filtered_TST_Manual_Scoring_D10_M1.xlsx
│   ├── Filtered_TST_Manual_Scoring_D10_M2.xlsx
│   └── ...
models/
└── best.pt               # Your trained YOLO model
```

### 2. Convert Labels

Convert Excel labels to CSV format:

```bash
python convert_labels_from_excel.py convert-batch \
    --excel-glob "data/labels_excel/*.xlsx" \
    --out-dir "data/labels_csv"
```

Expected output format:
```csv
second,label
0,0      # 0 = mobile
1,0
2,1      # 1 = immobile
...
```

### 3. Extract YOLO Features

**Important**: First inspect your YOLO model to choose the best layer:

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
    --yolo_path /home/tarislada/Documents/Extra_python_projects/Natalie/tst_pipeline/best.pt \
    --layer_name model.18 \
    --imgsz 1080 \
    --batch_size 8 \
    --device cuda
```

**This will take a while!** For 71 videos × 3 minutes = ~3-4 hours.

### 4. Train the Model

Train the transformer classifier:

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
    --use_class_weights \
    --device cuda
```

**Monitor training** with TensorBoard:
```bash
tensorboard --logdir ../logs
```

### 5. Run Inference

Predict on a new video:

```bash
python inference.py \
    --video ../data/videos/KHC2_M1_D10.mp4 \
    --model ../checkpoints/best_model.pt \
    --yolo_path /home/tarislada/Documents/Extra_python_projects/Natalie/tst_pipeline/best.pt \
    --output predictions.csv \
    --device cuda
```

Or batch process multiple videos:

```bash
python batch_inference.py \
    --video_dir ../data/videos \
    --model ../checkpoints/best_model.pt \
    --yolo_path /home/tarislada/Documents/Extra_python_projects/Natalie/tst_pipeline/best.pt \
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

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--hidden_dim` (try 128)
- Reduce `--window_seconds` (try 1.0)
- Use smaller YOLO `--imgsz` during feature extraction

### Poor Performance
- Try different YOLO layers (model.15, model.18, model.21)
- Increase `--window_seconds` for more temporal context
- Enable `--use_class_weights` if classes are imbalanced
- Check if labels are correctly aligned with videos

### Training Not Converging
- Lower learning rate: `--lr 5e-5` or `--lr 1e-5`
- Add more regularization: `--dropout 0.2`
- Check class balance in data

## File Structure

```
tst_transformer/
├── model.py              # Model definitions
├── dataset.py            # Data loading
├── preprocess.py         # Feature extraction
├── train.py              # Training script
├── inference.py          # Single video inference
├── batch_inference.py    # Batch inference
├── requirements.txt      # Python dependencies
└── README.md            # This file

convert_labels_from_excel.py  # Label conversion utility
```

## Performance Notes

### Memory Usage
- Feature extraction: ~3-4 GB VRAM
- Training: ~4-6 GB VRAM (batch_size=32)
- Inference: ~2-3 GB VRAM

### Speed
- Feature extraction: ~5-10 seconds per video (3 min video)
- Training: ~2-3 minutes per epoch (71 videos, batch_size=32)
- Inference: ~5-10 seconds per video

## Advanced: Custom YOLO Models

If you train a new YOLO model:

1. Update `--yolo_path` in all commands
2. Update `--imgsz` to match your YOLO training size
3. Re-run feature extraction (features are model-specific)
4. Retrain the transformer

## Citation

If you use this code, please cite:
```
[Your paper/repository citation here]
```

## License

[Your license here]
