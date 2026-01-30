"""
YOLO + Transformer model for TST immobility classification.

Architecture:
    Video frames → YOLO neck features → Temporal Transformer → Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from typing import List, Optional
import numpy as np


class YOLOFeatureExtractor:
    """
    Extract features from a frozen YOLO segmentation model's neck layer.
    Uses ROI pooling on the largest detected mask per frame.
    """
    
    def __init__(
        self, 
        yolo_path: str,
        layer_name: str = "model.18",  # Default neck layer
        device: str = "cuda"
    ):
        """
        Args:
            yolo_path: Path to trained YOLO .pt file
            layer_name: Which layer to extract features from (e.g., "model.18")
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load YOLO model
        print(f"Loading YOLO model from {yolo_path}...")
        self.yolo = YOLO(yolo_path)
        self.yolo.to(self.device)
        self.yolo.model.eval()
        
        # Freeze all parameters
        for param in self.yolo.model.parameters():
            param.requires_grad = False
        
        # Setup feature hook
        self.layer_name = layer_name
        self.feature_cache = {}
        self._register_hook()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        print(f"✓ YOLO loaded, extracting from {layer_name}, feature dim: {self.feature_dim}")
    
    def _register_hook(self):
        """Register forward hook to capture features from specified layer."""
        target_layer = self._get_layer_by_name(self.yolo.model, self.layer_name)
        
        def hook_fn(module, input, output):
            self.feature_cache['features'] = output.detach()
        
        target_layer.register_forward_hook(hook_fn)
    
    def _get_layer_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """Navigate to layer by dotted name (e.g., 'model.18')."""
        parts = name.split('.')
        layer = model
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer
    
    def _get_feature_dim(self) -> int:
        """Determine output feature dimension by running a dummy forward pass."""
        dummy = torch.randn(1, 3, 640, 640).to(self.device)
        with torch.no_grad():
            _ = self.yolo.model(dummy)
            feat = self.feature_cache['features']
            # Shape: [batch, channels, h, w]
            return feat.shape[1]
    
    @torch.no_grad()
    def extract_video_features(
        self, 
        video_path: str, 
        imgsz: int = 1080,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Extract features for all frames in a video.
        
        Args:
            video_path: Path to .mp4 video
            imgsz: Input size for YOLO (should match training size)
            batch_size: Number of frames to process at once
            
        Returns:
            features: numpy array of shape [num_frames, feature_dim]
        """
        import cv2
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {total_frames} frames at {fps} fps")
        
        all_features = []
        frame_idx = 0
        
        while True:
            # Read batch of frames
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
            
            if not batch_frames:
                break
            
            # Run YOLO segmentation
            results = self.yolo.predict(
                batch_frames,
                imgsz=imgsz,
                verbose=False,
                device=self.device,
                half=False,  # Use full precision for feature extraction
            )
            
            # Get features from hook
            batch_features = self.feature_cache['features']  # [B, C, H, W]
            
            # For each frame, extract features using the largest mask
            for i, result in enumerate(results):
                feat_map = batch_features[i:i+1]  # [1, C, H, W]
                
                # Get largest mask if available
                if result.masks is not None and len(result.masks) > 0:
                    # Find largest mask
                    masks = result.masks.data.cpu().numpy()  # [num_masks, h, w]
                    areas = [mask.sum() for mask in masks]
                    largest_idx = np.argmax(areas)
                    mask = masks[largest_idx]
                    
                    # Resize mask to match feature map size
                    H, W = feat_map.shape[2:]
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8) * 255, 
                        (W, H), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_resized = torch.from_numpy(mask_resized > 127).float().to(self.device)
                    
                    # Apply mask and pool
                    masked_feat = feat_map * mask_resized.unsqueeze(0).unsqueeze(0)
                    pooled = masked_feat.sum(dim=[2, 3]) / (mask_resized.sum() + 1e-6)
                else:
                    # No mask found, use global average pooling
                    pooled = feat_map.mean(dim=[2, 3])
                
                all_features.append(pooled.cpu().numpy())
            
            frame_idx += len(batch_frames)
            if frame_idx % 300 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        
        features = np.concatenate(all_features, axis=0)  # [T, C]
        print(f"✓ Extracted features: {features.shape}")
        return features

class TSTModel(nn.Module):
    """
    Transformer-based Temporal Sequence Transformer for TST classification.
    
    Architecture:
    1. Input: [batch, seq_len, yolo_feature_dim]
    2. Linear projection to hidden_dim
    3. Add positional encoding
    4. Transformer encoder layers
    5. Global average pooling over time
    6. Classification head
    """
    
    def __init__(
        self,
        yolo_feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_length: int = 300,  # Changed from 100 to 300 (10 seconds at 30fps)
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Feature projection
        self.input_projection = nn.Linear(yolo_feature_dim, hidden_dim)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Binary classification head (1 output instead of 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Changed from 2 to 1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with per-frame binary predictions.
        
        Args:
            x: [batch_size, seq_len, yolo_feature_dim]
            
        Returns:
            logits: [batch_size, seq_len, 1] binary logits
        """
        batch_size, seq_len, _ = x.shape
        
        if seq_len > self.max_seq_length:
            raise ValueError(f"Seq length {seq_len} exceeds max {self.max_seq_length}")
        
        # Project and add positional encoding
        x = self.input_projection(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)  # [B, T, hidden_dim]
        
        # NO POOLING - classify each frame independently
        logits = self.classifier(x)  # [B, T, 1]
        
        return logits


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Simulate YOLO features
    batch_size = 4
    seq_len = 60  # 2 seconds at 30fps
    feature_dim = 512  # Typical YOLO neck dimension
    
    dummy_features = torch.randn(batch_size, seq_len, feature_dim)
    
    # Create model
    model = TSTModel(
        yolo_feature_dim=feature_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
    )
    
    # Forward pass
    logits = model(dummy_features)
    predictions = model.predict(dummy_features)
    probas = model.predict_proba(dummy_features)
    
    print(f"✓ Model created successfully")
    print(f"  Trainable parameters: {count_parameters(model):,}")
    print(f"  Input shape: {dummy_features.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Probabilities: {probas.shape}")
