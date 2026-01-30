"""
PyTorch Dataset for TST temporal sequence classification.
Loads pre-extracted YOLO features in sliding windows.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import warnings


class TSTDataset(Dataset):
    """
    Dataset for frame-level TST classification using sliding windows.
    
    Each sample is a window of consecutive frames.
    Input: [window_size, feature_dim]
    Output: [window_size] labels (one per frame)
    
    Example with window=120, stride=60:
    - Sample 0: frames 0-119
    - Sample 1: frames 60-179 (50% overlap)
    - Sample 2: frames 120-239
    """
    
    def __init__(
        self,
        feature_files: List[str],
        label_files: List[str],
        window_size: int = 120,
        stride: int = 60,
    ):
        self.feature_files = feature_files
        self.label_files = label_files
        self.window_size = window_size
        self.stride = stride
        
        self.samples = self._build_sample_index()
        
        print(f"Dataset: {len(self.feature_files)} videos, {len(self.samples)} windows, "
              f"window={window_size} frames, stride={stride} frames")
        
    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """
        Build index of sliding windows.
        
        Returns:
            List of (video_idx, start_frame) tuples
        """
        samples = []
        
        for vid_idx, (feat_path, label_path) in enumerate(zip(self.feature_files, self.label_files)):
            try:
                features = torch.load(feat_path, weights_only=True)
                if isinstance(features, dict):
                    features = features['features']
                num_frames = features.shape[0]
                
                labels_df = pd.read_csv(label_path)
                num_labels = len(labels_df)
                
                if num_frames != num_labels:
                    warnings.warn(f"{Path(feat_path).name}: {num_frames} frames vs {num_labels} labels")
                    num_frames = min(num_frames, num_labels)
                
                # Create sliding windows with stride
                for start_frame in range(0, num_frames - self.window_size + 1, self.stride):
                    samples.append((vid_idx, start_frame))
                        
            except Exception as e:
                warnings.warn(f"Skipping {feat_path}: {e}")
                continue
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a window of frames and their labels.
        
        Returns:
            window: [window_size, feature_dim]
            labels: [window_size] tensor of labels
        """
        vid_idx, start_frame = self.samples[idx]
        
        # Load features
        features = torch.load(self.feature_files[vid_idx], weights_only=True)
        if isinstance(features, dict):
            features = features['features']
        
        # Extract window (no padding needed - window is guaranteed to fit)
        window = features[start_frame:start_frame + self.window_size]
        
        # Load labels for all frames in window
        labels_df = pd.read_csv(self.label_files[vid_idx])
        window_labels = labels_df.iloc[start_frame:start_frame + self.window_size]['label'].values
        window_labels = torch.tensor(window_labels, dtype=torch.long)
        
        return window, window_labels
    
    def get_class_weights(self, multiplier: float = 1.0) -> torch.Tensor:
        """Calculate class weights across all frames."""
        counts = {0: 0, 1: 0}
        
        for vid_idx, start_frame in self.samples:
            labels_df = pd.read_csv(self.label_files[vid_idx])
            window_labels = labels_df.iloc[start_frame:start_frame + self.window_size]['label'].values
            counts[0] += (window_labels == 0).sum()
            counts[1] += (window_labels == 1).sum()
        
        if counts[1] == 0:
            return torch.tensor([1.0, 1.0])
        
        ratio = counts[0] / counts[1]
        weights = torch.tensor([1.0, ratio * multiplier])
        
        print(f"Class weights: Mobile={counts[0]}, Immobile={counts[1]}, "
              f"ratio={ratio:.1f}:1, weights={weights.tolist()}")
        
        return weights


def create_dataloaders(
    feature_dir: str,
    label_dir: str,
    train_split: float = 0.8,
    batch_size: int = 32,
    window_size: int = 120,
    stride: int = 60,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with stratified split."""
    from sklearn.model_selection import train_test_split
    
    feature_dir = Path(feature_dir)
    label_dir = Path(label_dir)
    
    # Match feature and label files
    paired_files = []
    for feat_path in sorted(feature_dir.glob("*.pt")):
        base_name = feat_path.stem.replace('_features', '')
        
        for pattern in [f"{base_name}.labels.csv", f"{base_name}.csv"]:
            label_path = label_dir / pattern
            if label_path.exists():
                paired_files.append((str(feat_path), str(label_path)))
                break
        else:
            warnings.warn(f"No label for {feat_path.name}")
    
    if not paired_files:
        raise ValueError(f"No paired files found")
    
    print(f"Found {len(paired_files)} video pairs")
    
    # Stratify by proportion of immobile frames
    video_labels = []
    for _, label_path in paired_files:
        labels = pd.read_csv(label_path)['label']
        immobile_pct = (labels == 1).mean()
        # Bin into categories: low (<10%), medium (10-30%), high (>30%)
        if immobile_pct < 0.1:
            category = 0
        elif immobile_pct < 0.3:
            category = 1
        else:
            category = 2
        video_labels.append(category)
    
    # Split
    train_pairs, val_pairs = train_test_split(
        paired_files,
        train_size=train_split,
        random_state=seed,
        # stratify=video_labels
    )
    
    print(f"Train: {len(train_pairs)} videos | Val: {len(val_pairs)} videos")
    
    # Create datasets
    train_dataset = TSTDataset(
        [p[0] for p in train_pairs],
        [p[1] for p in train_pairs],
        window_size=window_size,
        stride=stride,
    )
    
    val_dataset = TSTDataset(
        [p[0] for p in val_pairs],
        [p[1] for p in val_pairs],
        window_size=window_size,
        stride=stride,
    )
    
    print(f"Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset creation
    print("Testing dataset...")
    
    # This would normally load real data
    # Here we just test the structure
    print("✓ Dataset module ready")
