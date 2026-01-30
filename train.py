"""
Training script for TST Transformer classifier.

Trains a Transformer model on pre-extracted YOLO features.
Supports:
- Training/validation split
- Class balancing
- Learning rate scheduling
- Early stopping
- Checkpoint saving
- TensorBoard logging
- VS Code debugpy integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import argparse
import sys
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model import TSTModel, count_parameters
from dataset import create_dataloaders


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_epoch(model, loader, criterion, optimizer, device, epoch, threshold, writer=None):
    """Train for one epoch with per-frame predictions."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)  # [B, T, C]
        labels = labels.to(device).float()  # [B, T] as float
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features)  # [B, T, 1]
        
        # Flatten
        logits_flat = logits.squeeze(-1).reshape(-1)  # [B*T]
        labels_flat = labels.reshape(-1)  # [B*T]
        
        loss = criterion(logits_flat, labels_flat)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = (torch.sigmoid(logits_flat) > threshold).long()
        correct += (predictions == labels_flat.long()).sum().item()
        total += labels_flat.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100.0 * correct / total:.2f}%"
        })
        
        if writer is not None:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, loader, criterion, device, epoch, threshold, writer=None):
    """Validate for one epoch with per-frame predictions."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class metrics
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).float()
            
            # Forward pass
            logits = model(features)  # [B, T, 1]
            
            # Flatten
            logits_flat = logits.squeeze(-1).reshape(-1)
            labels_flat = labels.reshape(-1)
            
            loss = criterion(logits_flat, labels_flat)
            
            predictions = (torch.sigmoid(logits_flat) > threshold).long()
            labels_long = labels_flat.long()
            
            # Overall accuracy
            total += labels_flat.size(0)
            correct += (predictions == labels_long).sum().item()
            
            # Per-class accuracy
            for label in [0, 1]:
                mask = labels_long == label
                if mask.sum() > 0:
                    class_total[label] += mask.sum().item()
                    class_correct[label] += predictions[mask].eq(labels_long[mask]).sum().item()
            
            total_loss += loss.item()
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    mobile_acc = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
    immobile_acc = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0
    
    print(f"\nValidation Epoch {epoch}:")
    print(f"  Total frames: {total}")
    print(f"  Mobile: {class_total[0]} ({100*class_total[0]/total:.1f}%)")
    print(f"  Immobile: {class_total[1]} ({100*class_total[1]/total:.1f}%)")
    print(f"  Mobile Acc: {mobile_acc:.2f}%")
    print(f"  Immobile Acc: {immobile_acc:.2f}%\n")
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Accuracy', accuracy, epoch)
        writer.add_scalar('Val/Mobile_Acc', mobile_acc, epoch)
        writer.add_scalar('Val/Immobile_Acc', immobile_acc, epoch)
    
    return avg_loss, accuracy, (mobile_acc, immobile_acc)


def save_checkpoint(model, optimizer, epoch, val_loss, val_accuracy, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
    }, filepath)


def find_optimal_threshold(model, loader, device):
    """Find optimal threshold using F1 score on validation set."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            probs = torch.sigmoid(logits).squeeze(-1).reshape(-1)
            labels_flat = labels.reshape(-1)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels_flat.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Try thresholds from 0.1 to 0.9
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs > threshold).astype(int)
        
        # Calculate F1 for class 1 (immobile)
        tp = ((preds == 1) & (all_labels == 1)).sum()
        fp = ((preds == 1) & (all_labels == 0)).sum()
        fn = ((preds == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"  Optimal threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    return best_threshold


def main():
    parser = argparse.ArgumentParser(description="Train TST Transformer Classifier")
    
    # Data parameters
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing .pt feature files")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Directory containing .csv label files")
    parser.add_argument("--output_dir", type=str, default="../checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="../logs",
                        help="Directory for TensorBoard logs")
    
    # Window parameters (UPDATED)
    parser.add_argument("--window_size", type=int, default=120,
                        help="Window size in frames (e.g., 120 frames = 2s at 60fps)")
    parser.add_argument("--stride", type=int, default=60,
                        help="Stride in frames (e.g., 60 = 50%% overlap)")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Transformer hidden dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Loss parameters
    parser.add_argument("--use_class_weights", action="store_true",
                        help="Use class weights in loss function")
    parser.add_argument("--class_weight_multiplier", type=float, default=1.0,
                        help="Multiplier for minority class weight")
    parser.add_argument("--use_focal_loss", action="store_true",
                        help="Use focal loss instead of cross entropy")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--early_stopping", type=int, default=10,
                        help="Early stopping patience (0 to disable)")
    
    # Debug
    parser.add_argument("--debug", action="store_true",
                        help="Enable debugpy remote debugging")
    parser.add_argument("--debug_port", type=int, default=5678,
                        help="Port for debugpy")
    
    # Add threshold as a parameter
    parser.add_argument("--decision_threshold", type=float, default=0.5,
                        help="Decision threshold for binary classification (lower = more immobile predictions)")
    
    args = parser.parse_args()
    
    # Enable debugging if requested
    if args.debug:
        import debugpy
        debugpy.listen(("0.0.0.0", args.debug_port))
        print(f"🐛 Waiting for debugger to attach on port {args.debug_port}...")
        debugpy.wait_for_client()
        print("✓ Debugger attached!")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders (UPDATED call)
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        feature_dir=args.feature_dir,
        label_dir=args.label_dir,
        train_split=args.train_split,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    # Get feature dimension from first batch
    sample_features, _ = next(iter(train_loader))
    feature_dim = sample_features.shape[-1]
    
    print(f"\nDataset Info:")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Window size: {args.window_size} frames")
    print(f"  Stride: {args.stride} frames")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = TSTModel(
        yolo_feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        # num_classes=2,
        dropout=args.dropout,
        max_seq_length=args.window_size,
    )
    model = model.to(args.device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Setup loss function
    if args.use_class_weights:
        # Calculate class weights
        class_weights = train_loader.dataset.get_class_weights(args.class_weight_multiplier)
        pos_weight = class_weights[1] / class_weights[0]  # Weight for class 1
        pos_weight = pos_weight.to(args.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()


    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_accuracy = 0.0
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_accuracy = checkpoint.get('val_accuracy', 0.0)
        print(f"  Resumed from epoch {checkpoint['epoch']}")
        print(f"  Best val accuracy: {best_val_accuracy:.2f}%")
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir)
    
    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch, args.decision_threshold, writer
        )
        
        # Validate
        val_loss, val_acc, (mobile_acc, immobile_acc) = validate_epoch(
            model, val_loader, criterion, args.device, epoch, args.decision_threshold, writer
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                output_dir / "best_model.pt"
            )
            print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc,
            output_dir / "latest_model.pt"
        )
        
        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Find optimal threshold every 5 epochs
        if epoch % 5 == 0:  # Check every 5 epochs
            optimal_threshold = find_optimal_threshold(model, val_loader, args.device)
            print(f"  Consider using --decision_threshold {optimal_threshold:.2f}")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"{'='*60}\n")
    
    writer.close()


if __name__ == "__main__":
    main()
