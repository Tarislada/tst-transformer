"""Post-processing utilities for frame-level predictions."""

import pandas as pd
import numpy as np


def aggregate_frames_to_seconds(
    frame_predictions: np.ndarray,
    frame_probabilities: np.ndarray = None,
    fps: int = 60,
    method: str = 'majority',
    confidence_threshold: float = None
) -> np.ndarray:
    """
    Aggregate frame-level predictions to second-level with optional confidence filtering.
    
    Args:
        frame_predictions: [num_frames] array of 0/1 predictions
        frame_probabilities: [num_frames] array of confidence scores (0-1)
        fps: Frames per second
        method: 'majority', 'any', 'all', or 'mean_threshold'
        confidence_threshold: If provided, revert predictions to 0 when average
                            confidence in a second is below this threshold
    
    Returns:
        second_predictions: [num_seconds] array of 0/1 predictions
    """
    num_frames = len(frame_predictions)
    num_seconds = num_frames // fps
    
    second_preds = []
    
    for sec in range(num_seconds):
        start_frame = sec * fps
        end_frame = (sec + 1) * fps
        frames_in_second = frame_predictions[start_frame:end_frame]
        
        if method == 'majority':
            # Label=1 if >50% frames are immobile
            pred = 1 if frames_in_second.mean() > 0.5 else 0
        elif method == 'any':
            # Label=1 if ANY frame is immobile
            pred = 1 if frames_in_second.max() == 1 else 0
        elif method == 'all':
            # Label=1 if ALL frames are immobile
            pred = 1 if frames_in_second.min() == 1 else 0
        elif method == 'mean_threshold':
            # Configurable threshold
            pred = 1 if frames_in_second.mean() > 0.7 else 0
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply confidence-based filtering
        if confidence_threshold is not None and frame_probabilities is not None and pred == 1:
            # Only filter out positive predictions (immobility)
            probs_in_second = frame_probabilities[start_frame:end_frame]
            avg_confidence = probs_in_second.mean()
            
            if avg_confidence < confidence_threshold:
                pred = 0  # Revert to non-immobile
        
        second_preds.append(pred)
    
    return np.array(second_preds)