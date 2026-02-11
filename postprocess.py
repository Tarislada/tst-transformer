"""Post-processing utilities for frame-level predictions."""

import pandas as pd
import numpy as np


def aggregate_frames_to_seconds(
    frame_predictions: np.ndarray,
    frame_probabilities: np.ndarray = None,
    fps: int = 60,
    agg_threshold: float = 0.5,
    confidence_threshold: float = None
) -> np.ndarray:
    """
    Aggregate frame-level predictions to second-level with optional confidence filtering.
    
    Args:
        frame_predictions: [num_frames] array of 0/1 predictions
        frame_probabilities: [num_frames] array of confidence scores (0-1)
        fps: Frames per second
        agg_threshold: Threshold for aggregation (0.0 to 1.0):
                      - 0.0 = 'any' (at least one immobile frame → immobile second)
                      - 0.5 = 'majority' (>50% immobile frames → immobile second)
                      - 1.0 = 'all' (all frames immobile → immobile second)
                      - Any value in between works as custom threshold
        confidence_threshold: If provided, revert immobility predictions to mobile
                            when average confidence in a second is below this threshold
    
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
        
        # Calculate percentage of immobile frames
        immobile_ratio = frames_in_second.mean()
        
        # Apply aggregation threshold
        if agg_threshold == 0.0:
            # Special case: any immobile frame
            pred = 1 if immobile_ratio > 0 else 0
        else:
            # General case: threshold-based
            pred = 1 if immobile_ratio >= agg_threshold else 0
        
        # Apply confidence-based filtering (only for immobility predictions)
        if confidence_threshold is not None and frame_probabilities is not None and pred == 1:
            probs_in_second = frame_probabilities[start_frame:end_frame]
            avg_confidence = probs_in_second.mean()
            
            if avg_confidence < confidence_threshold:
                pred = 0  # Revert to mobile
        
        second_preds.append(pred)
    
    return np.array(second_preds)