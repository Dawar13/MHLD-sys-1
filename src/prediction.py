"""
Prediction — Phase 9
Simple velocity-based future trajectory extrapolation.
"""

import numpy as np
from . import config


def predict_trajectory(
    positions: np.ndarray,
    n_steps: int = None,
    velocity_window: int = None,
) -> np.ndarray:
    """
    Predict future trajectory based on recent velocity.
    
    Args:
        positions: (N, 3) historical positions
        n_steps: number of future steps to predict
        velocity_window: number of recent positions to estimate velocity from
        
    Returns:
        (n_steps, 3) predicted future positions
    """
    if n_steps is None:
        n_steps = config.PREDICTION_STEPS
    if velocity_window is None:
        velocity_window = config.VELOCITY_WINDOW

    if len(positions) < 3:
        return np.zeros((2, 3))

    # Compute smoothed velocity from recent positions
    window = min(velocity_window, len(positions) - 1)
    recent = positions[-window - 1:]
    velocities = np.diff(recent, axis=0)
    avg_velocity = np.mean(velocities, axis=0)

    # Compute acceleration (for slight curve prediction)
    if len(velocities) >= 3:
        accelerations = np.diff(velocities, axis=0)
        avg_acceleration = np.mean(accelerations, axis=0) * 0.3  # Damped
    else:
        avg_acceleration = np.zeros(3)

    # Extrapolate
    current_pos = positions[-1].copy()
    current_vel = avg_velocity.copy()
    predicted = [current_pos.copy()]

    for step in range(1, n_steps + 1):
        current_vel += avg_acceleration * 0.1  # Gradual acceleration
        next_pos = predicted[-1] + current_vel
        predicted.append(next_pos)

    return np.array(predicted)
