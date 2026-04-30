"""
Trajectory Tracker — Phase 4
Tracks system state (position) over time and provides line geometry.
"""

import numpy as np


class TrajectoryTracker:
    """Accumulates positions from poses and provides trajectory data."""

    def __init__(self):
        self.positions = []

    def add_pose(self, pose: np.ndarray):
        """
        Extract translation from a 3x4 pose matrix and append.
        """
        t = pose[:3, 3].copy()
        self.positions.append(t)

    def get_positions_array(self) -> np.ndarray:
        """Return all positions as (N, 3) array."""
        if len(self.positions) < 1:
            return np.zeros((1, 3))
        return np.array(self.positions)

    def get_current_position(self) -> np.ndarray:
        """Return the latest position."""
        if not self.positions:
            return np.zeros(3)
        return self.positions[-1]

    def get_velocity(self, window: int = 5) -> np.ndarray:
        """
        Estimate current velocity from recent positions.
        Returns velocity vector (3,).
        """
        if len(self.positions) < 2:
            return np.zeros(3)

        n = min(window, len(self.positions) - 1)
        recent = np.array(self.positions[-n - 1:])
        velocities = np.diff(recent, axis=0)
        return np.mean(velocities, axis=0)

    def get_heading(self) -> np.ndarray:
        """Get current heading direction (normalized velocity)."""
        v = self.get_velocity()
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0])
        return v / norm

    @property
    def count(self) -> int:
        return len(self.positions)
