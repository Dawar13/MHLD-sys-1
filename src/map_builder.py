"""
Map Builder — Phase 5-6
Accumulates past observations into a unified global map.
Handles coordinate transformation (sensor alignment).
"""

import numpy as np
import open3d as o3d
from . import config


class MapBuilder:
    """
    Maintains a global point cloud map by accumulating transformed frames.
    """

    def __init__(self):
        self.global_points = None
        self._frame_count = 0

    def transform_points(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """
        Transform points from sensor coordinates to world coordinates.
        pose: (3, 4) transformation matrix [R | t]
        Returns transformed (N, 3) points.
        """
        R = pose[:3, :3]
        t = pose[:3, 3]
        # Apply: world_point = R @ sensor_point + t
        transformed = (R @ points.T).T + t
        return transformed

    def add_frame(self, points: np.ndarray, pose: np.ndarray):
        """
        Transform and merge a frame into the global map.
        """
        transformed = self.transform_points(points, pose)

        if self.global_points is None:
            self.global_points = transformed
        else:
            self.global_points = np.vstack([self.global_points, transformed])

        self._frame_count += 1

        # Periodic downsampling to keep map manageable
        if self._frame_count % 20 == 0:
            self.downsample()

    def downsample(self):
        """Voxel downsample the global map to control memory."""
        if self.global_points is None or len(self.global_points) == 0:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.global_points)
        pcd_down = pcd.voxel_down_sample(config.MAP_VOXEL_SIZE)
        self.global_points = np.asarray(pcd_down.points)

        # Hard cap on total points
        if len(self.global_points) > config.MAX_MAP_POINTS:
            indices = np.random.choice(
                len(self.global_points), config.MAX_MAP_POINTS, replace=False
            )
            self.global_points = self.global_points[indices]

    def get_map_points(self) -> np.ndarray:
        """Return the current global map points."""
        if self.global_points is None:
            return np.zeros((1, 3))
        return self.global_points

    @property
    def point_count(self) -> int:
        if self.global_points is None:
            return 0
        return len(self.global_points)
