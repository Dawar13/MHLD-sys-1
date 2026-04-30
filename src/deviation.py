"""
Deviation Detection - Phase 7-8
Compares current frame against global map to detect misalignment.
Uses nearest-neighbor distances with green->red color gradient.
Optimized for real-time performance with large maps.
"""

import numpy as np
import open3d as o3d
from . import config


def compute_deviation_fast(
    current_points: np.ndarray,
    map_points: np.ndarray,
    sample_size: int = 2000,
) -> np.ndarray:
    """
    Fast deviation computation using downsampled map and current frame.
    Green = close to map (aligned), Red = far from map (deviated).
    """
    n = len(current_points)
    
    if map_points is None or len(map_points) < 50:
        return np.tile(config.COLOR_LIVE, (n, 1))

    green = np.array(config.COLOR_DEVIATION_LOW)
    red = np.array(config.COLOR_DEVIATION_HIGH)
    threshold = config.DEVIATION_THRESHOLD
    max_dist = config.DEVIATION_MAX

    # Subsample map for KDTree (keep it under 50K points for speed)
    map_sub = map_points
    if len(map_points) > 50000:
        idx = np.random.choice(len(map_points), 50000, replace=False)
        map_sub = map_points[idx]

    # Build KDTree on subsampled map
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_sub)
    kdtree = o3d.geometry.KDTreeFlann(map_pcd)

    # Sample current points for speed
    if n > sample_size:
        sample_idx = np.random.choice(n, sample_size, replace=False)
    else:
        sample_idx = np.arange(n)

    # Compute distances for sampled points
    sampled_distances = np.zeros(len(sample_idx))
    for j, i in enumerate(sample_idx):
        _, _, dist_sq = kdtree.search_knn_vector_3d(current_points[i], 1)
        sampled_distances[j] = np.sqrt(dist_sq[0])

    # For all points, assign average distance (fast approximation)
    avg_dist = np.mean(sampled_distances)
    distances = np.full(n, avg_dist)
    distances[sample_idx] = sampled_distances

    # Vectorized color computation
    t = np.clip((distances - threshold) / (max_dist - threshold), 0.0, 1.0)
    colors = np.outer(1.0 - t, green) + np.outer(t, red)

    return colors


def get_deviation_stats(
    current_points: np.ndarray, map_points: np.ndarray, sample_size: int = 500
) -> dict:
    """
    Compute statistics about deviation.
    Returns dict with mean, max, and percentage anomalous.
    """
    if map_points is None or len(map_points) < 50:
        return {"mean": 0, "max": 0, "anomalous_pct": 0}

    # Subsample map
    map_sub = map_points
    if len(map_points) > 30000:
        idx = np.random.choice(len(map_points), 30000, replace=False)
        map_sub = map_points[idx]

    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_sub)
    kdtree = o3d.geometry.KDTreeFlann(map_pcd)

    n = min(sample_size, len(current_points))
    indices = np.random.choice(len(current_points), n, replace=False)
    distances = []

    for i in indices:
        _, _, dist_sq = kdtree.search_knn_vector_3d(current_points[i], 1)
        distances.append(np.sqrt(dist_sq[0]))

    distances = np.array(distances)
    return {
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
        "anomalous_pct": float(np.mean(distances > config.DEVIATION_THRESHOLD) * 100),
    }
