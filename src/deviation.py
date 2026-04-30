"""
Deviation Detection — Phase 7-8
Compares current frame against global map to detect misalignment.
Uses nearest-neighbor distances with green→red color gradient.
"""

import numpy as np
import open3d as o3d
from . import config


def compute_deviation_colors(
    current_points: np.ndarray,
    map_points: np.ndarray,
    threshold: float = None,
    max_dist: float = None,
) -> np.ndarray:
    """
    Compute deviation colors for current frame points.
    Green = close to map (aligned), Red = far from map (deviated).
    
    Args:
        current_points: (N, 3) current frame in world coordinates
        map_points: (M, 3) global map points
        threshold: distance below which points are "aligned"
        max_dist: distance above which color is fully red
        
    Returns:
        (N, 3) RGB colors for each point
    """
    if threshold is None:
        threshold = config.DEVIATION_THRESHOLD
    if max_dist is None:
        max_dist = config.DEVIATION_MAX

    if map_points is None or len(map_points) < 10:
        # No map yet — return default live color
        return np.tile(config.COLOR_LIVE, (len(current_points), 1))

    # Build KD-tree from map
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points)
    kdtree = o3d.geometry.KDTreeFlann(map_pcd)

    n = len(current_points)
    colors = np.zeros((n, 3))

    green = np.array(config.COLOR_DEVIATION_LOW)
    red = np.array(config.COLOR_DEVIATION_HIGH)

    for i in range(n):
        # Find nearest neighbor in map
        _, idx, dist_sq = kdtree.search_knn_vector_3d(current_points[i], 1)
        dist = np.sqrt(dist_sq[0])

        # Normalize to [0, 1] range
        t = np.clip((dist - threshold) / (max_dist - threshold), 0.0, 1.0)

        # Interpolate green → red
        colors[i] = (1.0 - t) * green + t * red

    return colors


def compute_deviation_fast(
    current_points: np.ndarray,
    map_points: np.ndarray,
    sample_size: int = 5000,
) -> np.ndarray:
    """
    Fast deviation computation using random sampling.
    Samples a subset of current points for speed, applies color to all.
    """
    if map_points is None or len(map_points) < 10:
        return np.tile(config.COLOR_LIVE, (len(current_points), 1))

    n = len(current_points)

    # Build KD-tree from map
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points)
    kdtree = o3d.geometry.KDTreeFlann(map_pcd)

    green = np.array(config.COLOR_DEVIATION_LOW)
    red = np.array(config.COLOR_DEVIATION_HIGH)
    threshold = config.DEVIATION_THRESHOLD
    max_dist = config.DEVIATION_MAX

    colors = np.zeros((n, 3))

    # Sample for speed
    if n > sample_size:
        indices = np.random.choice(n, sample_size, replace=False)
    else:
        indices = np.arange(n)

    # Compute deviation for sampled points
    distances = np.zeros(n)
    for i in indices:
        _, _, dist_sq = kdtree.search_knn_vector_3d(current_points[i], 1)
        distances[i] = np.sqrt(dist_sq[0])

    # For non-sampled points, use average of nearby sampled points
    if n > sample_size:
        avg_dist = np.mean(distances[indices])
        distances[distances == 0] = avg_dist

    # Vectorized color computation
    t = np.clip((distances - threshold) / (max_dist - threshold), 0.0, 1.0)
    colors = np.outer(1.0 - t, green) + np.outer(t, red)

    return colors


def get_deviation_stats(
    current_points: np.ndarray, map_points: np.ndarray, sample_size: int = 1000
) -> dict:
    """
    Compute statistics about deviation.
    Returns dict with mean, max, and percentage anomalous.
    """
    if map_points is None or len(map_points) < 10:
        return {"mean": 0, "max": 0, "anomalous_pct": 0}

    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_points)
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
