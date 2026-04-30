"""
Synthetic LiDAR Generator
Generates realistic street-like point clouds using real poses.
Produces .bin files in KITTI format so the pipeline works identically.
"""

import os
import numpy as np
from . import config


def _generate_ground_plane(n_points: int, range_m: float) -> np.ndarray:
    """Generate a flat ground plane with slight noise."""
    x = np.random.uniform(-range_m, range_m, n_points)
    y = np.random.uniform(-range_m, range_m, n_points)
    z = np.random.normal(-1.7, 0.05, n_points)  # ~1.7m below sensor (car roof height)
    return np.column_stack([x, y, z])


def _generate_buildings(n_points: int, range_m: float) -> np.ndarray:
    """Generate building-like structures on both sides of the road."""
    points = []
    for side in [-1, 1]:
        n = n_points // 2
        # Wall face
        x = np.random.uniform(0, range_m * 0.8, n) * np.sign(np.random.randn(n))
        y = np.full(n, side * np.random.uniform(8, 15, n))
        z = np.random.uniform(-1.7, 6.0, n)
        points.append(np.column_stack([x, y, z]))
    return np.vstack(points)


def _generate_poles(n_poles: int, range_m: float) -> np.ndarray:
    """Generate vertical pole-like structures (street lights, signs)."""
    points = []
    for _ in range(n_poles):
        pole_x = np.random.uniform(-range_m * 0.5, range_m * 0.5)
        pole_y = np.random.choice([-1, 1]) * np.random.uniform(3, 6)
        n_pts = 30
        z = np.linspace(-1.7, 4.0, n_pts)
        x = np.full(n_pts, pole_x) + np.random.normal(0, 0.03, n_pts)
        y = np.full(n_pts, pole_y) + np.random.normal(0, 0.03, n_pts)
        points.append(np.column_stack([x, y, z]))
    if points:
        return np.vstack(points)
    return np.zeros((0, 3))


def _generate_vegetation(n_points: int, range_m: float) -> np.ndarray:
    """Generate tree-like clusters."""
    points = []
    n_trees = max(1, n_points // 100)
    pts_per_tree = n_points // n_trees
    for _ in range(n_trees):
        cx = np.random.uniform(-range_m * 0.6, range_m * 0.6)
        cy = np.random.choice([-1, 1]) * np.random.uniform(5, 12)
        # Trunk
        n_trunk = pts_per_tree // 4
        trunk_z = np.random.uniform(-1.7, 2.0, n_trunk)
        trunk_x = cx + np.random.normal(0, 0.1, n_trunk)
        trunk_y = cy + np.random.normal(0, 0.1, n_trunk)
        # Canopy (sphere-ish)
        n_canopy = pts_per_tree - n_trunk
        theta = np.random.uniform(0, 2 * np.pi, n_canopy)
        phi = np.random.uniform(0, np.pi, n_canopy)
        r = np.random.uniform(0.5, 2.5, n_canopy)
        canopy_x = cx + r * np.sin(phi) * np.cos(theta)
        canopy_y = cy + r * np.sin(phi) * np.sin(theta)
        canopy_z = 3.0 + r * np.cos(phi)
        points.append(np.column_stack([trunk_x, trunk_y, trunk_z]))
        points.append(np.column_stack([canopy_x, canopy_y, canopy_z]))
    if points:
        return np.vstack(points)
    return np.zeros((0, 3))


def _simulate_lidar_scan(points: np.ndarray, max_range: float) -> np.ndarray:
    """
    Simulate LiDAR scanning: filter by range and add noise.
    """
    distances = np.linalg.norm(points, axis=1)
    mask = distances < max_range
    points = points[mask]
    # Add sensor noise
    noise = np.random.normal(0, 0.02, points.shape)
    points += noise
    return points


def generate_single_frame(frame_idx: int, seed: int = None) -> np.ndarray:
    """
    Generate a single synthetic LiDAR frame.
    Returns (N, 3) point cloud in sensor coordinates.
    """
    if seed is not None:
        np.random.seed(seed + frame_idx)
    
    total = config.SYNTHETIC_POINTS_PER_FRAME
    range_m = config.SYNTHETIC_RANGE
    
    # Allocate point budget
    ground = _generate_ground_plane(int(total * 0.40), range_m)
    buildings = _generate_buildings(int(total * 0.30), range_m)
    poles = _generate_poles(np.random.randint(3, 8), range_m)
    trees = _generate_vegetation(int(total * 0.15), range_m)
    
    # Combine
    all_points = np.vstack([ground, buildings, poles, trees])
    
    # Simulate LiDAR properties
    all_points = _simulate_lidar_scan(all_points, range_m)
    
    return all_points.astype(np.float32)


def generate_dataset(poses: np.ndarray, output_dir: str, frame_skip: int = 1):
    """
    Generate a full synthetic dataset matching the number of poses.
    Saves .bin files in KITTI format.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(poses)
    
    print(f"  Generating {n_frames} synthetic LiDAR frames...")
    
    for i in range(0, n_frames, frame_skip):
        points = generate_single_frame(i, seed=42)
        
        # Convert to KITTI format (N, 4) with reflectance channel
        reflectance = np.random.uniform(0.1, 1.0, (len(points), 1)).astype(np.float32)
        kitti_points = np.hstack([points, reflectance])
        
        # Save as .bin
        filename = os.path.join(output_dir, f"{i:06d}.bin")
        kitti_points.tofile(filename)
        
        if i % 500 == 0:
            print(f"    Frame {i}/{n_frames} generated")
    
    print(f"  [OK] Dataset generated: {len(os.listdir(output_dir))} frames in {output_dir}")
