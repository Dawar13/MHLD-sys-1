"""
Data Loader — Phase 1
Loads KITTI-format point clouds (.bin) and ground-truth poses.
"""

import os
import glob
import numpy as np


def load_point_cloud(file_path: str) -> np.ndarray:
    """
    Load a single point cloud from a KITTI .bin file.
    Returns (N, 3) array of XYZ coordinates.
    """
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3]


def get_frame_files(velodyne_dir: str) -> list:
    """
    Get sorted list of .bin file paths from velodyne directory.
    """
    files = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
    return files


def load_poses(poses_file: str) -> np.ndarray:
    """
    Load poses from KITTI poses.txt file.
    Each line has 12 values (3x4 transformation matrix, row-major).
    Returns (N, 3, 4) array.
    """
    poses_raw = np.loadtxt(poses_file)
    n_poses = poses_raw.shape[0]
    poses = poses_raw.reshape(n_poses, 3, 4)
    return poses


def get_transform(pose: np.ndarray):
    """
    Extract rotation matrix R (3x3) and translation vector t (3x1) from a 3x4 pose.
    Returns (R, t).
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    return R, t


def get_positions(poses: np.ndarray) -> np.ndarray:
    """
    Extract all XYZ positions from poses array.
    Returns (N, 3) array.
    """
    return poses[:, :3, 3]
