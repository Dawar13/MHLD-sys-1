"""
Pipeline - Master Orchestrator
Ties all phases together into a cohesive demo flow.
Supports both real KITTI data (.bin files) and on-the-fly synthetic generation.
"""

import os
import sys
import time
import numpy as np
import open3d as o3d

from . import config
from .data_loader import load_point_cloud, get_frame_files, load_poses
from .synthetic_generator import generate_single_frame
from .visualizer import SceneVisualizer
from .trajectory import TrajectoryTracker
from .map_builder import MapBuilder
from .deviation import compute_deviation_fast, get_deviation_stats
from .prediction import predict_trajectory


def _print_header():
    """Print project banner."""
    print()
    print("=" * 65)
    print("  PROJECT KD -- Multi-Sensor State Reconstruction")
    print("  -------------------------------------------------")
    print("  Real-time LiDAR visualization | Trajectory tracking")
    print("  Map building | Deviation detection | Prediction")
    print("=" * 65)
    print()


def _print_narration(key: str):
    """Print demo narration."""
    msg = {
        "raw":        "> SCENE 1: Raw sensor data -- fragmented and unstructured.",
        "motion":     "> SCENE 2: Motion starts -- streaming LiDAR in real-time.",
        "trajectory": "> SCENE 3: System state over time -- trajectory building.",
        "map":        "> SCENE 4: System memory -- map accumulating.",
        "overlay":    "> SCENE 5: Expected vs observed -- live overlay on map.",
        "deviation":  "> SCENE 6: Intelligence begins -- detecting mismatch.",
        "prediction": "> SCENE 7: Future prediction -- anticipating trajectory.",
        "insight":    "\n[!] KEY INSIGHT: Before any AI model, you need a correct\n"
                      "    representation of system behavior.\n"
                      "    Most industrial systems don't have this layer.\n",
    }
    if key in msg:
        print(f"\n{msg[key]}")


def run(frame_skip: int = None, max_frames: int = None, playback_delay: float = None):
    """
    Run the full demo pipeline.
    """
    if frame_skip is None:
        frame_skip = config.FRAME_SKIP
    if max_frames is None:
        max_frames = config.MAX_FRAMES
    if playback_delay is None:
        playback_delay = config.PLAYBACK_DELAY

    _print_header()

    # -- Phase 0: Load data --
    print("[Phase 0] Loading data...")
    
    if not os.path.exists(config.POSES_FILE):
        print(f"  [X] Poses file not found: {config.POSES_FILE}")
        print("  Please place poses.txt in the data/ directory.")
        sys.exit(1)

    poses = load_poses(config.POSES_FILE)
    print(f"  [OK] Loaded {len(poses)} poses from {config.POSES_FILE}")

    # Check for real KITTI data
    bin_files = get_frame_files(config.VELODYNE_DIR)
    use_real_data = len(bin_files) > 0

    if use_real_data:
        print(f"  [OK] Found {len(bin_files)} real KITTI frames")
        n_total = min(len(bin_files), len(poses))
    else:
        print("  [!] No velodyne .bin files found.")
        print("  --> Using on-the-fly synthetic LiDAR generation (zero disk usage)")
        n_total = len(poses)

    # Compute frame indices
    frame_indices = list(range(0, n_total, frame_skip))
    if max_frames > 0:
        frame_indices = frame_indices[:max_frames]

    print(f"  --> Processing {len(frame_indices)} frames (skip={frame_skip})")

    # -- Phase 1: Initialize components --
    print("\n[Phase 1] Initializing visualization engine...")
    
    vis = SceneVisualizer()
    tracker = TrajectoryTracker()
    mapper = MapBuilder()

    # -- Narration phases (based on step count, not frame index) --
    phase_thresholds = {
        "raw": 0,
        "motion": 3,
        "trajectory": 15,
        "map": 40,
        "overlay": 80,
        "deviation": 120,
        "prediction": 160,
    }

    _print_narration("raw")

    # -- Main loop --
    print("\n[Phase 2-9] Running demo...")
    print("  (Close the 3D window to stop)\n")

    announced_phases = set()
    start_time = time.time()

    for step, frame_idx in enumerate(frame_indices):
        # -- Narration --
        for phase_name, threshold in phase_thresholds.items():
            if step >= threshold and phase_name not in announced_phases:
                _print_narration(phase_name)
                announced_phases.add(phase_name)

        try:
            # -- Load or generate point cloud --
            if use_real_data:
                points = load_point_cloud(bin_files[frame_idx])
            else:
                points = generate_single_frame(frame_idx, seed=42)

            # Downsample current frame for speed
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_down = pcd.voxel_down_sample(config.VOXEL_SIZE)
            points_down = np.asarray(pcd_down.points)

            # -- Get pose --
            pose = poses[frame_idx]

            # -- Phase 4: Track trajectory --
            tracker.add_pose(pose)

            # -- Phase 6: Transform to world coordinates --
            world_points = mapper.transform_points(points_down, pose)

            # -- Phase 5: Add to map (every few frames for performance) --
            if step % 4 == 0:
                mapper.add_frame(points_down, pose)

            # -- Phase 8: Compute deviation colors --
            if step > phase_thresholds.get("deviation", 120):
                colors = compute_deviation_fast(
                    world_points, mapper.get_map_points(), sample_size=1500
                )
            elif step > phase_thresholds.get("overlay", 80):
                # Overlay phase: brighter cyan
                colors = np.tile([0.0, 1.0, 1.0], (len(world_points), 1))
            else:
                # Before overlay: default live color
                colors = np.tile(config.COLOR_LIVE, (len(world_points), 1))

            # -- Phase 9: Prediction --
            positions = tracker.get_positions_array()
            if step > phase_thresholds.get("prediction", 160) and len(positions) > 10:
                predicted = predict_trajectory(positions)
                vis.update_prediction(predicted)

            # -- Update visualization --
            vis.update_live_cloud(world_points, colors)
            vis.update_trajectory(positions)

            if step % 5 == 0 and mapper.point_count > 0:
                vis.update_map_cloud(mapper.get_map_points())

            # Camera: initialize on frame 3 (after window is properly sized),
            # then follow every 30 frames
            if step == 3:
                vis.setup_initial_view(tracker.get_current_position())
            elif step > 3 and step % 30 == 0:
                vis.follow_position(tracker.get_current_position())

            # -- Render --
            vis.tick()
            time.sleep(playback_delay)

            # -- Status line --
            elapsed = time.time() - start_time
            fps = (step + 1) / max(elapsed, 0.001)
            if step % 50 == 0:
                stats_str = ""
                if step > phase_thresholds.get("deviation", 120):
                    stats = get_deviation_stats(
                        world_points, mapper.get_map_points(), sample_size=300
                    )
                    stats_str = (
                        f" | Dev: {stats['mean']:.2f}m"
                        f" ({stats['anomalous_pct']:.0f}% anomalous)"
                    )
                print(
                    f"  Frame {frame_idx:5d}/{n_total}"
                    f" | Map: {mapper.point_count:,} pts"
                    f" | FPS: {fps:.1f}"
                    f"{stats_str}"
                )

        except Exception as e:
            print(f"  [!] Frame {frame_idx}: {e}")
            continue

    # -- End --
    _print_narration("insight")
    print(f"\n  [OK] Demo complete. {len(frame_indices)} frames processed.")
    print("  --> Window stays open. Close it to exit.\n")

    try:
        vis.run_until_closed()
    except Exception:
        pass
    finally:
        vis.destroy()
