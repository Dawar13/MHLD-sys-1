#!/usr/bin/env python3
"""
Project KD — Multi-Sensor System State Reconstruction & Visualization
Entry point.

Usage:
    python main.py                     # Run with defaults
    python main.py --skip 5            # Process every 5th frame
    python main.py --max-frames 500    # Stop after 500 frames
    python main.py --delay 0.1         # Slower playback
    python main.py --generate          # Force re-generate synthetic data
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Project KD — Multi-Sensor System State Reconstruction & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                   Run with default settings
  python main.py --skip 5          Process every 5th frame (faster)
  python main.py --max-frames 200  Stop after 200 frames
  python main.py --delay 0.02      Faster playback
  python main.py --generate        Force re-generate synthetic data
        """,
    )

    parser.add_argument(
        "--skip", type=int, default=None,
        help="Process every Nth frame (default: 2)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to process (0 = all, default: 0)",
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help="Delay between frames in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Force re-generate synthetic LiDAR data",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to data directory (default: ./data)",
    )

    args = parser.parse_args()

    # Override config if data-dir specified
    from src import config
    if args.data_dir:
        config.DATA_DIR = args.data_dir
        config.VELODYNE_DIR = os.path.join(args.data_dir, "velodyne")
        config.POSES_FILE = os.path.join(args.data_dir, "poses.txt")

    # Force regenerate synthetic data
    if args.generate:
        import shutil
        if os.path.exists(config.VELODYNE_DIR):
            shutil.rmtree(config.VELODYNE_DIR)
            os.makedirs(config.VELODYNE_DIR, exist_ok=True)
        print("  Cleared velodyne directory. Will regenerate synthetic data.")

    # Run pipeline
    from src.pipeline import run
    run(
        frame_skip=args.skip,
        max_frames=args.max_frames,
        playback_delay=args.delay,
    )


if __name__ == "__main__":
    main()
