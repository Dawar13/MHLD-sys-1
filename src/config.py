"""
Central configuration for Project KD.
All tunable parameters in one place.
"""

import os

# --- Paths -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VELODYNE_DIR = os.path.join(DATA_DIR, "velodyne")
POSES_FILE = os.path.join(DATA_DIR, "poses.txt")

# --- Playback ----------------------------------------------------
FRAME_SKIP = 2            # Process every Nth frame
PLAYBACK_DELAY = 0.05     # Seconds between frames
MAX_FRAMES = 0            # 0 = all frames

# --- Point Cloud Processing -------------------------------------
VOXEL_SIZE = 0.4           # Downsample current frame (meters)
MAP_VOXEL_SIZE = 0.8       # Downsample global map (meters)
MAX_MAP_POINTS = 2_000_000 # Cap map size to prevent slowdown
POINT_SIZE = 1.5           # Rendering point size

# --- Deviation Detection ----------------------------------------
DEVIATION_THRESHOLD = 0.5  # Distance threshold (meters)
DEVIATION_MAX = 3.0        # Max distance for color scale

# --- Prediction --------------------------------------------------
PREDICTION_STEPS = 30      # Number of future steps to predict
VELOCITY_WINDOW = 10       # Frames to average velocity over

# --- Synthetic Data ----------------------------------------------
SYNTHETIC_POINTS_PER_FRAME = 60_000  # Points per synthetic frame
SYNTHETIC_RANGE = 40.0               # Max range in meters

# --- Colors (RGB, 0-1 range) ------------------------------------
COLOR_LIVE = [0.0, 0.9, 1.0]        # Cyan -- current frame
COLOR_MAP = [0.45, 0.45, 0.50]      # Gray -- accumulated map
COLOR_TRAJECTORY = [1.0, 0.85, 0.0] # Gold -- trajectory line
COLOR_PREDICTION = [1.0, 0.3, 0.6]  # Pink -- predicted path
COLOR_DEVIATION_LOW = [0.1, 1.0, 0.3]   # Green -- aligned
COLOR_DEVIATION_HIGH = [1.0, 0.1, 0.1]  # Red -- deviated
COLOR_BACKGROUND = [0.05, 0.05, 0.08]   # Near-black background

# --- Window ------------------------------------------------------
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
WINDOW_TITLE = "Project KD -- Multi-Sensor State Reconstruction"

# --- Demo Narration ----------------------------------------------
NARRATION = {
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
