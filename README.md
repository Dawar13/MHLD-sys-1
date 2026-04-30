# 🧠 Project KD — Multi-Sensor System State Reconstruction & Visualization

A real-time 3D LiDAR visualization system that reconstructs environment state from multi-sensor data, tracks system trajectory, and detects spatial deviations.

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Open3D](https://img.shields.io/badge/Open3D-0.17+-green.svg)

---

## 🎯 What It Does

1. **Loads LiDAR data** — Real KITTI or synthetic point clouds
2. **Streams 3D visualization** — Real-time animated point cloud rendering
3. **Builds a spatial map** — Accumulates past observations into unified world model
4. **Tracks trajectory** — Shows system state evolution over time
5. **Detects deviations** — Highlights mismatch between expected and observed
6. **Predicts future path** — Velocity-based trajectory extrapolation

## 🚀 Quick Start

### Install
```bash
pip install open3d numpy
```

### Run
```bash
python main.py
```

That's it! If no KITTI data is present, synthetic LiDAR data is auto-generated.

### Options
```bash
python main.py --skip 5          # Faster (every 5th frame)
python main.py --max-frames 300  # Shorter demo
python main.py --delay 0.02      # Faster playback
python main.py --generate        # Regenerate synthetic data
```

## 📂 Project Structure

```
S_Rep/
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── data/
│   ├── velodyne/            # Point cloud .bin files
│   └── poses.txt            # Ground truth poses (KITTI format)
└── src/
    ├── config.py            # All tunable parameters
    ├── data_loader.py       # Phase 1: Load .bin + poses
    ├── synthetic_generator.py  # Generate realistic LiDAR data
    ├── visualizer.py        # Phase 2-3: Open3D rendering
    ├── trajectory.py        # Phase 4: State tracking
    ├── map_builder.py       # Phase 5-6: Map + alignment
    ├── deviation.py         # Phase 7-8: Mismatch detection
    ├── prediction.py        # Phase 9: Future prediction
    └── pipeline.py          # Master orchestrator
```

## 🧱 Architecture

```
Data Loader → Representation Layer → State Estimation → Visualization → Analysis
   (bin)        (transform/align)      (trajectory)       (Open3D)      (deviation)
```

## 🔵 Demo Phases

| Phase | What Happens | What You See |
|-------|-------------|--------------|
| 1 | Data loading | Console logs |
| 2 | Raw point cloud | 3D scatter of environment |
| 3 | Streaming | Animated LiDAR playback |
| 4 | Trajectory | Gold line building behind motion |
| 5 | Map building | Dense accumulated environment |
| 6 | Alignment | All frames in unified coordinates |
| 7 | Overlay | Gray map + cyan live points |
| 8 | Deviation | Green (aligned) → Red (deviated) |
| 9 | Prediction | Pink dotted line ahead |

## 🎨 Color Legend

- **Cyan** — Current live frame
- **Gray** — Accumulated map (memory)
- **Gold** — Trajectory path
- **Green** — Aligned with map
- **Red** — Deviating from map
- **Pink** — Predicted future path

## ⚙️ Using Real KITTI Data

1. Download KITTI Odometry velodyne data (sequence 00)
2. Place `.bin` files in `data/velodyne/`
3. Place `poses.txt` in `data/` (already included)
4. Run `python main.py`

## 🔧 Configuration

Edit `src/config.py` to tune:
- Frame skip rate, playback speed
- Voxel sizes for downsampling
- Deviation thresholds
- Colors, window size
- Prediction parameters

## 💡 Key Insight

> *"Before any AI model, you need a correct representation of system behavior. Most industrial systems don't have this layer."*

## 📜 License

MIT
