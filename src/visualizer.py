"""
Visualization Engine — Phase 2-3
Open3D-based 3D rendering with real-time streaming updates.
"""

import numpy as np
import open3d as o3d
from . import config


class SceneVisualizer:
    """
    Manages all Open3D geometries and the rendering window.
    All geometries are created once and updated in-place.
    """

    def __init__(self):
        # Create visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=config.WINDOW_TITLE,
            width=config.WINDOW_WIDTH,
            height=config.WINDOW_HEIGHT,
        )

        # Set render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array(config.COLOR_BACKGROUND)
        opt.point_size = config.POINT_SIZE
        opt.show_coordinate_frame = False

        # ── Live point cloud (current frame) ──
        self.pcd_live = o3d.geometry.PointCloud()
        self.pcd_live.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        self.pcd_live.colors = o3d.utility.Vector3dVector(np.array([config.COLOR_LIVE]))
        self.vis.add_geometry(self.pcd_live)

        # ── Global map point cloud ──
        self.pcd_map = o3d.geometry.PointCloud()
        self.pcd_map.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
        self.pcd_map.colors = o3d.utility.Vector3dVector(np.array([config.COLOR_MAP]))
        self.vis.add_geometry(self.pcd_map)

        # ── Trajectory line ──
        self.trajectory_line = o3d.geometry.LineSet()
        self.trajectory_line.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
        self.trajectory_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        self.trajectory_line.colors = o3d.utility.Vector3dVector(
            np.array([config.COLOR_TRAJECTORY])
        )
        self.vis.add_geometry(self.trajectory_line)

        # ── Prediction line ──
        self.prediction_line = o3d.geometry.LineSet()
        self.prediction_line.points = o3d.utility.Vector3dVector(np.zeros((2, 3)))
        self.prediction_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        self.prediction_line.colors = o3d.utility.Vector3dVector(
            np.array([config.COLOR_PREDICTION])
        )
        self.vis.add_geometry(self.prediction_line)

        # ── Current position marker ──
        self.position_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        self.position_marker.paint_uniform_color(config.COLOR_TRAJECTORY)
        self.position_marker.compute_vertex_normals()
        self.vis.add_geometry(self.position_marker)

        # Track if camera has been set
        self._camera_set = False
        self._frame_count = 0

    def update_live_cloud(self, points: np.ndarray, colors: np.ndarray = None):
        """Update the live/current frame point cloud."""
        if points is None or len(points) == 0:
            return

        self.pcd_live.points = o3d.utility.Vector3dVector(points)

        if colors is not None:
            self.pcd_live.colors = o3d.utility.Vector3dVector(colors)
        else:
            n = len(points)
            self.pcd_live.colors = o3d.utility.Vector3dVector(
                np.tile(config.COLOR_LIVE, (n, 1))
            )

        self.vis.update_geometry(self.pcd_live)

    def update_map_cloud(self, points: np.ndarray):
        """Update the global map point cloud."""
        if points is None or len(points) == 0:
            return

        n = len(points)
        self.pcd_map.points = o3d.utility.Vector3dVector(points)
        self.pcd_map.colors = o3d.utility.Vector3dVector(
            np.tile(config.COLOR_MAP, (n, 1))
        )
        self.vis.update_geometry(self.pcd_map)

    def update_trajectory(self, positions: np.ndarray):
        """Update trajectory line from array of positions (N, 3)."""
        if positions is None or len(positions) < 2:
            return

        n = len(positions)
        lines = [[i, i + 1] for i in range(n - 1)]

        self.trajectory_line.points = o3d.utility.Vector3dVector(positions)
        self.trajectory_line.lines = o3d.utility.Vector2iVector(np.array(lines))
        self.trajectory_line.colors = o3d.utility.Vector3dVector(
            np.tile(config.COLOR_TRAJECTORY, (len(lines), 1))
        )
        self.vis.update_geometry(self.trajectory_line)

        # Update position marker
        current_pos = positions[-1]
        self.position_marker.translate(current_pos, relative=False)
        self.vis.update_geometry(self.position_marker)

    def update_prediction(self, positions: np.ndarray):
        """Update prediction line (dotted future trajectory)."""
        if positions is None or len(positions) < 2:
            return

        n = len(positions)
        # Create dashed effect: skip every other line segment
        lines = [[i, i + 1] for i in range(0, n - 1, 2)]
        if not lines:
            return

        self.prediction_line.points = o3d.utility.Vector3dVector(positions)
        self.prediction_line.lines = o3d.utility.Vector2iVector(np.array(lines))
        self.prediction_line.colors = o3d.utility.Vector3dVector(
            np.tile(config.COLOR_PREDICTION, (len(lines), 1))
        )
        self.vis.update_geometry(self.prediction_line)

    def setup_camera(self, look_at: np.ndarray = None):
        """Set up camera to a good viewing angle."""
        ctr = self.vis.get_view_control()
        if look_at is not None:
            ctr.set_lookat(look_at)
        ctr.set_zoom(0.15)
        ctr.set_front([0.0, -0.3, -1.0])
        ctr.set_up([0.0, -1.0, 0.2])
        self._camera_set = True

    def follow_camera(self, position: np.ndarray):
        """Make camera follow the current position (subtle)."""
        if self._frame_count % 10 == 0:  # Don't update every frame (too jittery)
            ctr = self.vis.get_view_control()
            ctr.set_lookat(position)

    def tick(self) -> bool:
        """
        Process one render cycle.
        Returns False if the window was closed.
        """
        self._frame_count += 1
        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def run_until_closed(self):
        """Keep the window open until user closes it."""
        self.vis.run()

    def destroy(self):
        """Cleanup."""
        try:
            self.vis.destroy_window()
        except Exception:
            pass
