"""
Synthetic LiDAR Generator
Generates realistic street-like point clouds using real poses.
Creates coherent urban environments with proper structure.
"""

import numpy as np
from . import config


def _generate_ground(range_m: float, density: float = 0.3) -> np.ndarray:
    """Generate a dense, flat road + sidewalk surface."""
    # Road surface (dense)
    n_road = int((2 * range_m) * (8) * density * 100)
    road_x = np.random.uniform(-range_m, range_m, n_road)
    road_y = np.random.uniform(-4, 4, n_road)  # Road width ~8m
    road_z = np.random.normal(-1.73, 0.02, n_road)
    road = np.column_stack([road_x, road_y, road_z])
    
    # Sidewalks
    n_side = int(n_road * 0.3)
    for side_y in [(-6, -4), (4, 6)]:
        sx = np.random.uniform(-range_m, range_m, n_side)
        sy = np.random.uniform(side_y[0], side_y[1], n_side)
        sz = np.random.normal(-1.60, 0.02, n_side)  # Slightly elevated
        road = np.vstack([road, np.column_stack([sx, sy, sz])])
    
    return road


def _generate_building_wall(x_start, x_end, y_pos, height, depth=0.3, n_pts=2000):
    """Generate a single building wall as a dense flat surface."""
    n = n_pts
    x = np.random.uniform(x_start, x_end, n)
    y = np.random.normal(y_pos, depth * 0.3, n)
    z = np.random.uniform(-1.6, height, n)
    return np.column_stack([x, y, z])


def _generate_buildings(range_m: float, frame_idx: int) -> np.ndarray:
    """Generate coherent building facades on both sides of the road."""
    points = []
    np.random.seed(1000 + (frame_idx // 50))  # Buildings change slowly
    
    for side in [-1, 1]:
        base_y = side * np.random.uniform(7, 12)
        
        # Generate 4-8 building segments
        n_buildings = np.random.randint(4, 9)
        x_cursor = -range_m
        
        for _ in range(n_buildings):
            width = np.random.uniform(5, 15)
            height = np.random.uniform(3, 10)
            gap = np.random.uniform(0.5, 3)
            
            x_end = min(x_cursor + width, range_m)
            
            if x_cursor < range_m:
                # Front face
                wall = _generate_building_wall(
                    x_cursor, x_end, base_y, height, n_pts=1500
                )
                points.append(wall)
                
                # Side edges (returns of the building)
                for edge_x in [x_cursor, x_end]:
                    n_edge = 200
                    ex = np.random.normal(edge_x, 0.05, n_edge)
                    ey = np.random.uniform(base_y - 0.5, base_y + 0.5, n_edge)
                    ez = np.random.uniform(-1.6, height, n_edge)
                    points.append(np.column_stack([ex, ey, ez]))
                
                # Roofline
                n_roof = 300
                rx = np.random.uniform(x_cursor, x_end, n_roof)
                ry = np.random.normal(base_y, 0.3, n_roof)
                rz = np.random.normal(height, 0.05, n_roof)
                points.append(np.column_stack([rx, ry, rz]))
            
            x_cursor = x_end + gap
    
    # Reset random state
    np.random.seed(None)
    
    if points:
        return np.vstack(points)
    return np.zeros((0, 3))


def _generate_parked_cars(range_m: float, frame_idx: int) -> np.ndarray:
    """Generate parked car-like box shapes along the road."""
    points = []
    np.random.seed(2000 + (frame_idx // 30))
    
    n_cars = np.random.randint(3, 8)
    for _ in range(n_cars):
        cx = np.random.uniform(-range_m * 0.7, range_m * 0.7)
        cy = np.random.choice([-1, 1]) * np.random.uniform(3.5, 5.5)
        
        # Car dimensions: ~4.5m x 1.8m x 1.5m
        car_l, car_w, car_h = 4.5, 1.8, 1.5
        n_car = 400
        
        # Top surface
        n_top = n_car // 3
        tx = cx + np.random.uniform(-car_l/2, car_l/2, n_top)
        ty = cy + np.random.uniform(-car_w/2, car_w/2, n_top)
        tz = np.random.normal(-1.73 + car_h, 0.03, n_top)
        points.append(np.column_stack([tx, ty, tz]))
        
        # Side surfaces
        for sy in [-car_w/2, car_w/2]:
            n_s = n_car // 4
            sx = cx + np.random.uniform(-car_l/2, car_l/2, n_s)
            ssy = cy + np.random.normal(sy, 0.03, n_s)
            sz = np.random.uniform(-1.73, -1.73 + car_h, n_s)
            points.append(np.column_stack([sx, ssy, sz]))
        
        # Front/back
        for fx in [-car_l/2, car_l/2]:
            n_f = n_car // 6
            fsx = cx + np.random.normal(fx, 0.03, n_f)
            fsy = cy + np.random.uniform(-car_w/2, car_w/2, n_f)
            fsz = np.random.uniform(-1.73, -1.73 + car_h, n_f)
            points.append(np.column_stack([fsx, fsy, fsz]))
    
    np.random.seed(None)
    
    if points:
        return np.vstack(points)
    return np.zeros((0, 3))


def _generate_poles_and_signs(range_m: float, frame_idx: int) -> np.ndarray:
    """Generate street lights, poles, and signs."""
    points = []
    np.random.seed(3000 + (frame_idx // 40))
    
    n_poles = np.random.randint(4, 10)
    for _ in range(n_poles):
        px = np.random.uniform(-range_m * 0.8, range_m * 0.8)
        py = np.random.choice([-1, 1]) * np.random.uniform(4.5, 7)
        pole_height = np.random.uniform(3, 6)
        
        # Pole shaft
        n_shaft = 80
        z = np.linspace(-1.73, pole_height, n_shaft)
        x = np.random.normal(px, 0.04, n_shaft)
        y = np.random.normal(py, 0.04, n_shaft)
        points.append(np.column_stack([x, y, z]))
        
        # Light fixture / sign at top
        n_top = 30
        tx = np.random.normal(px, 0.3, n_top)
        ty = np.random.normal(py, 0.15, n_top)
        tz = np.random.normal(pole_height, 0.1, n_top)
        points.append(np.column_stack([tx, ty, tz]))
    
    np.random.seed(None)
    
    if points:
        return np.vstack(points)
    return np.zeros((0, 3))


def _generate_trees(range_m: float, frame_idx: int) -> np.ndarray:
    """Generate tree shapes with trunk + spherical canopy."""
    points = []
    np.random.seed(4000 + (frame_idx // 60))
    
    n_trees = np.random.randint(3, 8)
    for _ in range(n_trees):
        tx = np.random.uniform(-range_m * 0.6, range_m * 0.6)
        ty = np.random.choice([-1, 1]) * np.random.uniform(6, 11)
        
        # Trunk
        n_trunk = 60
        trunk_z = np.linspace(-1.73, 2.5, n_trunk)
        trunk_x = np.random.normal(tx, 0.08, n_trunk)
        trunk_y = np.random.normal(ty, 0.08, n_trunk)
        points.append(np.column_stack([trunk_x, trunk_y, trunk_z]))
        
        # Canopy (dense sphere)
        n_canopy = 500
        theta = np.random.uniform(0, 2 * np.pi, n_canopy)
        phi = np.random.uniform(0, np.pi, n_canopy)
        r = np.random.uniform(0, 2.0, n_canopy) ** 0.5 * 2.0  # Square root for uniform density
        cx = tx + r * np.sin(phi) * np.cos(theta)
        cy = ty + r * np.sin(phi) * np.sin(theta)
        cz = 4.0 + r * np.cos(phi) * 0.7  # Slightly squished
        points.append(np.column_stack([cx, cy, cz]))
    
    np.random.seed(None)
    
    if points:
        return np.vstack(points)
    return np.zeros((0, 3))


def _apply_lidar_properties(points: np.ndarray, max_range: float) -> np.ndarray:
    """
    Simulate LiDAR scanning properties:
    - Range limit
    - Distance-based density falloff
    - Sensor noise
    """
    distances = np.linalg.norm(points[:, :2], axis=1)  # Horizontal distance
    
    # Range filter
    mask = distances < max_range
    points = points[mask]
    distances = distances[mask]
    
    if len(points) == 0:
        return np.zeros((1, 3))
    
    # Distance-based density falloff (closer = denser, like real LiDAR)
    keep_prob = np.clip(1.0 - (distances / max_range) ** 2, 0.15, 1.0)
    keep_mask = np.random.random(len(points)) < keep_prob
    points = points[keep_mask]
    
    # Sensor noise (increases with distance)
    if len(points) > 0:
        d = np.linalg.norm(points[:, :2], axis=1)
        noise_scale = 0.01 + 0.005 * d / max_range
        noise = np.random.normal(0, 1, points.shape) * noise_scale[:, np.newaxis]
        points += noise
    
    return points


def generate_single_frame(frame_idx: int, seed: int = None) -> np.ndarray:
    """
    Generate a single synthetic LiDAR frame.
    Returns (N, 3) point cloud in sensor coordinates.
    """
    if seed is not None:
        np.random.seed(seed + frame_idx * 7)
    
    range_m = config.SYNTHETIC_RANGE
    
    # Generate scene components
    ground = _generate_ground(range_m)
    buildings = _generate_buildings(range_m, frame_idx)
    cars = _generate_parked_cars(range_m, frame_idx)
    poles = _generate_poles_and_signs(range_m, frame_idx)
    trees = _generate_trees(range_m, frame_idx)
    
    # Combine all
    all_points = np.vstack([ground, buildings, cars, poles, trees])
    
    # Apply LiDAR simulation
    all_points = _apply_lidar_properties(all_points, range_m)
    
    # Reset random state
    if seed is not None:
        np.random.seed(None)
    
    return all_points.astype(np.float32)


def generate_dataset(poses: np.ndarray, output_dir: str, frame_skip: int = 1):
    """
    Generate a full synthetic dataset matching the number of poses.
    Saves .bin files in KITTI format.
    """
    import os
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
