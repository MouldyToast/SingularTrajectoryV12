#!/usr/bin/env python3
"""
Generate vector fields from cursor trajectory data.

Unlike pedestrian scenarios where vector fields come from scene images (walkable areas),
cursor vector fields are derived FROM the trajectory data itself, showing:
- Typical flow patterns (where cursors tend to go)
- Density of trajectories (high traffic vs low traffic areas)
- Average movement direction at each location

Input:  datasets/cursor/{group}/{train,val,test}.txt
Output: datasets/cursor/{group}/vector_field.npy
"""

import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# Distance groups
DISTANCE_GROUPS = ["XSmall", "Small", "Medium", "Large", "XLarge"]

# Vector field grid resolution
# Since trajectories are normalized to ~[0,1] x [-0.5, 0.5], we use a grid that covers this space
GRID_RESOLUTION = 64  # 64x64 grid
GRID_MIN_X, GRID_MAX_X = -0.2, 1.2  # Extend slightly beyond [0,1] to capture overshoots
GRID_MIN_Y, GRID_MAX_Y = -0.6, 0.6  # Symmetric around Y=0


def load_trajectories(data_dir, group_name):
    """
    Load all trajectories for a distance group (train + val + test).

    Expects flat structure: data_dir/group_name/{train,val,test}.txt

    Returns list of trajectories, each as numpy array of shape (n_points, 2)
    """
    trajectories = []
    group_dir = os.path.join(data_dir, group_name)

    for split in ["train", "val", "test"]:
        # Look for split.txt directly in group directory
        filepath = os.path.join(group_dir, f"{split}.txt")
        if os.path.exists(filepath):
            trajectories.extend(parse_trajectory_file(filepath))

    return trajectories


def parse_trajectory_file(filepath):
    """
    Parse a trajectory file in format: frame_idx, trajectory_id, x, y

    Returns list of trajectories, each as numpy array of shape (n_points, 2)
    """
    trajectories_dict = defaultdict(list)

    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            frame_idx, traj_id, x, y = parts
            traj_id = int(traj_id)
            trajectories_dict[traj_id].append((float(x), float(y)))

    # Convert to numpy arrays
    trajectories = []
    for traj_id in sorted(trajectories_dict.keys()):
        points = np.array(trajectories_dict[traj_id])
        if len(points) >= 2:
            trajectories.append(points)

    return trajectories


def compute_velocity_vectors(trajectory):
    """
    Compute velocity vectors for each point in trajectory.

    Returns array of shape (n_points-1, 4): [x, y, vx, vy]
    where (x,y) is the midpoint and (vx,vy) is the velocity direction.
    """
    if len(trajectory) < 2:
        return np.array([])

    # Compute differences
    diff = trajectory[1:] - trajectory[:-1]

    # Midpoints between consecutive points
    midpoints = (trajectory[1:] + trajectory[:-1]) / 2

    # Normalize velocity to unit vectors
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    velocities = diff / norms

    return np.hstack([midpoints, velocities])


def create_vector_field(trajectories, grid_resolution=GRID_RESOLUTION):
    """
    Create a vector field from trajectory data.

    The vector field represents:
    - Average movement direction at each grid cell
    - Magnitude represents confidence (based on sample count)

    Returns:
        vector_field: numpy array of shape (grid_h, grid_w, 2)
        density_field: numpy array of shape (grid_h, grid_w) - count of samples per cell
    """
    grid_h = grid_w = grid_resolution

    # Accumulate velocity vectors per grid cell
    velocity_sum = np.zeros((grid_h, grid_w, 2))
    sample_count = np.zeros((grid_h, grid_w))

    # Grid cell size
    cell_w = (GRID_MAX_X - GRID_MIN_X) / grid_w
    cell_h = (GRID_MAX_Y - GRID_MIN_Y) / grid_h

    for traj in trajectories:
        vectors = compute_velocity_vectors(traj)
        if len(vectors) == 0:
            continue

        for x, y, vx, vy in vectors:
            # Convert to grid coordinates
            grid_x = int((x - GRID_MIN_X) / cell_w)
            grid_y = int((y - GRID_MIN_Y) / cell_h)

            # Check bounds
            if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                velocity_sum[grid_y, grid_x, 0] += vx
                velocity_sum[grid_y, grid_x, 1] += vy
                sample_count[grid_y, grid_x] += 1

    # Average velocities
    with np.errstate(divide='ignore', invalid='ignore'):
        vector_field = velocity_sum / sample_count[:, :, np.newaxis]
        vector_field = np.nan_to_num(vector_field, nan=0.0)

    # Normalize to unit vectors where we have data
    norms = np.linalg.norm(vector_field, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vector_field = vector_field / norms

    # Zero out cells with no data
    vector_field[sample_count == 0] = 0

    return vector_field, sample_count


def interpolate_empty_cells(vector_field, density_field, iterations=3):
    """
    Fill in empty cells by interpolating from neighbors.

    This helps create smoother vector fields even in sparse areas.
    """
    vf = vector_field.copy()
    df = density_field.copy()

    for _ in range(iterations):
        # Find empty cells
        empty_mask = df == 0

        # For each empty cell, average neighbors
        for y in range(vf.shape[0]):
            for x in range(vf.shape[1]):
                if not empty_mask[y, x]:
                    continue

                # Gather neighbor vectors
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < vf.shape[0] and 0 <= nx < vf.shape[1]:
                            if df[ny, nx] > 0:
                                neighbors.append(vf[ny, nx])

                if neighbors:
                    avg = np.mean(neighbors, axis=0)
                    norm = np.linalg.norm(avg)
                    if norm > 1e-8:
                        vf[y, x] = avg / norm
                        df[y, x] = 0.1  # Mark as interpolated

    return vf, df


def generate_vector_fields(data_dir, output_dir=None, visualize=False):
    """
    Generate vector fields for all distance groups.

    Args:
        data_dir: Directory containing cursor data (datasets/cursor/)
        output_dir: Output directory (defaults to data_dir)
        visualize: Whether to save visualization images
    """
    if output_dir is None:
        output_dir = data_dir

    for group_name in DISTANCE_GROUPS:
        group_dir = os.path.join(data_dir, group_name)
        if not os.path.exists(group_dir):
            print(f"Skipping {group_name}: directory not found")
            continue

        print(f"\nProcessing {group_name}...")

        # Load trajectories
        trajectories = load_trajectories(data_dir, group_name)
        print(f"  Loaded {len(trajectories)} trajectories")

        if len(trajectories) == 0:
            print(f"  No trajectories found, skipping")
            continue

        # Compute vector field
        vector_field, density_field = create_vector_field(trajectories)
        print(f"  Vector field shape: {vector_field.shape}")
        print(f"  Grid cells with data: {np.sum(density_field > 0)} / {density_field.size}")

        # Interpolate sparse areas
        vector_field_interp, density_interp = interpolate_empty_cells(vector_field, density_field)
        print(f"  After interpolation: {np.sum(density_interp > 0)} / {density_interp.size}")

        # Save vector field inside the group directory
        output_file = os.path.join(group_dir, "vector_field.npy")
        np.save(output_file, vector_field_interp)
        print(f"  Saved: {output_file}")

        # Also save density field for analysis
        density_file = os.path.join(group_dir, "density.npy")
        np.save(density_file, density_interp)

        # Save visualization if requested
        if visualize:
            save_visualization(vector_field_interp, density_interp, group_name, group_dir)


def save_visualization(vector_field, density_field, group_name, output_dir):
    """Save a visualization of the vector field."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Matplotlib not available, skipping visualization")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Density plot
    ax = axes[0]
    im = ax.imshow(density_field, origin='lower', cmap='hot')
    ax.set_title(f'{group_name} - Trajectory Density')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    plt.colorbar(im, ax=ax, label='Sample count')

    # Vector field plot
    ax = axes[1]
    Y, X = np.mgrid[0:vector_field.shape[0], 0:vector_field.shape[1]]
    U = vector_field[:, :, 0]
    V = vector_field[:, :, 1]

    # Subsample for clearer visualization
    step = max(1, vector_field.shape[0] // 16)
    ax.quiver(X[::step, ::step], Y[::step, ::step],
              U[::step, ::step], V[::step, ::step],
              scale=20, alpha=0.7)
    ax.set_title(f'{group_name} - Flow Direction')
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    ax.set_aspect('equal')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "vector_field_visualization.png")
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Visualization: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate vector fields from cursor trajectory data")
    parser.add_argument("--data-dir", "-d", type=str, default="./datasets/cursor",
                        help="Directory containing converted cursor data")
    parser.add_argument("--visualize", "-v", action="store_true",
                        help="Generate visualization images")
    parser.add_argument("--resolution", "-r", type=int, default=GRID_RESOLUTION,
                        help=f"Grid resolution (default: {GRID_RESOLUTION})")

    args = parser.parse_args()

    global GRID_RESOLUTION
    GRID_RESOLUTION = args.resolution

    print("=" * 60)
    print("Cursor Vector Field Generator")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Grid resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}")
    print(f"Visualize: {args.visualize}")
    print("=" * 60)

    generate_vector_fields(args.data_dir, visualize=args.visualize)

    print("\n" + "=" * 60)
    print("Vector field generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
