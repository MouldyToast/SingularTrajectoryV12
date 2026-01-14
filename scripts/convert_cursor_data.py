#!/usr/bin/env python3
"""
Convert cursor trajectory JSON files to SingularTrajectory training format.

Input:  trajectories/trajectory_0001.json, trajectory_0002.json, ...
Output: datasets/cursor/{XSmall,Small,Medium,Large,XLarge}/{train,val,test}.txt

Each output file contains trajectories in format:
    <frame_idx> <trajectory_id> <x> <y>

Trajectories are:
- Categorized by distance group based on actual_distance
- Split into train/val/test (60/20/20)
- Normalized: origin at start point, rotated to face East (0°), scaled to unit length
"""

import os
import json
import glob
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# Distance group definitions
DISTANCE_GROUPS = [
    {"name": "XSmall", "id": 0, "min": 0,   "max": 50},
    {"name": "Small",  "id": 1, "min": 50,  "max": 100},
    {"name": "Medium", "id": 2, "min": 100, "max": 200},
    {"name": "Large",  "id": 3, "min": 200, "max": 400},
    {"name": "XLarge", "id": 4, "min": 400, "max": 900},
]


def get_distance_group(actual_distance):
    """Determine which distance group a trajectory belongs to."""
    for group in DISTANCE_GROUPS:
        if group["min"] <= actual_distance < group["max"]:
            return group["name"]
    # Handle edge case for exactly max value of last group
    if actual_distance >= DISTANCE_GROUPS[-1]["max"]:
        return DISTANCE_GROUPS[-1]["name"]
    return None


def load_trajectory(json_path):
    """Load a single trajectory from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def normalize_trajectory(x_coords, y_coords):
    """
    Normalize trajectory:
    - Origin at start point (0, 0)
    - Rotate so end point is along positive X axis
    - Scale to unit length (distance from start to end = 1.0)

    Returns normalized coordinates and normalization parameters for denormalization.
    """
    x = np.array(x_coords, dtype=np.float64)
    y = np.array(y_coords, dtype=np.float64)

    # Store original start and end for denormalization params
    start_x, start_y = x[0], y[0]
    end_x, end_y = x[-1], y[-1]

    # 1. Translate: origin at start point
    x = x - start_x
    y = y - start_y

    # 2. Calculate rotation angle (angle from start to end)
    dx = end_x - start_x
    dy = end_y - start_y
    angle = math.atan2(dy, dx)

    # 3. Rotate to align with positive X axis (end point on X axis)
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a

    # 4. Scale to unit length
    distance = math.sqrt(dx**2 + dy**2)
    if distance > 0:
        x_norm = x_rot / distance
        y_norm = y_rot / distance
    else:
        # Handle zero-distance case (stationary)
        x_norm = x_rot
        y_norm = y_rot
        distance = 1.0  # Avoid division by zero in denorm

    # Store normalization parameters
    norm_params = {
        "start": (start_x, start_y),
        "angle": angle,
        "scale": distance
    }

    return x_norm.tolist(), y_norm.tolist(), norm_params


def convert_trajectories(input_dir, output_dir, normalize=True, seed=42):
    """
    Convert all JSON trajectory files to training format.

    Args:
        input_dir: Directory containing trajectory_XXXX.json files
        output_dir: Output directory (datasets/cursor/)
        normalize: Whether to normalize trajectories
        seed: Random seed for train/val/test split
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find all trajectory files
    json_pattern = os.path.join(input_dir, "trajectory_*.json")
    json_files = sorted(glob.glob(json_pattern))

    if not json_files:
        print(f"No trajectory files found matching: {json_pattern}")
        return

    print(f"Found {len(json_files)} trajectory files")

    # Group trajectories by distance group
    grouped_trajectories = defaultdict(list)
    skipped = 0

    for json_path in json_files:
        try:
            data = load_trajectory(json_path)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {json_path}: {e}")
            skipped += 1
            continue

        # Get actual distance and determine group
        actual_distance = data.get("actual_distance", 0)
        group_name = get_distance_group(actual_distance)

        if group_name is None:
            print(f"Skipping {json_path}: distance {actual_distance} doesn't fit any group")
            skipped += 1
            continue

        # Extract coordinates
        x_coords = data.get("x", [])
        y_coords = data.get("y", [])

        if len(x_coords) < 2 or len(y_coords) < 2:
            print(f"Skipping {json_path}: insufficient points ({len(x_coords)})")
            skipped += 1
            continue

        if len(x_coords) != len(y_coords):
            print(f"Skipping {json_path}: x/y length mismatch")
            skipped += 1
            continue

        # Normalize if requested
        if normalize:
            x_coords, y_coords, norm_params = normalize_trajectory(x_coords, y_coords)
        else:
            norm_params = None

        # Store trajectory with metadata
        traj_id = os.path.basename(json_path).replace("trajectory_", "").replace(".json", "")
        grouped_trajectories[group_name].append({
            "id": traj_id,
            "x": x_coords,
            "y": y_coords,
            "actual_distance": actual_distance,
            "norm_params": norm_params,
            "source_file": json_path
        })

    print(f"\nGrouped trajectories:")
    for group_name, trajs in sorted(grouped_trajectories.items()):
        print(f"  {group_name}: {len(trajs)} trajectories")
    print(f"  Skipped: {skipped}")

    # Create output directory structure and split data
    for group_name, trajectories in grouped_trajectories.items():
        group_dir = os.path.join(output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        # Shuffle for random split
        random.shuffle(trajectories)

        # Split 60/20/20
        n = len(trajectories)
        n_train = int(n * 0.6)
        n_val = int(n * 0.2)

        splits = {
            "train": trajectories[:n_train],
            "val": trajectories[n_train:n_train + n_val],
            "test": trajectories[n_train + n_val:]
        }

        print(f"\n{group_name} split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        # Write each split to file
        for split_name, split_trajectories in splits.items():
            # Create split directory
            split_dir = os.path.join(group_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)

            output_file = os.path.join(split_dir, f"{group_name}_{split_name}.txt")

            with open(output_file, 'w') as f:
                for traj_idx, traj in enumerate(split_trajectories, start=1):
                    x_coords = traj["x"]
                    y_coords = traj["y"]

                    # Write each point: frame_idx, trajectory_id, x, y
                    for frame_idx, (x, y) in enumerate(zip(x_coords, y_coords)):
                        f.write(f"{frame_idx}\t{traj_idx}\t{x:.6f}\t{y:.6f}\n")

            print(f"  Written: {output_file}")

        # Save normalization parameters for later denormalization
        if normalize:
            params_file = os.path.join(group_dir, "norm_params.json")
            params_data = {
                traj["id"]: traj["norm_params"]
                for split_trajs in splits.values()
                for traj in split_trajs
            }
            with open(params_file, 'w') as f:
                json.dump(params_data, f, indent=2)
            print(f"  Normalization params: {params_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert cursor trajectory JSONs to training format")
    parser.add_argument("--input", "-i", type=str, default="./trajectories",
                        help="Input directory containing trajectory_XXXX.json files")
    parser.add_argument("--output", "-o", type=str, default="./datasets/cursor",
                        help="Output directory for converted data")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip normalization (keep raw coordinates)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test split")

    args = parser.parse_args()

    print("=" * 60)
    print("Cursor Trajectory Data Converter")
    print("=" * 60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Normalize:        {not args.no_normalize}")
    print(f"Random seed:      {args.seed}")
    print("=" * 60)

    convert_trajectories(
        input_dir=args.input,
        output_dir=args.output,
        normalize=not args.no_normalize,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
