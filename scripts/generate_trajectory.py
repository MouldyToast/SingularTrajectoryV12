#!/usr/bin/env python3
"""
Generate realistic cursor trajectories between two points.

This script loads a trained SingularTrajectory model and generates
diverse, realistic cursor trajectories given start and end points.

Usage:
    python scripts/generate_trajectory.py --start 100,200 --end 500,400 --num_samples 20

Output:
    JSON file with generated trajectories, or visualization if --plot is specified.
"""

import os
import sys
import json
import math
import argparse
import pickle
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SingularTrajectory.cursor_normalizer import CursorTrajNorm


# Distance group definitions (must match training)
DISTANCE_GROUPS = [
    {"name": "XSmall", "min": 0,   "max": 50},
    {"name": "Small",  "min": 50,  "max": 100},
    {"name": "Medium", "min": 100, "max": 200},
    {"name": "Large",  "min": 200, "max": 400},
    {"name": "XLarge", "min": 400, "max": 900},
]


def get_distance_group(distance):
    """Determine which distance group to use based on pixel distance."""
    for group in DISTANCE_GROUPS:
        if group["min"] <= distance < group["max"]:
            return group["name"]
    # Default to XLarge for very long distances
    return "XLarge"


def load_model(checkpoint_dir, device='cuda'):
    """Load a trained SingularTrajectory model.

    Args:
        checkpoint_dir: Path to checkpoint directory (contains model_best.pth, config.pkl, args.pkl)
        device: Device to load model on

    Returns:
        model: Loaded model
        hyper_params: Model hyperparameters
    """
    import baseline
    from SingularTrajectory import SingularTrajectory
    from utils import DotDict

    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.pkl')
    with open(config_path, 'rb') as f:
        hyper_params = pickle.load(f)

    # Create model
    cfg = DotDict({
        'scheduler': 'ddim',
        'steps': 10,
        'beta_start': 1.e-4,
        'beta_end': 5.e-2,
        'beta_schedule': 'linear',
        'k': hyper_params.k,
        's': hyper_params.num_samples
    })

    PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
    hook_func = DotDict({
        "model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
        "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
        "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook
    })

    predictor_model = PredictorModel(cfg).to(device)
    model = SingularTrajectory(
        baseline_model=predictor_model,
        hook_func=hook_func,
        hyper_params=hyper_params
    ).to(device)

    # Load weights
    model_path = os.path.join(checkpoint_dir, 'model_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, hyper_params


def create_synthetic_observation(start, end, obs_len=3):
    """Create a synthetic observed trajectory for conditioning.

    The observation is a short trajectory from start point moving
    towards the end point. This provides the model with:
    - Starting position
    - Initial direction
    - Initial velocity

    Args:
        start: (x, y) start point
        end: (x, y) end point
        obs_len: Number of observation points

    Returns:
        obs_traj: Tensor of shape (1, obs_len, 2)
    """
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)

    # Direction and distance
    direction = end - start
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        # Handle zero-distance case
        obs = np.tile(start, (obs_len, 1))
    else:
        # Create observation points moving from start towards end
        # Use small steps (e.g., 1-2% of total distance per step)
        unit_dir = direction / distance
        step_size = min(distance * 0.02, 5.0)  # Small steps

        obs = np.zeros((obs_len, 2), dtype=np.float32)
        for i in range(obs_len):
            obs[i] = start + unit_dir * step_size * i

    return torch.from_numpy(obs).unsqueeze(0)  # Shape: (1, obs_len, 2)


def generate_cursor_trajectories(
    model,
    hyper_params,
    start,
    end,
    num_samples=20,
    device='cuda'
):
    """Generate cursor trajectories between start and end points.

    Args:
        model: Trained SingularTrajectory model
        hyper_params: Model hyperparameters
        start: (x, y) start point in pixels
        end: (x, y) end point in pixels
        num_samples: Number of trajectory samples to generate
        device: Computation device

    Returns:
        trajectories: List of numpy arrays, each of shape (seq_len, 2)
    """
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)

    # Calculate distance for potential distance-group specific handling
    distance = np.linalg.norm(end - start)

    # Create normalizer
    normalizer = CursorTrajNorm(ori=True, rot=True, sca=True)

    # Create synthetic observation
    obs_traj = create_synthetic_observation(start, end, hyper_params.obs_len)
    obs_traj = obs_traj.to(device)

    # Calculate normalization parameters from start/end
    start_tensor = torch.from_numpy(start).unsqueeze(0).to(device)  # (1, 2)
    end_tensor = torch.from_numpy(end).unsqueeze(0).to(device)  # (1, 2)
    normalizer.calculate_params_from_endpoints(start_tensor, end_tensor)

    # Normalize observation
    obs_traj_norm = normalizer.normalize(obs_traj)

    # Get anchor from model's pre-trained cluster centers
    # Anchors are in Singular space with shape (k, s) where k=num_components, s=num_samples
    # Model expects adaptive_anchor of shape (n_traj, k, s)
    k = hyper_params.k
    s = hyper_params.num_samples

    # Use the moving trajectory anchors (cursor trajectories are all "moving")
    C_anchor = model.adaptive_anchor_m.C_anchor.detach()  # (k, s)
    anchor = C_anchor.unsqueeze(0)  # (1, k, s)

    # Create scene mask (single trajectory)
    scene_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

    # Additional info
    addl_info = {
        "scene_mask": scene_mask,
        "num_samples": s  # Use model's num_samples
    }

    # Generate trajectories
    with torch.no_grad():
        output = model(obs_traj_norm, anchor, addl_info=addl_info)

    # Get generated trajectories
    # Output shape: (num_samples, 1, pred_len, 2) or (1, pred_len, 2)
    if "recon_traj" in output:
        pred_traj_norm = output["recon_traj"]
    else:
        # Fallback
        pred_traj_norm = anchor.unsqueeze(0).expand(num_samples, -1, -1, -1)

    # Handle different output shapes
    if pred_traj_norm.dim() == 3:
        pred_traj_norm = pred_traj_norm.unsqueeze(0)

    # Denormalize trajectories
    trajectories = []
    for i in range(pred_traj_norm.shape[0]):
        traj_norm = pred_traj_norm[i]  # (1, pred_len, 2) or (pred_len, 2)
        if traj_norm.dim() == 2:
            traj_norm = traj_norm.unsqueeze(0)

        # Denormalize
        traj = normalizer.denormalize(traj_norm)
        traj = traj.squeeze(0).cpu().numpy()

        # Prepend observation
        obs = obs_traj.squeeze(0).cpu().numpy()
        full_traj = np.vstack([obs, traj])

        trajectories.append(full_traj)

    return trajectories


def interpolate_trajectory(trajectory, num_points):
    """Interpolate trajectory to have exactly num_points.

    Useful for creating smooth, evenly-spaced trajectories.
    """
    from scipy import interpolate

    t_orig = np.linspace(0, 1, len(trajectory))
    t_new = np.linspace(0, 1, num_points)

    fx = interpolate.interp1d(t_orig, trajectory[:, 0], kind='cubic')
    fy = interpolate.interp1d(t_orig, trajectory[:, 1], kind='cubic')

    return np.column_stack([fx(t_new), fy(t_new)])


def save_trajectories(trajectories, output_path, start, end):
    """Save generated trajectories to JSON file."""
    data = {
        "start": list(start),
        "end": list(end),
        "num_trajectories": len(trajectories),
        "trajectories": [
            {
                "x": traj[:, 0].tolist(),
                "y": traj[:, 1].tolist(),
                "length": len(traj)
            }
            for traj in trajectories
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def plot_trajectories(trajectories, start, end, output_path=None):
    """Visualize generated trajectories."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectories
    for i, traj in enumerate(trajectories):
        alpha = 0.5 if len(trajectories) > 5 else 0.8
        ax.plot(traj[:, 0], traj[:, 1], '-', alpha=alpha, linewidth=1.5, label=f'Traj {i+1}' if i < 5 else None)

    # Mark start and end
    ax.scatter([start[0]], [start[1]], c='green', s=100, zorder=5, marker='o', label='Start')
    ax.scatter([end[0]], [end[1]], c='red', s=100, zorder=5, marker='x', label='End')

    # Draw direct line
    ax.plot([start[0], end[0]], [start[1], end[1]], 'k--', alpha=0.3, linewidth=2, label='Direct')

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Generated Cursor Trajectories ({len(trajectories)} samples)')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Screen coordinates have Y increasing downward
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate cursor trajectories")
    parser.add_argument("--checkpoint", "-c", type=str, default=None,
                        help="Path to model checkpoint directory (auto-selects based on distance if not specified)")
    parser.add_argument("--checkpoint_base", type=str, default="./checkpoints/CursorTraj",
                        help="Base path for checkpoints when auto-selecting")
    parser.add_argument("--start", "-s", type=str, required=True,
                        help="Start point as 'x,y' (e.g., '100,200')")
    parser.add_argument("--end", "-e", type=str, required=True,
                        help="End point as 'x,y' (e.g., '500,400')")
    parser.add_argument("--num_samples", "-n", type=int, default=20,
                        help="Number of trajectory samples to generate")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--plot", "-p", type=str, default=None,
                        help="Output plot image path")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Parse coordinates
    start = tuple(map(float, args.start.split(',')))
    end = tuple(map(float, args.end.split(',')))

    # Calculate distance and determine group
    distance = np.linalg.norm(np.array(end) - np.array(start))
    distance_group = get_distance_group(distance)

    # Auto-select checkpoint based on distance group if not specified
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.checkpoint_base, distance_group)

    print("=" * 60)
    print("Cursor Trajectory Generator")
    print("=" * 60)
    print(f"Start point: {start}")
    print(f"End point: {end}")
    print(f"Distance: {distance:.1f} pixels")
    print(f"Distance group: {distance_group}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Num samples: {args.num_samples}")
    print("=" * 60)

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Load model
    print("\nLoading model...")
    model, hyper_params = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    # Generate trajectories
    print("\nGenerating trajectories...")
    trajectories = generate_cursor_trajectories(
        model=model,
        hyper_params=hyper_params,
        start=start,
        end=end,
        num_samples=args.num_samples,
        device=device
    )
    print(f"Generated {len(trajectories)} trajectories")

    # Save output
    if args.output:
        save_trajectories(trajectories, args.output, start, end)

    # Plot if requested
    if args.plot:
        plot_trajectories(trajectories, start, end, args.plot)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    lengths = [len(t) for t in trajectories]
    print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")

    # Calculate path efficiency (actual distance / ideal distance)
    ideal_dist = np.linalg.norm(np.array(end) - np.array(start))
    actual_dists = [np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)) for t in trajectories]
    efficiencies = [ideal_dist / max(d, 1e-6) for d in actual_dists]
    print(f"  Path efficiency: min={min(efficiencies):.2%}, max={max(efficiencies):.2%}, avg={np.mean(efficiencies):.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
