#!/usr/bin/env python3
"""
Domain Adaptation Testing Script

Test a model trained on one distance group against another distance group.
This validates the model's generalization capability across different trajectory distances.

Usage:
    # Test a Medium-trained model on Large data
    python scripts/domain_adaptation_test.py \
        --source_checkpoint ./checkpoints/CursorTraj/Medium \
        --target_dataset Large

    # Full cross-domain matrix (test all combinations)
    python scripts/domain_adaptation_test.py --cross_matrix
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baseline
from SingularTrajectory import SingularTrajectory
from utils.utils import DotDict
from utils.cursor_dataloader import get_cursor_dataloader
from utils.metrics import compute_batch_ade, compute_batch_fde, AverageMeter


DISTANCE_GROUPS = ["XSmall", "Small", "Medium", "Large", "XLarge"]


def load_model(checkpoint_dir, device='cuda'):
    """Load a trained SingularTrajectory model."""
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


def test_on_dataset(model, hyper_params, target_dataset_dir, device='cuda'):
    """Test a model on a target dataset.

    Args:
        model: Trained SingularTrajectory model
        hyper_params: Model hyperparameters
        target_dataset_dir: Path to target dataset directory
        device: Computation device

    Returns:
        Dictionary with ADE and FDE metrics
    """
    # Create dataloader for target dataset
    loader = get_cursor_dataloader(
        data_dir=target_dataset_dir,
        phase='test',
        obs_len=hyper_params.obs_len,
        pred_len=hyper_params.pred_len,
        batch_size=1
    )

    if len(loader.dataset) == 0:
        return {"ADE": float('nan'), "FDE": float('nan'), "count": 0}

    # Initialize adaptive anchors for target dataset
    dataset = loader.dataset

    # Calculate base anchors using source model's cluster centers
    n_traj = len(dataset)
    k = hyper_params.k
    s = hyper_params.num_samples

    # Use moving trajectory anchors (cursor trajectories are all "moving")
    C_anchor = model.adaptive_anchor_m.C_anchor.detach().cpu()
    dataset.anchor = C_anchor.unsqueeze(0).expand(n_traj, -1, -1).clone()

    # Metrics
    stats_func = {'ADE': compute_batch_ade, 'FDE': compute_batch_fde}
    stats_meter = {x: AverageMeter() for x in stats_func.keys()}

    # Run evaluation
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            obs_traj = batch["obs_traj"].to(device, non_blocking=True)
            pred_traj = batch["pred_traj"].to(device, non_blocking=True)
            adaptive_anchor = batch["anchor"].to(device, non_blocking=True)
            scene_mask = batch["scene_mask"].to(device, non_blocking=True)

            addl_info = {
                "scene_mask": scene_mask,
                "num_samples": hyper_params.num_samples
            }

            output = model(obs_traj, adaptive_anchor, addl_info=addl_info)

            for metric in stats_func.keys():
                value = stats_func[metric](output["recon_traj"], pred_traj)
                stats_meter[metric].extend(value)

    results = {x: stats_meter[x].mean() for x in stats_meter.keys()}
    results["count"] = len(dataset)
    return results


def run_cross_domain_matrix(checkpoint_base, dataset_base, device='cuda'):
    """Run full cross-domain evaluation matrix.

    Tests each trained model on all distance groups.
    """
    results_matrix = {}

    print("\n" + "=" * 80)
    print("Cross-Domain Adaptation Matrix")
    print("=" * 80)

    for source_group in DISTANCE_GROUPS:
        source_checkpoint = os.path.join(checkpoint_base, source_group)

        if not os.path.exists(os.path.join(source_checkpoint, 'model_best.pth')):
            print(f"  [SKIP] No checkpoint for {source_group}")
            continue

        print(f"\nSource: {source_group}")

        try:
            model, hyper_params = load_model(source_checkpoint, device)
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            continue

        results_matrix[source_group] = {}

        for target_group in DISTANCE_GROUPS:
            target_dataset = os.path.join(dataset_base, target_group)

            if not os.path.exists(target_dataset):
                print(f"  -> {target_group}: [NO DATA]")
                continue

            results = test_on_dataset(model, hyper_params, target_dataset, device)
            results_matrix[source_group][target_group] = results

            if results["count"] > 0:
                print(f"  -> {target_group}: ADE={results['ADE']:.4f}, FDE={results['FDE']:.4f} (n={results['count']})")
            else:
                print(f"  -> {target_group}: [NO TEST DATA]")

    # Print summary matrix
    print("\n" + "=" * 80)
    print("Summary Matrix (ADE)")
    print("=" * 80)

    # Header
    header = "Source\\Target".ljust(15)
    for target in DISTANCE_GROUPS:
        header += target.center(12)
    print(header)
    print("-" * 80)

    # Data rows
    for source in DISTANCE_GROUPS:
        if source not in results_matrix:
            continue
        row = source.ljust(15)
        for target in DISTANCE_GROUPS:
            if target in results_matrix[source] and results_matrix[source][target]["count"] > 0:
                ade = results_matrix[source][target]["ADE"]
                row += f"{ade:.4f}".center(12)
            else:
                row += "-".center(12)
        print(row)

    return results_matrix


def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation Testing")
    parser.add_argument("--source_checkpoint", "-s", type=str, default=None,
                        help="Path to source model checkpoint directory")
    parser.add_argument("--target_dataset", "-t", type=str, default=None,
                        help="Target dataset name (e.g., 'Large') or path")
    parser.add_argument("--cross_matrix", action="store_true",
                        help="Run full cross-domain evaluation matrix")
    parser.add_argument("--checkpoint_base", type=str, default="./checkpoints/CursorTraj",
                        help="Base path for checkpoints (for cross_matrix mode)")
    parser.add_argument("--dataset_base", type=str, default="./datasets/cursor",
                        help="Base path for datasets (for cross_matrix mode)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    if args.cross_matrix:
        # Run full cross-domain matrix
        run_cross_domain_matrix(args.checkpoint_base, args.dataset_base, device)
    else:
        # Single source -> target evaluation
        if args.source_checkpoint is None or args.target_dataset is None:
            print("Error: --source_checkpoint and --target_dataset required for single evaluation")
            print("Use --cross_matrix for full matrix evaluation")
            return

        # Resolve target dataset path
        if os.path.isdir(args.target_dataset):
            target_dir = args.target_dataset
        else:
            target_dir = os.path.join(args.dataset_base, args.target_dataset)

        print("=" * 60)
        print("Domain Adaptation Test")
        print("=" * 60)
        print(f"Source checkpoint: {args.source_checkpoint}")
        print(f"Target dataset: {target_dir}")

        # Load model
        print("\nLoading model...")
        model, hyper_params = load_model(args.source_checkpoint, device)

        # Extract source dataset name from checkpoint path
        source_name = os.path.basename(args.source_checkpoint.rstrip('/'))
        target_name = os.path.basename(target_dir.rstrip('/'))

        # Run evaluation
        print(f"\nEvaluating {source_name} model on {target_name} data...")
        results = test_on_dataset(model, hyper_params, target_dir, device)

        # Print results
        print("\n" + "=" * 60)
        print(f"Results: {source_name} -> {target_name}")
        print("=" * 60)
        if results["count"] > 0:
            print(f"  ADE: {results['ADE']:.6f}")
            print(f"  FDE: {results['FDE']:.6f}")
            print(f"  Test samples: {results['count']}")
        else:
            print("  No test samples available")
        print("=" * 60)


if __name__ == "__main__":
    main()
