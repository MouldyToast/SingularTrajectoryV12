"""
Cursor Trajectory DataLoader

Simplified dataloader for cursor trajectory data. Key differences from pedestrian dataloader:
- No homography transformation needed
- Vector fields are derived from trajectory data (optional)
- Each trajectory is independent (no multi-agent scene masks)
- Variable trajectory lengths handled via padding/truncation
"""

import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from collections import defaultdict


def get_cursor_dataloader(data_dir, phase, obs_len, pred_len, batch_size, skip=1, max_traj_len=None):
    """Get dataloader for cursor trajectory data.

    Args:
        data_dir: Path to the dataset directory (e.g., ./datasets/cursor/Medium/)
        phase: One of 'train', 'val', 'test'
        obs_len: Number of observed time steps
        pred_len: Number of predicted time steps
        batch_size: Batch size
        skip: Frame skip (default 1)
        max_traj_len: Maximum trajectory length (for padding). If None, uses obs_len + pred_len.

    Returns:
        DataLoader for the specified phase
    """
    assert phase in ['train', 'val', 'test']

    data_path = os.path.join(data_dir, phase)
    shuffle = phase == 'train'
    drop_last = phase == 'train'

    if max_traj_len is None:
        max_traj_len = obs_len + pred_len

    dataset = CursorTrajectoryDataset(
        data_dir=data_path,
        obs_len=obs_len,
        pred_len=pred_len,
        skip=skip,
        max_traj_len=max_traj_len
    )

    # Use custom batch sampler for variable-length handling
    if batch_size > 1:
        sampler = CursorBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        loader = DataLoader(dataset, collate_fn=cursor_collate_fn, batch_sampler=sampler, pin_memory=True)
    else:
        loader = DataLoader(dataset, collate_fn=cursor_collate_fn, shuffle=shuffle, pin_memory=True)

    return loader


def cursor_collate_fn(data):
    """Collate function for cursor trajectory batches.

    Args:
        data: List of dictionaries from dataset __getitem__

    Returns:
        Collated batch dictionary
    """
    data_collated = {}
    for k in data[0].keys():
        data_collated[k] = [d[k] for d in data]

    # Calculate sequence boundaries
    _len = [len(seq) for seq in data_collated["obs_traj"]]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    seq_start_end = torch.LongTensor(seq_start_end)

    # Scene mask (for compatibility - cursor trajectories are independent)
    total_trajs = sum(_len)
    scene_mask = torch.zeros(total_trajs, total_trajs, dtype=torch.bool)
    for idx, (start, end) in enumerate(seq_start_end):
        scene_mask[start:end, start:end] = 1

    # Concatenate tensors
    data_collated["obs_traj"] = torch.cat(data_collated["obs_traj"], dim=0)
    data_collated["pred_traj"] = torch.cat(data_collated["pred_traj"], dim=0)
    data_collated["full_traj"] = torch.cat(data_collated["full_traj"], dim=0)
    data_collated["loss_mask"] = torch.cat(data_collated["loss_mask"], dim=0)
    data_collated["traj_len"] = torch.cat(data_collated["traj_len"], dim=0)
    data_collated["scene_mask"] = scene_mask
    data_collated["seq_start_end"] = seq_start_end

    # Handle anchor if present
    if data_collated["anchor"][0] is not None:
        data_collated["anchor"] = torch.cat(data_collated["anchor"], dim=0)
    else:
        data_collated["anchor"] = None

    # Non-linear ped (for compatibility)
    data_collated["non_linear_ped"] = torch.cat(data_collated["non_linear_ped"], dim=0)

    # Frame IDs
    data_collated["frame"] = torch.cat(data_collated["frame"], dim=0)

    # Scene IDs
    data_collated["scene_id"] = np.concatenate(data_collated["scene_id"], axis=0)

    return data_collated


class CursorBatchSampler(Sampler):
    """Batch sampler for cursor trajectories.

    Groups trajectories into batches based on count rather than
    total pedestrians (since each cursor trajectory is independent).
    """

    def __init__(self, data_source, batch_size=64, shuffle=False, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = len(self.data_source)

        if self.shuffle:
            indices = torch.randperm(n).tolist()
        else:
            indices = list(range(n))

        # Simple batching by count
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size


def read_trajectory_file(path, delim='\t'):
    """Read trajectory file and return grouped trajectories.

    Args:
        path: Path to trajectory file
        delim: Delimiter (default tab)

    Returns:
        Dictionary mapping trajectory_id to list of (frame, x, y) tuples
    """
    trajectories = defaultdict(list)

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delim)
            if len(parts) != 4:
                continue
            frame_idx, traj_id, x, y = parts
            trajectories[int(traj_id)].append((int(float(frame_idx)), float(x), float(y)))

    return trajectories


def poly_fit(traj, traj_len, threshold):
    """Determine if trajectory is non-linear.

    Args:
        traj: numpy array of shape (2, traj_len)
        traj_len: Length of trajectory
        threshold: Threshold for non-linearity

    Returns:
        1.0 if non-linear, 0.0 if linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if len(res_x) == 0:
        res_x = np.array([0])
    if len(res_y) == 0:
        res_y = np.array([0])
    if res_x[0] + res_y[0] >= threshold:
        return 1.0
    return 0.0


class CursorTrajectoryDataset(Dataset):
    """Dataset for cursor trajectory data.

    Handles variable-length trajectories with padding/truncation.
    """

    def __init__(self, data_dir, obs_len=3, pred_len=25, skip=1, max_traj_len=None,
                 threshold=0.02, delim='\t'):
        """
        Args:
            data_dir: Directory containing trajectory files
            obs_len: Number of observed time steps
            pred_len: Number of predicted time steps
            skip: Frame skip
            max_traj_len: Maximum trajectory length for padding
            threshold: Threshold for non-linear detection
            delim: File delimiter
        """
        super().__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = obs_len + pred_len
        self.max_traj_len = max_traj_len if max_traj_len else self.seq_len
        self.delim = delim
        self.threshold = threshold

        # Storage
        self.trajectories = []
        self.traj_lens = []
        self.non_linear = []
        self.scene_ids = []

        # Get scene name from path
        parent_dir = os.path.dirname(data_dir.rstrip('/'))
        self.scene_name = os.path.basename(parent_dir)

        # Load vector field if available
        vectorfield_dir = os.path.join(os.path.dirname(parent_dir), "vectorfield")
        vf_path = os.path.join(vectorfield_dir, f"cursor_{self.scene_name}_vector_field.npy")
        if os.path.exists(vf_path):
            self.vector_field = np.load(vf_path)
        else:
            self.vector_field = None

        # Load all trajectory files
        if os.path.exists(data_dir):
            all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])
            all_files = [os.path.join(data_dir, f) for f in all_files]

            for path in all_files:
                self._load_file(path)

        # Convert to arrays
        self.num_trajectories = len(self.trajectories)
        self.traj_lens = np.array(self.traj_lens)
        self.non_linear = np.array(self.non_linear)
        self.scene_ids = np.array(self.scene_ids)

        # Anchor placeholder (set by trainer)
        self.anchor = None

        # For compatibility with batch sampler
        self.num_peds_in_seq = np.ones(self.num_trajectories, dtype=np.int32)

    def _load_file(self, path):
        """Load trajectories from a single file."""
        trajectories_dict = read_trajectory_file(path, self.delim)

        for traj_id, points in trajectories_dict.items():
            if len(points) < self.seq_len:
                # Skip trajectories that are too short
                # In future, could pad or use B-spline interpolation
                continue

            # Sort by frame index
            points = sorted(points, key=lambda x: x[0])

            # Extract coordinates
            coords = np.array([[p[1], p[2]] for p in points])  # Shape: (T, 2)

            # Apply skip
            if self.skip > 1:
                coords = coords[::self.skip]

            if len(coords) < self.seq_len:
                continue

            # Truncate to max length if needed
            if len(coords) > self.max_traj_len:
                coords = coords[:self.max_traj_len]

            # Check non-linearity
            traj_for_poly = coords.T  # Shape: (2, T)
            is_nonlinear = poly_fit(traj_for_poly, min(len(coords), self.pred_len), self.threshold)

            self.trajectories.append(coords)
            self.traj_lens.append(len(coords))
            self.non_linear.append(is_nonlinear)
            self.scene_ids.append(self.scene_name)

    def __len__(self):
        return self.num_trajectories

    def __getitem__(self, index):
        """Get a single trajectory.

        Returns dictionary with:
            obs_traj: (1, obs_len, 2) observed portion
            pred_traj: (1, pred_len, 2) prediction target
            full_traj: (1, traj_len, 2) full trajectory
            loss_mask: (1, seq_len) mask for valid positions
            traj_len: (1,) actual trajectory length
            non_linear_ped: (1,) whether trajectory is non-linear
            anchor: (1, ...) anchor if set, else None
            frame: (1,) frame index (always 0 for cursor)
            scene_id: (1,) scene identifier
        """
        traj = self.trajectories[index]
        traj_len = self.traj_lens[index]

        # Pad trajectory if needed
        if len(traj) < self.seq_len:
            padded = np.zeros((self.seq_len, 2))
            padded[:len(traj)] = traj
            traj = padded

        # Split into obs and pred
        obs_traj = traj[:self.obs_len]
        pred_traj = traj[self.obs_len:self.seq_len]

        # Create loss mask
        loss_mask = np.zeros(self.seq_len)
        loss_mask[:min(traj_len, self.seq_len)] = 1.0

        # Convert to tensors
        obs_traj = torch.from_numpy(obs_traj).float().unsqueeze(0)  # (1, obs_len, 2)
        pred_traj = torch.from_numpy(pred_traj).float().unsqueeze(0)  # (1, pred_len, 2)
        full_traj = torch.from_numpy(traj[:self.seq_len]).float().unsqueeze(0)  # (1, seq_len, 2)
        loss_mask = torch.from_numpy(loss_mask).float().unsqueeze(0)  # (1, seq_len)
        traj_len_tensor = torch.tensor([traj_len], dtype=torch.long)
        non_linear = torch.tensor([self.non_linear[index]], dtype=torch.float).gt(0.5)

        # Anchor
        if self.anchor is not None:
            anchor = self.anchor[index:index+1]
        else:
            anchor = None

        return {
            "obs_traj": obs_traj,
            "pred_traj": pred_traj,
            "full_traj": full_traj,
            "loss_mask": loss_mask,
            "traj_len": traj_len_tensor,
            "non_linear_ped": non_linear,
            "anchor": anchor,
            "frame": torch.tensor([0], dtype=torch.long),
            "scene_id": np.array([self.scene_ids[index]])
        }

    def get_all_trajectories(self):
        """Get all trajectories as numpy array for anchor initialization.

        Returns:
            Array of shape (N, seq_len, 2)
        """
        all_trajs = []
        for traj in self.trajectories:
            if len(traj) >= self.seq_len:
                all_trajs.append(traj[:self.seq_len])
            else:
                padded = np.zeros((self.seq_len, 2))
                padded[:len(traj)] = traj
                all_trajs.append(padded)
        return np.array(all_trajs)

    def set_anchor(self, anchor):
        """Set anchor tensor for the dataset."""
        self.anchor = anchor
