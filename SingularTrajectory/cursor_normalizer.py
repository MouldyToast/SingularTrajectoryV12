import torch
import math


class CursorTrajNorm:
    """Normalize cursor trajectory with shape (num_trajectories, length_of_time, 2)

    Unlike pedestrian TrajNorm which normalizes from the last observed point,
    CursorTrajNorm normalizes from the START point, with rotation aligned to
    the start-to-end direction.

    This is appropriate for cursor trajectories where we know both the start
    and end points, and want to generate paths between them.

    Args:
        ori (bool): Whether to normalize the trajectory with the origin (at start)
        rot (bool): Whether to normalize the trajectory with the rotation (aligned to end)
        sca (bool): Whether to normalize the trajectory with the scale (unit distance)
    """

    def __init__(self, ori=True, rot=True, sca=True):
        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori = None  # Start point
        self.traj_rot = None  # Rotation matrix
        self.traj_sca = None  # Scale factor
        self.traj_end = None  # End point (for conditioning)

    def calculate_params(self, traj, end_points=None):
        """Calculate the normalization parameters.

        Args:
            traj: Trajectory tensor of shape (N, T, 2)
            end_points: Optional end points tensor of shape (N, 2).
                        If None, uses last point of each trajectory.
        """
        # Origin is the first point
        if self.ori:
            self.traj_ori = traj[:, [0]]  # Shape: (N, 1, 2)

        # Get end points
        if end_points is not None:
            end = end_points
        else:
            end = traj[:, -1]  # Shape: (N, 2)

        start = traj[:, 0]  # Shape: (N, 2)

        # Direction from start to end
        if self.rot:
            direction = end - start
            rot_angle = torch.atan2(direction[:, 1], direction[:, 0])
            # Rotation matrix to align direction with positive X axis
            cos_a = torch.cos(-rot_angle)
            sin_a = torch.sin(-rot_angle)
            self.traj_rot = torch.stack([
                torch.stack([cos_a, -sin_a], dim=1),
                torch.stack([sin_a, cos_a], dim=1)
            ], dim=1)  # Shape: (N, 2, 2)

        # Scale is the distance from start to end
        if self.sca:
            direction = end - start
            distance = direction.norm(p=2, dim=-1)
            # Avoid division by zero for stationary trajectories
            distance = torch.clamp(distance, min=1e-6)
            self.traj_sca = (1.0 / distance)[:, None, None]  # Shape: (N, 1, 1)

        # Store end points for reference
        self.traj_end = end

    def calculate_params_from_endpoints(self, start_points, end_points):
        """Calculate normalization parameters directly from start and end points.

        Useful for generation when we only have start and end, not full trajectory.

        Args:
            start_points: Start points tensor of shape (N, 2)
            end_points: End points tensor of shape (N, 2)
        """
        if self.ori:
            self.traj_ori = start_points.unsqueeze(1)  # Shape: (N, 1, 2)

        direction = end_points - start_points

        if self.rot:
            rot_angle = torch.atan2(direction[:, 1], direction[:, 0])
            cos_a = torch.cos(-rot_angle)
            sin_a = torch.sin(-rot_angle)
            self.traj_rot = torch.stack([
                torch.stack([cos_a, -sin_a], dim=1),
                torch.stack([sin_a, cos_a], dim=1)
            ], dim=1)

        if self.sca:
            distance = direction.norm(p=2, dim=-1)
            distance = torch.clamp(distance, min=1e-6)
            self.traj_sca = (1.0 / distance)[:, None, None]

        self.traj_end = end_points

    def get_params(self):
        """Get the normalization parameters."""
        return (self.ori, self.rot, self.sca,
                self.traj_ori, self.traj_rot, self.traj_sca, self.traj_end)

    def set_params(self, ori, rot, sca, traj_ori, traj_rot, traj_sca, traj_end=None):
        """Set the normalization parameters."""
        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori, self.traj_rot, self.traj_sca = traj_ori, traj_rot, traj_sca
        self.traj_end = traj_end

    def normalize(self, traj):
        """Normalize the trajectory.

        Transforms trajectory so that:
        - Start point is at origin (0, 0)
        - End point is on positive X axis
        - Distance from start to end is 1.0

        Args:
            traj: Trajectory tensor of shape (N, T, 2)

        Returns:
            Normalized trajectory of shape (N, T, 2)
        """
        if self.ori:
            traj = traj - self.traj_ori
        if self.rot:
            traj = traj @ self.traj_rot
        if self.sca:
            traj = traj * self.traj_sca
        return traj

    def denormalize(self, traj):
        """Denormalize the trajectory.

        Reverse the normalization to get back to original coordinate space.

        Args:
            traj: Normalized trajectory tensor of shape (N, T, 2)

        Returns:
            Denormalized trajectory of shape (N, T, 2)
        """
        if self.sca:
            traj = traj / self.traj_sca
        if self.rot:
            # Transpose rotation matrix for inverse
            traj = traj @ self.traj_rot.transpose(-1, -2)
        if self.ori:
            traj = traj + self.traj_ori
        return traj

    def get_normalized_endpoint(self):
        """Get the normalized end point (should be approximately (1, 0))."""
        if self.traj_end is None:
            return None
        # The end point after normalization should be at (distance, 0) before scaling
        # and (1, 0) after scaling
        return torch.ones(self.traj_end.shape[0], 2, device=self.traj_end.device) * torch.tensor([1.0, 0.0])


class CursorTrajNormNumpy:
    """NumPy version of CursorTrajNorm for use in data preprocessing."""

    def __init__(self, ori=True, rot=True, sca=True):
        import numpy as np
        self.np = np
        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori = None
        self.traj_rot = None
        self.traj_sca = None

    def calculate_params(self, traj):
        """Calculate params for a single trajectory.

        Args:
            traj: numpy array of shape (T, 2)
        """
        np = self.np

        if self.ori:
            self.traj_ori = traj[0]  # Start point

        start = traj[0]
        end = traj[-1]
        direction = end - start

        if self.rot:
            angle = np.arctan2(direction[1], direction[0])
            cos_a = np.cos(-angle)
            sin_a = np.sin(-angle)
            self.traj_rot = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])

        if self.sca:
            distance = np.linalg.norm(direction)
            self.traj_sca = 1.0 / max(distance, 1e-6)

    def normalize(self, traj):
        """Normalize trajectory.

        Args:
            traj: numpy array of shape (T, 2)

        Returns:
            Normalized trajectory of shape (T, 2)
        """
        if self.ori:
            traj = traj - self.traj_ori
        if self.rot:
            traj = traj @ self.traj_rot.T
        if self.sca:
            traj = traj * self.traj_sca
        return traj

    def denormalize(self, traj):
        """Denormalize trajectory."""
        if self.sca:
            traj = traj / self.traj_sca
        if self.rot:
            traj = traj @ self.traj_rot
        if self.ori:
            traj = traj + self.traj_ori
        return traj
