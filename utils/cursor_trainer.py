"""
Cursor Trajectory Trainer

Properly inherits from the existing trainer infrastructure.
Only overrides what's necessary for cursor data:
- Dataloader initialization (uses cursor dataloader)
- Adaptive anchor calculation (simplified for screen coordinates)
"""

import os
import torch
import numpy as np
from tqdm import tqdm

from .trainer import STTrainer
from .cursor_dataloader import get_cursor_dataloader
from .utils import DotDict


class CursorTransformerDiffusionTrainer(STTrainer):
    """Cursor trajectory trainer using transformer diffusion model.

    Inherits from STTrainer and overrides only what's necessary for cursor data.
    """

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        # Call parent init (sets up checkpoint dirs, logging, etc.)
        super().__init__(args, hyper_params)

        # Override dataset_dir construction for cursor flat structure
        # Parent does: hyper_params.dataset_dir + hyper_params.dataset + '/'
        # We need: hyper_params.dataset_dir + hyper_params.dataset (no trailing slash issues)
        self.dataset_dir = os.path.join(hyper_params.dataset_dir.rstrip('/'), hyper_params.dataset)

        # Initialize dataloaders with cursor-specific loader
        batch_size = hyper_params.batch_size
        obs_len = hyper_params.obs_len
        pred_len = hyper_params.pred_len
        skip = hyper_params.skip

        self.loader_train = get_cursor_dataloader(
            self.dataset_dir, 'train', obs_len, pred_len,
            batch_size=batch_size, skip=skip
        )
        self.loader_val = get_cursor_dataloader(
            self.dataset_dir, 'val', obs_len, pred_len,
            batch_size=batch_size
        )
        self.loader_test = get_cursor_dataloader(
            self.dataset_dir, 'test', obs_len, pred_len,
            batch_size=1
        )

        print(f"Loaded {len(self.loader_train.dataset)} training trajectories")
        print(f"Loaded {len(self.loader_val.dataset)} validation trajectories")
        print(f"Loaded {len(self.loader_test.dataset)} test trajectories")

        # Initialize model (same as STTransformerDiffusionTrainer)
        cfg = DotDict({
            'scheduler': 'ddim',
            'steps': 10,
            'beta_start': 1.e-4,
            'beta_end': 5.e-2,
            'beta_schedule': 'linear',
            'k': hyper_params.k,
            's': hyper_params.num_samples
        })

        predictor_model = base_model(cfg).cuda()
        eigentraj_model = model(
            baseline_model=predictor_model,
            hook_func=hook_func,
            hyper_params=hyper_params
        ).cuda()

        self.model = eigentraj_model

        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=hyper_params.lr,
            weight_decay=hyper_params.weight_decay
        )

        if hyper_params.lr_schd:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=hyper_params.lr_schd_step,
                gamma=hyper_params.lr_schd_gamma
            )

    def init_descriptor(self):
        """Initialize Singular space and anchors for cursor data.

        Override parent to handle the case where one trajectory category
        (moving or static) may be empty. For cursor data, typically ALL
        trajectories are "moving" since they represent mouse movements.
        """
        from .utils import augment_trajectory

        print("Singular space initialization...")
        obs_traj = self.loader_train.dataset.obs_traj
        pred_traj = self.loader_train.dataset.pred_traj

        # Augment trajectories (rotation augmentation)
        obs_traj, pred_traj = augment_trajectory(obs_traj, pred_traj)

        # Split into moving and static categories
        mask = self.model.calculate_mask(obs_traj)
        n_moving = mask.sum().item()
        n_static = (~mask).sum().item()

        print(f"  Moving trajectories: {n_moving}")
        print(f"  Static trajectories: {n_static}")

        obs_m_traj, pred_m_traj = obs_traj[mask], pred_traj[mask]
        obs_s_traj, pred_s_traj = obs_traj[~mask], pred_traj[~mask]

        # Initialize moving trajectories (if any)
        if n_moving >= self.hyper_params.num_samples:
            print("  Initializing moving trajectory space...")
            data_m = self.model.Singular_space_m.parameter_initialization(obs_m_traj, pred_m_traj)
            self.model.adaptive_anchor_m.anchor_initialization(*data_m)
        elif n_moving > 0:
            print(f"  Warning: Only {n_moving} moving trajectories, need {self.hyper_params.num_samples} for KMeans")
            print("  Using all moving trajectories as anchor samples...")
            data_m = self.model.Singular_space_m.parameter_initialization(obs_m_traj, pred_m_traj)
            # Use fewer clusters if we have fewer samples
            self._anchor_init_with_limited_samples(
                self.model.adaptive_anchor_m,
                data_m[0], data_m[1],
                n_moving
            )
        else:
            print("  No moving trajectories - skipping moving space initialization")

        # Initialize static trajectories (if any)
        if n_static >= self.hyper_params.num_samples:
            print("  Initializing static trajectory space...")
            data_s = self.model.Singular_space_s.parameter_initialization(obs_s_traj, pred_s_traj)
            self.model.adaptive_anchor_s.anchor_initialization(*data_s)
        elif n_static > 0:
            print(f"  Warning: Only {n_static} static trajectories, need {self.hyper_params.num_samples} for KMeans")
            print("  Using all static trajectories as anchor samples...")
            data_s = self.model.Singular_space_s.parameter_initialization(obs_s_traj, pred_s_traj)
            self._anchor_init_with_limited_samples(
                self.model.adaptive_anchor_s,
                data_s[0], data_s[1],
                n_static
            )
        else:
            print("  No static trajectories - skipping static space initialization")

        print("Anchor generation complete.")

    def _anchor_init_with_limited_samples(self, anchor_module, pred_traj_norm, V_pred_trunc, n_samples):
        """Initialize anchors when we have fewer samples than num_samples.

        Uses all available samples as anchors instead of KMeans clustering.
        """
        from sklearn.cluster import KMeans
        import torch.nn as nn

        # Project trajectories to Singular space
        C_pred = anchor_module.to_Singular_space(pred_traj_norm, evec=V_pred_trunc).T.detach().numpy()

        # Use KMeans with n_clusters = min(n_samples, num_samples)
        n_clusters = min(n_samples, anchor_module.s)
        C_anchor = torch.FloatTensor(
            KMeans(n_clusters=n_clusters, random_state=0, init='k-means++', n_init=1)
            .fit(C_pred).cluster_centers_.T
        )

        # Pad with zeros if we have fewer clusters than expected
        if n_clusters < anchor_module.s:
            padded = torch.zeros((anchor_module.k, anchor_module.s))
            padded[:, :n_clusters] = C_anchor
            C_anchor = padded

        anchor_module.C_anchor = nn.Parameter(C_anchor.to(anchor_module.C_anchor.device))

    def init_adaptive_anchor(self, dataset):
        """Initialize adaptive anchors for cursor dataset.

        Uses vector field to adapt anchor endpoints based on scene flow.
        """
        print("Adaptive anchor initialization...")

        # Check if vector field is available
        if dataset.vector_field and len(dataset.vector_field) > 0:
            print("  Using vector field adaptation...")
            dataset.anchor = self._calculate_adaptive_anchor(dataset)
        else:
            print("  No vector field - using base anchors...")
            dataset.anchor = self._calculate_base_anchor(dataset)

    def _calculate_adaptive_anchor(self, dataset):
        """Calculate adaptive anchors using vector field (like original).

        For cursor data:
        - Screen coordinates are used directly (no homography)
        - Vector field guides trajectory endpoints
        """
        obs_traj = dataset.obs_traj
        pred_traj = dataset.pred_traj
        scene_id = dataset.scene_id
        vector_field = dataset.vector_field
        homography = dataset.homography  # Identity for cursor

        n_traj = obs_traj.size(0)
        mask = self.model.calculate_mask(obs_traj)

        k = self.model.k
        s = self.model.s
        anchor = torch.zeros((n_traj, k, s), dtype=torch.float)

        # Calculate for moving trajectories
        n_moving = mask.sum().item()
        if n_moving > 0:
            obs_m_traj = obs_traj[mask]
            scene_id_m = scene_id[mask]
            anchor[mask] = self._adaptive_anchor_with_vectorfield(
                obs_m_traj, scene_id_m, vector_field, homography,
                self.model.adaptive_anchor_m, self.model.Singular_space_m
            )

        # Calculate for static trajectories
        n_static = (~mask).sum().item()
        if n_static > 0:
            obs_s_traj = obs_traj[~mask]
            scene_id_s = scene_id[~mask]
            anchor[~mask] = self._adaptive_anchor_with_vectorfield(
                obs_s_traj, scene_id_s, vector_field, homography,
                self.model.adaptive_anchor_s, self.model.Singular_space_s
            )

        return anchor

    def _adaptive_anchor_with_vectorfield(self, obs_traj, scene_id, vector_field, homography, anchor_module, space):
        """Cursor-specific adaptive anchor calculation.

        Mirrors AdaptiveAnchor.adaptive_anchor_calculation but simplified for cursor:
        - No world/image coordinate transform (identity homography)
        - Vector field in normalized cursor space
        - Properly handles per-trajectory scene IDs with bilinear interpolation
        """
        n_ped = obs_traj.size(0)
        V_trunc = space.V_trunc
        s = anchor_module.s

        # Calculate normalization params for this batch
        space.traj_normalizer.calculate_params(obs_traj.cuda().detach())

        # Get initial anchors and expand for each trajectory
        init_anchor = anchor_module.C_anchor.unsqueeze(dim=0).repeat_interleave(repeats=n_ped, dim=0).detach()
        init_anchor = init_anchor.permute(2, 1, 0)  # (s, k, n_ped)

        # Convert to Euclidean space
        init_anchor_euclidean = space.batch_to_Euclidean_space(init_anchor, evec=V_trunc)
        init_anchor_euclidean = space.traj_normalizer.denormalize(init_anchor_euclidean).cpu().numpy()
        adaptive_anchor_euclidean = init_anchor_euclidean.copy()
        obs_traj_np = obs_traj.cpu().numpy()

        # Process each trajectory with its own scene's vector field
        for ped_id in range(n_ped):
            # Get vector field for THIS trajectory's scene (not just first one)
            scene_name = scene_id[ped_id] if ped_id < len(scene_id) else None
            vf = vector_field.get(scene_name) if scene_name else None

            if vf is None:
                continue

            # Vector field grid parameters (from generate_cursor_vector_field.py)
            # Normalized space: x in [-0.2, 1.2], y in [-0.6, 0.6]
            grid_h, grid_w = vf.shape[:2]
            x_min, x_max = -0.2, 1.2
            y_min, y_max = -0.6, 0.6
            cell_w = (x_max - x_min) / grid_w
            cell_h = (y_max - y_min) / grid_h

            startpoint = obs_traj_np[ped_id, -1]  # Last observed point

            for sample_idx in range(s):
                # Get prototype trajectory endpoint
                prototype = init_anchor_euclidean[sample_idx, ped_id]  # (pred_len, 2)
                endpoint = prototype[-1]  # End of this anchor

                # Convert to grid coordinates (float for interpolation)
                grid_x_f = (endpoint[0] - x_min) / cell_w
                grid_y_f = (endpoint[1] - y_min) / cell_h

                # Bilinear interpolation for smoother vector field lookup
                grid_x0 = int(np.floor(grid_x_f))
                grid_y0 = int(np.floor(grid_y_f))
                grid_x1 = grid_x0 + 1
                grid_y1 = grid_y0 + 1

                # Clamp to grid bounds
                grid_x0 = max(0, min(grid_w - 1, grid_x0))
                grid_x1 = max(0, min(grid_w - 1, grid_x1))
                grid_y0 = max(0, min(grid_h - 1, grid_y0))
                grid_y1 = max(0, min(grid_h - 1, grid_y1))

                # Interpolation weights
                wx = grid_x_f - np.floor(grid_x_f)
                wy = grid_y_f - np.floor(grid_y_f)

                # Bilinear interpolation of flow target
                flow_target = (
                    vf[grid_y0, grid_x0] * (1 - wx) * (1 - wy) +
                    vf[grid_y0, grid_x1] * wx * (1 - wy) +
                    vf[grid_y1, grid_x0] * (1 - wx) * wy +
                    vf[grid_y1, grid_x1] * wx * wy
                )

                # Check if this is a valid flow (non-zero)
                if np.linalg.norm(flow_target) > 0.01:
                    # Scale trajectory endpoints to match flow
                    if np.linalg.norm(endpoint - startpoint) > 0.01:
                        scale_xy = (flow_target - startpoint) / (endpoint - startpoint + 1e-6)
                        # Clamp scale to reasonable range
                        scale_xy = np.clip(scale_xy, 0.5, 2.0)

                        # Apply scaling to entire trajectory
                        for t in range(prototype.shape[0]):
                            adaptive_anchor_euclidean[sample_idx, ped_id, t] = (
                                (prototype[t] - startpoint) * scale_xy + startpoint
                            )

        # Convert back to Singular space
        adaptive_anchor_euclidean = space.traj_normalizer.normalize(
            torch.FloatTensor(adaptive_anchor_euclidean).cuda()
        )
        adaptive_anchor = space.batch_to_Singular_space(adaptive_anchor_euclidean, evec=V_trunc)
        adaptive_anchor = adaptive_anchor.permute(2, 1, 0).cpu()  # (n_ped, k, s)

        return adaptive_anchor

    def _calculate_base_anchor(self, dataset):
        """Calculate anchors using pre-computed cluster centers (no vector field)."""
        obs_traj = dataset.obs_traj
        n_traj = obs_traj.size(0)

        mask = self.model.calculate_mask(obs_traj)

        k = self.model.k
        s = self.model.s
        anchor = torch.zeros((n_traj, k, s), dtype=torch.float)

        # Get base anchors from model (move to CPU since anchor tensor is on CPU)
        C_anchor_m = self.model.adaptive_anchor_m.C_anchor.detach().cpu()
        C_anchor_s = self.model.adaptive_anchor_s.C_anchor.detach().cpu()

        n_moving = mask.sum().item()
        n_static = (~mask).sum().item()

        if n_moving > 0:
            anchor[mask] = C_anchor_m.unsqueeze(0).expand(n_moving, -1, -1)
        if n_static > 0:
            anchor[~mask] = C_anchor_s.unsqueeze(0).expand(n_static, -1, -1)

        return anchor

    def train(self, epoch):
        """Training step with complete loss function.

        Uses all three loss components from original SingularTrajectory:
        - loss_eigentraj: Coefficient space error (low-rank approximation)
        - loss_euclidean_ade: Average displacement error in Euclidean space
        - loss_euclidean_fde: Final displacement error in Euclidean space
        """
        self.model.train()
        loss_batch = 0

        if self.loader_train.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_train.dataset)

        # Loss weights (can be tuned or made configurable)
        # Higher FDE weight emphasizes endpoint accuracy (critical for cursor generation)
        w_eigentraj = 0.1  # Coefficient space loss
        w_ade = 1.0        # Average displacement error
        w_fde = 1.0        # Final displacement error (endpoint accuracy)

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj = batch["obs_traj"].cuda(non_blocking=True)
            pred_traj = batch["pred_traj"].cuda(non_blocking=True)
            adaptive_anchor = batch["anchor"].cuda(non_blocking=True)
            scene_mask = batch["scene_mask"].cuda(non_blocking=True)

            self.optimizer.zero_grad()

            additional_information = {
                "scene_mask": scene_mask,
                "num_samples": self.hyper_params.num_samples
            }
            output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info=additional_information)

            # Combined loss with all three components
            loss_eigentraj = output.get("loss_eigentraj", torch.tensor(0.0).cuda())
            loss_ade = output["loss_euclidean_ade"]
            loss_fde = output.get("loss_euclidean_fde", torch.tensor(0.0).cuda())

            # Handle NaN values
            if torch.isnan(loss_eigentraj):
                loss_eigentraj = torch.tensor(0.0).cuda()
            if torch.isnan(loss_ade):
                loss_ade = torch.tensor(0.0).cuda()
            if torch.isnan(loss_fde):
                loss_fde = torch.tensor(0.0).cuda()

            loss = w_eigentraj * loss_eigentraj + w_ade * loss_ade + w_fde * loss_fde
            loss_batch += loss.item()

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / max(len(self.loader_train), 1))

    @torch.no_grad()
    def valid(self, epoch):
        """Validation step - same as STTransformerDiffusionTrainer."""
        self.model.eval()
        loss_batch = 0

        if self.loader_val.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_val.dataset)

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj = batch["obs_traj"].cuda(non_blocking=True)
            pred_traj = batch["pred_traj"].cuda(non_blocking=True)
            adaptive_anchor = batch["anchor"].cuda(non_blocking=True)
            scene_mask = batch["scene_mask"].cuda(non_blocking=True)

            additional_information = {
                "scene_mask": scene_mask,
                "num_samples": self.hyper_params.num_samples
            }
            output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info=additional_information)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_traj = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / max(num_traj, 1))

    @torch.no_grad()
    def test(self):
        """Test step - same as STTransformerDiffusionTrainer."""
        self.model.eval()
        self.reset_metric()

        if self.loader_test.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_test.dataset)

        for cnt, batch in enumerate(tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()}")):
            obs_traj = batch["obs_traj"].cuda(non_blocking=True)
            pred_traj = batch["pred_traj"].cuda(non_blocking=True)
            adaptive_anchor = batch["anchor"].cuda(non_blocking=True)
            scene_mask = batch["scene_mask"].cuda(non_blocking=True)

            additional_information = {
                "scene_mask": scene_mask,
                "num_samples": self.hyper_params.num_samples
            }
            output = self.model(obs_traj, adaptive_anchor, addl_info=additional_information)

            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}
