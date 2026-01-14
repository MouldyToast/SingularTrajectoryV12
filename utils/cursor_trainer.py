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

    # init_descriptor is inherited from STTrainer - works as-is

    def init_adaptive_anchor(self, dataset):
        """Initialize adaptive anchors for cursor dataset.

        For cursor data, we use a simplified approach:
        - Screen coordinates are used directly (no world/image transform)
        - Vector field adaptation is optional
        """
        print("Adaptive anchor initialization...")

        obs_traj = dataset.obs_traj
        n_traj = obs_traj.size(0)

        # Check if vector field is available
        if dataset.vector_field is not None:
            # Use full adaptive anchor calculation with vector field
            dataset.anchor = self._calculate_adaptive_anchor_with_vectorfield(dataset)
        else:
            # Use base anchors without scene adaptation
            dataset.anchor = self._calculate_base_anchor(dataset)

    def _calculate_base_anchor(self, dataset):
        """Calculate anchors using pre-computed cluster centers (no vector field)."""
        obs_traj = dataset.obs_traj
        n_traj = obs_traj.size(0)

        mask = self.model.calculate_mask(obs_traj)

        k = self.model.k
        s = self.model.s
        anchor = torch.zeros((n_traj, k, s), dtype=torch.float)

        # Get base anchors from model
        C_anchor_m = self.model.adaptive_anchor_m.C_anchor.detach()
        C_anchor_s = self.model.adaptive_anchor_s.C_anchor.detach()

        n_moving = mask.sum().item()
        n_static = (~mask).sum().item()

        if n_moving > 0:
            anchor[mask] = C_anchor_m.unsqueeze(0).expand(n_moving, -1, -1)
        if n_static > 0:
            anchor[~mask] = C_anchor_s.unsqueeze(0).expand(n_static, -1, -1)

        return anchor

    def _calculate_adaptive_anchor_with_vectorfield(self, dataset):
        """Calculate anchors with vector field adaptation.

        Adapts the anchor endpoints based on the cursor flow field.
        Since cursor data uses screen coordinates directly, we skip the
        homography transforms but still use the vector field for adaptation.
        """
        obs_traj = dataset.obs_traj
        scene_id = dataset.scene_id
        vector_field = dataset.vector_field
        n_traj = obs_traj.size(0)

        mask = self.model.calculate_mask(obs_traj)

        k = self.model.k
        s = self.model.s
        anchor = torch.zeros((n_traj, k, s), dtype=torch.float)

        # Calculate for moving trajectories
        if mask.sum() > 0:
            obs_m_traj = obs_traj[mask]
            scene_id_m = scene_id[mask]
            anchor[mask] = self._adaptive_anchor_cursor(
                obs_m_traj, scene_id_m, vector_field,
                self.model.adaptive_anchor_m, self.model.Singular_space_m
            )

        # Calculate for static trajectories
        if (~mask).sum() > 0:
            obs_s_traj = obs_traj[~mask]
            scene_id_s = scene_id[~mask]
            anchor[~mask] = self._adaptive_anchor_cursor(
                obs_s_traj, scene_id_s, vector_field,
                self.model.adaptive_anchor_s, self.model.Singular_space_s
            )

        return anchor

    def _adaptive_anchor_cursor(self, obs_traj, scene_id, vector_field, anchor_module, space):
        """Cursor-specific adaptive anchor calculation.

        Similar to AdaptiveAnchor.adaptive_anchor_calculation but without
        homography transforms (screen coords = world coords for cursor).
        """
        n_ped = obs_traj.size(0)
        V_trunc = space.V_trunc

        space.traj_normalizer.calculate_params(obs_traj.cuda().detach())

        # Get initial anchors
        init_anchor = anchor_module.C_anchor.unsqueeze(dim=0).repeat_interleave(repeats=n_ped, dim=0).detach()
        init_anchor = init_anchor.permute(2, 1, 0)

        # Convert to Euclidean space
        init_anchor_euclidean = space.batch_to_Euclidean_space(init_anchor, evec=V_trunc)
        init_anchor_euclidean = space.traj_normalizer.denormalize(init_anchor_euclidean).cpu().numpy()
        adaptive_anchor_euclidean = init_anchor_euclidean.copy()
        obs_traj_np = obs_traj.cpu().numpy()

        # Vector field parameters (from generate_cursor_vector_field.py)
        GRID_MIN_X, GRID_MAX_X = -0.2, 1.2
        GRID_MIN_Y, GRID_MAX_Y = -0.6, 0.6

        if vector_field is not None:
            grid_h, grid_w = vector_field.shape[:2]
            cell_w = (GRID_MAX_X - GRID_MIN_X) / grid_w
            cell_h = (GRID_MAX_Y - GRID_MIN_Y) / grid_h

            for ped_id in range(n_ped):
                # For cursor, screen coords are used directly
                # Adapt anchors based on vector field flow
                prototype = init_anchor_euclidean[:, ped_id]  # (s, t, 2)
                startpoint = obs_traj_np[ped_id, -1]  # Last observed point

                for sample_idx in range(prototype.shape[0]):
                    endpoint = prototype[sample_idx, -1]  # End of this anchor

                    # Convert to grid coordinates
                    grid_x = int((endpoint[0] - GRID_MIN_X) / cell_w)
                    grid_y = int((endpoint[1] - GRID_MIN_Y) / cell_h)

                    # Clamp to grid bounds
                    grid_x = max(0, min(grid_w - 1, grid_x))
                    grid_y = max(0, min(grid_h - 1, grid_y))

                    # Get flow direction from vector field
                    flow = vector_field[grid_y, grid_x]

                    if np.linalg.norm(flow) > 0.1:
                        # Adjust endpoint along flow direction
                        # Scale adjustment based on distance to endpoint
                        dist_to_end = np.linalg.norm(endpoint - startpoint)
                        adjustment = flow * dist_to_end * 0.1  # Small adjustment

                        # Apply adjustment to entire trajectory proportionally
                        for t in range(prototype.shape[1]):
                            t_ratio = (t + 1) / prototype.shape[1]
                            adaptive_anchor_euclidean[sample_idx, ped_id, t] += adjustment * t_ratio

        # Convert back to Singular space
        adaptive_anchor_euclidean = space.traj_normalizer.normalize(
            torch.FloatTensor(adaptive_anchor_euclidean).cuda()
        )
        adaptive_anchor = space.batch_to_Singular_space(adaptive_anchor_euclidean, evec=V_trunc)
        adaptive_anchor = adaptive_anchor.permute(2, 1, 0).cpu()

        return adaptive_anchor

    def train(self, epoch):
        """Training step - same as STTransformerDiffusionTrainer."""
        self.model.train()
        loss_batch = 0

        if self.loader_train.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_train.dataset)

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

            loss = output["loss_euclidean_ade"]
            loss[torch.isnan(loss)] = 0
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
