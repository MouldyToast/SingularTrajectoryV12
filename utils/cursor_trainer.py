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

        For cursor data, we use base anchors (no scene-specific adaptation).
        Vector field adaptation can be added later.
        """
        print("Adaptive anchor initialization...")

        # Use base anchors (the KMeans cluster centers)
        dataset.anchor = self._calculate_base_anchor(dataset)

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
