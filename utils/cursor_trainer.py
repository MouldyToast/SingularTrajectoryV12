"""
Cursor Trajectory Trainer

Trainer adapted for cursor trajectory prediction.
Key differences from pedestrian trainer:
- Uses cursor-specific dataloader
- Simplified anchor calculation (no world/image coordinate transforms)
- Screen coordinates are used directly (identity homography)
"""

import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from .cursor_dataloader import get_cursor_dataloader
from .metrics import compute_batch_ade, compute_batch_fde, AverageMeter
from .utils import reproducibility_settings, DotDict, augment_trajectory


class CursorTrainer:
    """Base trainer class for cursor trajectory prediction."""

    def __init__(self, args, hyper_params):
        print("Cursor Trainer initiating...")

        # Reproducibility
        reproducibility_settings(seed=0)

        self.args, self.hyper_params = args, hyper_params
        self.model, self.optimizer, self.scheduler = None, None, None
        self.loader_train, self.loader_val, self.loader_test = None, None, None

        # Dataset directory: e.g., ./datasets/cursor/Medium/
        self.dataset_dir = os.path.join(hyper_params.dataset_dir, hyper_params.dataset)
        self.checkpoint_dir = os.path.join(
            hyper_params.checkpoint_dir, args.tag, hyper_params.dataset
        )
        print(f"Dataset dir: {self.dataset_dir}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")

        self.log = {'train_loss': [], 'val_loss': []}
        self.stats_func, self.stats_meter = None, None
        self.reset_metric()

        if not args.test:
            # Save arguments and configs
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            with open(os.path.join(self.checkpoint_dir, 'args.pkl'), 'wb') as fp:
                pickle.dump(args, fp)

            with open(os.path.join(self.checkpoint_dir, 'config.pkl'), 'wb') as fp:
                pickle.dump(hyper_params, fp)

    def init_descriptor(self):
        """Initialize Singular space from training trajectories."""
        print("Singular space initialization...")
        obs_traj = self.loader_train.dataset.obs_traj
        pred_traj = self.loader_train.dataset.pred_traj

        if len(obs_traj) == 0:
            raise ValueError("No trajectories loaded for training. Check your data files.")

        # Augment trajectories (flip, reverse)
        obs_traj, pred_traj = augment_trajectory(obs_traj, pred_traj)
        self.model.calculate_parameters(obs_traj, pred_traj)
        print("Anchor generation...")

    def init_adaptive_anchor(self, dataset):
        """Initialize adaptive anchors for a dataset.

        For cursor data, we use a simplified approach since we don't have
        scene images or world/image coordinate transforms.
        """
        print("Adaptive anchor initialization (cursor-simplified)...")
        dataset.anchor = self.calculate_cursor_anchor(dataset)

    def calculate_cursor_anchor(self, dataset):
        """Calculate anchors for cursor trajectories.

        Simplified version that doesn't require homography/vector field transforms.
        Uses the pre-computed cluster centers directly.
        """
        obs_traj = dataset.obs_traj
        pred_traj = dataset.pred_traj
        n_traj = obs_traj.size(0)

        # Get anchor from model's Singular space
        # For cursor, we use the initialized anchors without scene-based adaptation
        mask = self.model.calculate_mask(obs_traj)

        k = self.model.k
        s = self.model.s
        anchor = torch.zeros((n_traj, k, s), dtype=torch.float)

        # Get base anchors from model
        C_anchor_m = self.model.adaptive_anchor_m.C_anchor.detach()
        C_anchor_s = self.model.adaptive_anchor_s.C_anchor.detach()

        # Repeat for each trajectory (no per-scene adaptation for cursor)
        n_moving = mask.sum().item()
        n_static = (~mask).sum().item()

        if n_moving > 0:
            anchor[mask] = C_anchor_m.unsqueeze(0).expand(n_moving, -1, -1)
        if n_static > 0:
            anchor[~mask] = C_anchor_s.unsqueeze(0).expand(n_static, -1, -1)

        return anchor

    def train(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def valid(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

    def fit(self):
        """Main training loop."""
        print("Training started...")
        for epoch in range(self.hyper_params.num_epochs):
            self.train(epoch)
            self.valid(epoch)

            if self.hyper_params.lr_schd:
                self.scheduler.step()

            # Save the best model
            if epoch == 0 or self.log['val_loss'][-1] < min(self.log['val_loss'][:-1]):
                self.save_model()

            print(" ")
            print(f"Dataset: {self.hyper_params.dataset}, Epoch: {epoch}")
            print(f"Train_loss: {self.log['train_loss'][-1]:.8f}, Val_loss: {self.log['val_loss'][-1]:.8f}")
            print(f"Min_val_epoch: {np.array(self.log['val_loss']).argmin()}, "
                  f"Min_val_loss: {np.array(self.log['val_loss']).min():.8f}")
            print(" ")

        print("Training complete.")

    def reset_metric(self):
        self.stats_func = {'ADE': compute_batch_ade, 'FDE': compute_batch_fde}
        self.stats_meter = {x: AverageMeter() for x in self.stats_func.keys()}

    def get_metric(self):
        return self.stats_meter

    def load_model(self, filename='model_best.pth'):
        model_path = os.path.join(self.checkpoint_dir, filename)
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, filename='model_best.pth'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        model_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), model_path)


class CursorTransformerDiffusionTrainer(CursorTrainer):
    """Cursor trajectory trainer using transformer diffusion model."""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

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

        # Initialize model
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

    def train(self, epoch):
        self.model.train()
        loss_batch = 0

        # Initialize anchors if not done
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

        num_traj = len(self.loader_val.dataset)
        self.log['val_loss'].append(loss_batch / max(num_traj, 1))

    @torch.no_grad()
    def test(self):
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

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}
