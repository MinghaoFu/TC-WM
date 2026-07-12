import os
import json
import pickle
from pathlib import Path
from typing import Callable, Optional

import decord
import numpy as np
import torch
from decord import VideoReader
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset

decord.bridge.set_bridge("torch")


class DMCDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        relative: bool = True,
        action_scale: float = 1.0,
        with_velocity: bool = True,
        image_size: int = 224,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.relative = relative
        self.normalize_action = normalize_action
        self.image_size = image_size
        self.task_name = self.data_path.name

        self.states = torch.load(self.data_path / "states.pth").float()
        if relative:
            self.actions = torch.load(self.data_path / "rel_actions.pth").float()
        else:
            self.actions = torch.load(self.data_path / "abs_actions.pth").float()
        self.actions = self.actions / action_scale

        with open(self.data_path / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        self.n_rollout = n_rollout
        n = self.n_rollout if self.n_rollout else len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]

        try:
            self.proprios = torch.load(self.data_path / "proprios.pth").float()
            self.proprios = self.proprios[:n]
        except FileNotFoundError:
            self.proprios = self.states.clone()

        self.with_velocity = with_velocity
        if with_velocity and (self.data_path / "velocities.pth").exists():
            self.velocities = torch.load(self.data_path / "velocities.pth").float()
            self.velocities = self.velocities[:n]
        else:
            self.velocities = None

        self._maybe_trim_padded_dims()
        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean = torch.load(self.data_path / "action_mean.pth")
            self.action_std = torch.load(self.data_path / "action_std.pth")
            self.state_mean = torch.load(self.data_path / "state_mean.pth")
            self.state_std = torch.load(self.data_path / "state_std.pth")
            self.proprio_mean = torch.load(self.data_path / "proprio_mean.pth")
            self.proprio_std = torch.load(self.data_path / "proprio_std.pth")
            if self.action_mean.shape[-1] != self.action_dim:
                self.action_mean = self.action_mean[: self.action_dim]
                self.action_std = self.action_std[: self.action_dim]
            if self.state_mean.shape[-1] != self.state_dim:
                self.state_mean = self.state_mean[: self.state_dim]
                self.state_std = self.state_std[: self.state_dim]
            if self.proprio_mean.shape[-1] != self.proprio_dim:
                self.proprio_mean = self.proprio_mean[: self.proprio_dim]
                self.proprio_std = self.proprio_std[: self.proprio_dim]
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

        obs_dir = self.data_path / "obses"
        if not obs_dir.exists():
            raise FileNotFoundError(f"Missing obs directory: {obs_dir}")
        if not any(obs_dir.glob("episode_*.mp4")):
            raise FileNotFoundError(f"No episode_*.mp4 found in {obs_dir}")

    def _maybe_trim_padded_dims(self):
        # Trim padded trailing dims for actions/states/proprios, if present.
        action_dim = self._get_task_action_dim()
        if action_dim is not None and self.actions.shape[-1] > action_dim:
            self.actions = self.actions[..., :action_dim]
        else:
            self.actions = self._trim_trailing_zeros(self.actions)

        self.states = self._trim_trailing_zeros(self.states)
        self.proprios = self._trim_trailing_zeros(self.proprios)

    def _get_task_action_dim(self):
        tasks_fp = Path("tasks.json")  # override with your task-embedding file if using multitask
        if not tasks_fp.exists():
            return None
        try:
            with open(tasks_fp, "r") as f:
                tasks = json.load(f)
            info = tasks.get(self.task_name)
            if info is None:
                return None
            return int(info.get("action_dim"))
        except Exception:
            return None

    @staticmethod
    def _trim_trailing_zeros(tensor):
        # Only trim if trailing dims are exactly zero across all samples/timesteps.
        if tensor.numel() == 0:
            return tensor
        flat = tensor.reshape(-1, tensor.shape[-1])
        nonzero = (flat.abs().max(dim=0).values > 0)
        if not nonzero.any():
            return tensor[..., :1]
        last_idx = int(nonzero.nonzero()[-1]) + 1
        if last_idx == tensor.shape[-1]:
            return tensor
        return tensor[..., :last_idx]

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]

        vid_dir = self.data_path / "obses"
        reader = VideoReader(
            str(vid_dir / f"episode_{idx:05d}.mp4"), num_threads=1
        )
        image = reader.get_batch(frames)
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")

        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        if isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_dmc_slice_train_val(
    transform,
    data_path,
    n_rollout=50,
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    with_velocity=True,
    image_size=224,
):
    train_path = os.path.join(data_path, "train")
    val_path = os.path.join(data_path, "val")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_dset = DMCDataset(
            data_path=train_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            with_velocity=with_velocity,
            image_size=image_size,
        )
        val_dset = DMCDataset(
            data_path=val_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            with_velocity=with_velocity,
            image_size=image_size,
        )
    else:
        print(f"No train/val split found, using single dataset at {data_path}")
        full_dset = DMCDataset(
            data_path=data_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            with_velocity=with_velocity,
            image_size=image_size,
        )

        total_samples = len(full_dset)
        train_size = int(total_samples * split_ratio)
        val_size = total_samples - train_size
        print(f"Total samples: {total_samples}, Train: {train_size}, Val: {val_size}")

        train_dset = TrajSubset(full_dset, list(range(train_size)))
        val_dset = TrajSubset(full_dset, list(range(train_size, total_samples)))

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": train_dset, "valid": val_dset}
    return datasets, traj_dset
