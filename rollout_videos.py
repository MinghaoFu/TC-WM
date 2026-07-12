"""Standalone rollout video generator — works with all frameworks.

Reuses openloop_rollout logic from train_uwm.py. Checkpoint loading
follows the same resume_folder convention as training.

Usage:
    # Single-task (tcwm, dinowm, dreamerv3, etc.)
    CUDA_VISIBLE_DEVICES=0 python rollout_videos.py --config-name=train_dreamerv3 \
        env=lift \
        resume_folder=/path/to/ckpt_dir \
        +output_dir=rollout_results/dreamerv3_lift \
        +num_per_task=10

    # Multi-task (mtjepa variants)
    CUDA_VISIBLE_DEVICES=0 python rollout_videos.py --config-name=train_mtjepa_token \
        resume_folder=/path/to/ckpt_dir \
        +output_dir=rollout_results/mtjepa_token_50ep \
        +num_per_task=10

    # Minimal — auto-generate output dir from config
    CUDA_VISIBLE_DEVICES=0 python rollout_videos.py --config-name=train_tcwm \
        env=lift resume_folder=/path/to/ckpt_dir
"""
import os
import sys

# Disable wandb completely BEFORE any import
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# Monkey-patch wandb.login so train_uwm.py import doesn't fail
import wandb
wandb.login = lambda **kwargs: None

import hydra
import torch
import numpy as np
import imageio
import logging
import warnings
from pathlib import Path
from omegaconf import OmegaConf, open_dict
from einops import rearrange

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# Register resolvers BEFORE hydra processes config
OmegaConf.register_new_resolver("maybe_suffix", lambda s: f"-{s}" if s else "")
OmegaConf.register_new_resolver(
    "env_output_name",
    lambda env_name, dataset_name=None: env_name,
)
# Patch to tolerate duplicate registration from train_uwm.py import
_orig_register = OmegaConf.register_new_resolver
def _safe_register(name, resolver, **kw):
    try:
        _orig_register(name, resolver, **kw)
    except ValueError:
        pass  # already registered, ignore
OmegaConf.register_new_resolver = _safe_register


def denorm(x):
    """[-1,1] → [0,255] uint8"""
    return ((x.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)


def make_frame(gt_frame, pred_frame):
    """Stack GT (top) and pred (bottom) with a green/purple label bar."""
    gt = denorm(gt_frame).permute(1, 2, 0).numpy()      # (H, W, 3)
    pred = denorm(pred_frame).permute(1, 2, 0).numpy()
    W = gt.shape[1]
    gt_bar = np.full((4, W, 3), [34, 197, 94], dtype=np.uint8)
    pred_bar = np.full((4, W, 3), [139, 92, 246], dtype=np.uint8)
    return np.concatenate([gt_bar, gt, pred_bar, pred], axis=0)


def build_trainer(cfg):
    """Build a minimal Trainer: dataset + model + checkpoint, no wandb/training.

    Checkpoint loading follows the same path as train_uwm.py:
    init_models() internally calls load_ckpt(resume_folder/checkpoints/model_latest.pth).
    """
    from train_uwm import Trainer

    # Validate resume_folder before heavy init
    resume = cfg.resume_folder
    if resume in (None, "none", "", "null"):
        raise ValueError(
            "resume_folder is required. Pass resume_folder=/path/to/experiment_dir"
        )
    ckpt_path = Path(resume) / "checkpoints" / "model_latest.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        cfg["effective_batch_size"] = cfg.training.batch_size
        cfg["gpu_batch_size"] = cfg.training.batch_size

    from accelerate import Accelerator
    trainer.accelerator = Accelerator()
    trainer.device = trainer.accelerator.device
    trainer.epoch = 0
    trainer.video_save = False
    trainer.video_save_every = 999
    trainer.video_max_rollouts = 0
    trainer.video_fps = None
    trainer.video_resize = None
    trainer.video_writer = None
    trainer.base_path = os.path.dirname(os.path.abspath(__file__))
    trainer.train_encoder = cfg.model.train_encoder
    trainer.train_predictor = cfg.model.train_predictor
    trainer.train_decoder = cfg.model.train_decoder
    trainer.train_decoder_grad = cfg.model.train_decoder_grad
    trainer.train_emb_decoder_grad = cfg.model.train_emb_decoder_grad
    trainer.num_reconstruct_samples = cfg.training.num_reconstruct_samples
    trainer.total_epochs = cfg.training.epochs
    trainer._reported_grad_presence = False

    # Build save keys (same as Trainer.__init__)
    trainer._keys_to_save = ["epoch"]
    trainer._keys_to_save += ["encoder", "encoder_optimizer"] if trainer.train_encoder else []
    trainer._keys_to_save += ["predictor", "predictor_optimizer"] if trainer.train_predictor and cfg.has_predictor else []
    trainer._keys_to_save += ["decoder", "decoder_optimizer"] if trainer.train_decoder else []
    trainer._keys_to_save += ["emb_decoder", "emb_decoder_optimizer"] if trainer.train_decoder else []
    trainer._keys_to_save += [
        "action_encoder", "proprio_encoder",
        "post_concat_projection", "post_concat_projection_optimizer",
        "alignment_projection_optimizer",
    ]

    # Load dataset
    from utils import seed as set_seed
    set_seed(cfg.training.seed)
    trainer.datasets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    trainer.train_traj_dset = traj_dsets["train"]
    trainer.val_traj_dset = traj_dsets.get("valid", None)

    # Stub wandb_run
    class _FakeRun:
        def watch(self, *a, **kw): pass
    trainer.wandb_run = _FakeRun()

    # Init models — this internally calls load_ckpt(resume_folder) at the end
    trainer.encoder = None
    trainer.action_encoder = None
    trainer.proprio_encoder = None
    trainer.predictor = None
    trainer.emb_decoder = None
    trainer.decoder = None
    trainer.init_models()

    if hasattr(trainer, 'task_objectives') and len(trainer.task_objectives) > 0:
        trainer._keys_to_save += ["task_objectives"]

    log.info(f"Loaded epoch {trainer.epoch} from {ckpt_path}")
    trainer.model.eval()
    return trainer


@hydra.main(config_path="conf", version_base="1.1")
def main(cfg):
    orig_cwd = hydra.utils.get_original_cwd()

    # Output dir: user-specified or auto-generated
    output_dir = cfg.get("output_dir", None)
    if output_dir is None:
        output_dir = f"rollout_results/{cfg.env.name}"
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(orig_cwd, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    num_per_task = int(cfg.get("num_per_task", 10))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frameskip = cfg.frameskip
    num_hist = cfg.num_hist
    min_horizon = 2 + num_hist

    log.info(f"Output: {output_dir}")
    log.info(f"Num per task: {num_per_task}, Device: {device}")

    trainer = build_trainer(cfg)
    model = trainer.model
    env_name = cfg.env.name
    fps = max(1, int(30 / max(1, frameskip)))

    # --- Rollout logic (from train_uwm.py openloop_rollout) ---
    for split_name, dset in [("val", trainer.val_traj_dset), ("train", trainer.train_traj_dset)]:
        if dset is None or len(dset) == 0:
            continue
        log.info(f"=== {split_name} ({len(dset)} trajectories) ===")

        is_multitask = hasattr(dset, 'num_tasks') and dset.num_tasks > 1

        # Build task ranges (same as openloop_rollout task-balanced sampling)
        if is_multitask:
            task_ranges = []
            for t in range(dset.num_tasks):
                t_start = dset._cumulative[t]
                t_end = dset._cumulative[t + 1] - 1
                task_ranges.append((dset.task_names[t], t_start, t_end))
        else:
            # Single-task: use env name as task name
            task_ranges = [(env_name, 0, len(dset) - 1)]

        for tname, t_start, t_end in task_ranges:
            n_avail = t_end - t_start + 1
            n_sample = min(num_per_task, n_avail)
            if n_sample <= 0:
                continue

            indices = np.linspace(t_start, t_end, n_sample).astype(int)
            task_dir = os.path.join(output_dir, split_name, tname)
            os.makedirs(task_dir, exist_ok=True)

            for i, traj_idx in enumerate(indices):
                # --- Exactly matching openloop_rollout ---
                obs, act, state, _ = dset[traj_idx]
                act = act.to(device)

                # Full trajectory (rand_start_end=False)
                start = 0
                horizon = (obs["visual"].shape[0] - 1) // frameskip

                if horizon < min_horizon:
                    log.warning(f"  [{tname}_{i:02d}] too short (horizon={horizon}), skip")
                    continue

                # Slice obs and act (openloop_rollout lines 1544-1550)
                for k in obs.keys():
                    obs[k] = obs[k][start : start + horizon * frameskip + 1 : frameskip]
                act = act[start : start + horizon * frameskip]

                # Rearrange actions (openloop_rollout line 1552)
                act = rearrange(act, "(h f) d -> h (f d)", f=frameskip)

                obs_g = {k: v.unsqueeze(0).to(device) for k, v in obs.items()}
                actions = act.unsqueeze(0).to(device)

                with torch.no_grad():
                    # Context frames (openloop_rollout line 1568-1572)
                    obs_0 = {k: v[:, :num_hist] for k, v in obs_g.items()}

                    # Rollout (openloop_rollout line 1574)
                    z_dct, z = model.rollout(obs_0, actions)

                    if cfg.has_decoder:
                        # Decode (openloop_rollout lines 1589-1598)
                        z_emb = model.emb_decoder(z_dct["projected"])
                        pred_obs, _ = model.decode(z_emb)
                        pred_visual = pred_obs["visual"][0].cpu()
                        gt_visual = obs["visual"]  # already frameskip-sliced

                        if pred_visual.shape[-2:] != gt_visual.shape[-2:]:
                            pred_visual = torch.nn.functional.interpolate(
                                pred_visual,
                                size=gt_visual.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )

                        T = min(gt_visual.shape[0], pred_visual.shape[0])
                        gt_visual = gt_visual[:T]
                        pred_visual = pred_visual[:T]

                        # Save individual MP4: GT on top, pred on bottom
                        frames = [make_frame(gt_visual[t], pred_visual[t]) for t in range(T)]
                        vid_path = os.path.join(task_dir, f"{tname}_{i:02d}.mp4")
                        writer = imageio.get_writer(vid_path, fps=fps, codec="libx264", quality=8)
                        for f in frames:
                            writer.append_data(f)
                        writer.close()

                        log.info(f"  [{tname}_{i:02d}] {T} frames -> {vid_path}")
                    else:
                        log.info(f"  [{tname}_{i:02d}] no decoder, skip video")

            log.info(f"  [{tname}] done ({n_sample} rollouts)")

    log.info(f"All done! Videos at: {output_dir}")


if __name__ == "__main__":
    main()
