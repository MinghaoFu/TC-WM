import os
import json
import time
import queue
import hydra
import torch
import torch.nn as nn
import wandb
wandb.login()  # reads ~/.netrc; set via 
import logging
import warnings
import threading
import itertools
import numpy as np
import imageio
import h5py
import subprocess
import sys
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from utils.gpu import auto_select_gpus
import functools

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class _Tee:
    """Write to multiple file-like objects simultaneously (e.g. stdout + log file)."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()
    def fileno(self):
        return self.files[0].fileno()

# Hydra resolver to add a dash only when a description is provided.
OmegaConf.register_new_resolver("maybe_suffix", lambda s: f"-{s}" if s else "")
# Resolve output name for ckpt paths: use dataset_name for mmbench, else env name.
OmegaConf.register_new_resolver(
    "env_output_name",
    lambda env_name, dataset_name: dataset_name if env_name == "mmbench" and dataset_name else env_name,
)

# Rollout helpers for multiprocessing sampling.
_ROLLOUT_DSET = None

def _init_rollout_worker(dset):
    global _ROLLOUT_DSET
    import decord
    decord.bridge.set_bridge("torch")  # threads don't inherit module-level bridge setting
    _ROLLOUT_DSET = dset

def _prepare_rollout_sample(args):
    traj_idx, rand_start_end, min_horizon, frameskip, seed = args
    rng = np.random.RandomState(seed)
    valid_traj = False
    while not valid_traj:
        obs, act, state, _ = _ROLLOUT_DSET[traj_idx]
        if rand_start_end:
            if obs["visual"].shape[0] > min_horizon * frameskip + 1:
                start = rng.randint(
                    0, obs["visual"].shape[0] - min_horizon * frameskip - 1
                )
            else:
                start = 0
            max_horizon = (obs["visual"].shape[0] - start - 1) // frameskip
            if max_horizon > min_horizon:
                valid_traj = True
                horizon = rng.randint(min_horizon, max_horizon + 1)
        else:
            valid_traj = True
            start = 0
            horizon = (obs["visual"].shape[0] - 1) // frameskip
    for k in obs.keys():
        obs[k] = obs[k][start : start + horizon * frameskip + 1 : frameskip]
    act = act[start : start + horizon * frameskip]
    return traj_idx, start, horizon, obs, act, state

def _render_header_bar(width_px, label, bg_rgb, bar_h=28):
    """Render a colored header bar (bar_h, width_px, 3) uint8 with centered white label text."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (width_px, bar_h), tuple(int(c) for c in bg_rgb))
        draw = ImageDraw.Draw(img)
        font = None
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, 15)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = max(10, (width_px - tw) // 2)
        ty = max(0, (bar_h - th) // 2)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
        import numpy as _np
        return _np.array(img)
    except Exception:
        import numpy as _np
        bar = _np.full((bar_h, width_px, 3), [int(c) for c in bg_rgb], dtype=_np.uint8)
        return bar


def _save_image_async(args):
    imgs, num_columns, img_name = args
    os.makedirs(os.path.dirname(img_name), exist_ok=True)
    utils.save_image(
        imgs,
        img_name,
        nrow=num_columns,
        normalize=True,
        value_range=(-1, 1),
    )


class AsyncVideoWriter:
    def __init__(self, max_queue=8):
        self.queue = queue.Queue(maxsize=max_queue)
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            path, frames, fps = item
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
            try:
                for frame in frames:
                    writer.append_data(frame)
            finally:
                writer.close()
                self.queue.task_done()

    def enqueue(self, path, frames, fps):
        try:
            self.queue.put_nowait((path, frames, fps))
        except queue.Full:
            log.warning("Video queue full, dropping video: %s", path)

    def flush(self):
        """Block until all enqueued videos are written.

        Must be called before any os.fork() (e.g. DataLoader with num_workers>0)
        to avoid inheriting held locks from the writer thread into forked children,
        which causes a deadlock.
        """
        self.queue.join()

    def close(self):
        try:
            self.queue.put_nowait(None)
            self.worker.join(timeout=10)
        except Exception:
            pass


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.video_save = getattr(cfg, "video_save", False)
        self.video_save_every = getattr(cfg, "video_save_every", 1)
        self.video_max_rollouts = getattr(cfg, "video_max_rollouts", 2)
        self.video_fps = getattr(cfg, "video_fps", None)
        self.video_resize = getattr(cfg, "video_resize", None)
        self.video_writer = AsyncVideoWriter(
            max_queue=getattr(cfg, "video_queue_size", 8)
        )
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        # Redirect all IO (print / tqdm / logging) to save_dir/train.log as well
        # NOTE: We must NOT replace sys.stdout/stderr with _Tee because
        # multiprocessing (DataLoader workers) inherits them on fork, and
        # the shared _log_file fd causes resource_sharer deadlocks.
        # Instead, use a logging FileHandler for structured logs only.
        _log_path = os.path.join(cfg["saved_folder"], "train.log")
        _fh = logging.FileHandler(_log_path, mode="w")
        _fh.setLevel(logging.DEBUG)
        _fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(_fh)
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        # Check if we should use multiple GPUs (only for SLURM multirun)
        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()

        # Check CUDA availability after environment variable is set
        if torch.cuda.is_available():
            log.info(f"CUDA is available. GPU count: {torch.cuda.device_count()}")
            # Force using the first available GPU (which is GPU 0 after CUDA_VISIBLE_DEVICES is set)
            self.accelerator = Accelerator(log_with="wandb", device_placement=True)
        else:
            log.warning("CUDA is not available! Running on CPU.")
            self.accelerator = Accelerator(log_with="wandb", cpu=True)
        
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # Set wandb API key and login
            os.environ.setdefault("WANDB_API_KEY", "")  # set externally if needed
            wandb.login()  # reads ~/.netrc; set via 
            
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="minimal_wm",
                    entity="minghao_workaholic",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="minimal_wm",
                    entity="minghao_workaholic", # TODO: add entitiy to all the wandb init in the code
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        data_path_log = self.cfg.env.dataset.get("data_path", "(multi-task)")
        log.info(f"Loading dataset from {data_path_log} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.emb_decoder = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        self.train_decoder_grad = self.cfg.model.train_decoder_grad
        self.train_emb_decoder_grad = self.cfg.model.train_emb_decoder_grad
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += (
            ["emb_decoder", "emb_decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += [
            "action_encoder",
            "proprio_encoder",
            "post_concat_projection",
            "post_concat_projection_optimizer",
            "alignment_projection_optimizer",
        ]
        # select all the keys that are neural network modules
        
        
        self.init_models()

        # Add task_objectives to save keys after init_models
        if hasattr(self, 'task_objectives') and len(self.task_objectives) > 0:
            self._keys_to_save += ["task_objectives"]
        self.init_optimizers()

        self.epoch_log = OrderedDict()
        self._reported_grad_presence = False

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            # Keep only the most recent 10 epoch checkpoints (plus model_latest.pth)
            existing = sorted(
                [p for p in Path("checkpoints").glob("model_[0-9]*.pth")],
                key=lambda p: int(p.stem.split("_")[1]),
            )
            for old in existing[:-10]:
                old.unlink()
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename, weights_only=False)
        for k, v in ckpt.items():
            if isinstance(v, torch.nn.Module) and hasattr(self.model, k):
                cur = getattr(self.model, k)
                if type(cur) == type(v):
                    # Same class: try full replacement, fall back to strict=False on shape mismatch
                    try:
                        cur.load_state_dict(v.state_dict(), strict=True)
                        self.__dict__[k] = v
                        setattr(self.model, k, v)
                    except RuntimeError:
                        # Shape mismatch (e.g. predictor pos_embedding changed after adding tokens)
                        missing, unexpected = cur.load_state_dict(v.state_dict(), strict=False)
                        log.info(
                            "ckpt '%s' same class but shape mismatch, loaded strict=False "
                            "(missing=%s, unexpected=%s)",
                            k, missing, unexpected,
                        )
                        self.__dict__[k] = cur  # keep current module (weights already updated in-place)
                else:
                    # Class mismatch (e.g. decoder variant changed): load compatible weights
                    missing, unexpected = cur.load_state_dict(v.state_dict(), strict=False)
                    log.info(
                        "ckpt '%s' class mismatch (%s → %s), loaded state_dict strict=False "
                        "(missing=%s, unexpected=%s)",
                        k, type(v).__name__, type(cur).__name__, missing, unexpected,
                    )
                    self.__dict__[k] = cur  # keep current module (weights already updated in-place)
            else:
                self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        # Initialize encoder first
        if not hasattr(self, 'encoder') or self.encoder is None:
            if hasattr(self.cfg, "encoder") and str(self.cfg.encoder.get("_target_", "")).endswith("CosmosCIEncoder"):
                with open_dict(self.cfg):
                    for key in ("name", "feature_key"):
                        if key in self.cfg.encoder:
                            del self.cfg.encoder[key]
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if hasattr(self.encoder, "emb_dim") and self.encoder.emb_dim != self.cfg.encoder_emb_dim:
            with open_dict(self.cfg):
                self.cfg.encoder_emb_dim = self.encoder.emb_dim
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        # Get emb_dim BEFORE wrapping with accelerate - avoid DDP attribute access issues  
        encoder_emb_dim = self.encoder.emb_dim
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        # Get emb_dim BEFORE wrapping with accelerate - avoid DDP attribute access issues
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")
        
        self.projection_input_dim = self.cfg.encoder_emb_dim + self.cfg.proprio_emb_dim * self.cfg.num_proprio_repeat
        projector_target = str(self.cfg.projector.get("_target_", ""))
        if projector_target == "torch.nn.Identity":
            self.post_concat_projection = hydra.utils.instantiate(self.cfg.projector)
        else:
            self.post_concat_projection = hydra.utils.instantiate(
                self.cfg.projector,
                visual_emb_dim=self.cfg.encoder_emb_dim,
                proprio_emb_dim=self.cfg.proprio_emb_dim * self.cfg.num_proprio_repeat,
                projected_dim=self.cfg.projected_dim,
            )
        self.post_concat_projection = self.accelerator.prepare(self.post_concat_projection)

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # compute predictor dims early (used in dinowm identity projector case)
        raw_proprio_dim = self.datasets["train"].proprio_dim

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            if (
                getattr(self.encoder, "name", "") in ("cosmos_ci", "wan_vae")
                and hasattr(self.encoder, "patch_size")
                and self.encoder.patch_size
            ):
                num_side_patches = self.cfg.img_size // self.encoder.patch_size
            else:
                decoder_scale = 16  # from vqvae
                num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        # Token-conditioning models (MTJEPATokenWorldModel): action and task are
        # separate tokens in the sequence, not tiled into the feature dim.
        _model_target = str(self.cfg.model.get("_target_", ""))
        token_conditioning = "TokenWorldModel" in _model_target
        if token_conditioning:
            num_patches += 3  # action token + task token + proprio token

        if self.cfg.has_predictor:
            if self.predictor is None:
                projector_target = str(self.cfg.projector.get("_target_", ""))
                task_proj_dim = getattr(self.cfg, "task_proj_dim", 0) or 0
                if token_conditioning:
                    # All tokens are encoder_emb_dim; no action/proprio in feature dim
                    predictor_dim = self.cfg.encoder_emb_dim
                elif projector_target == "torch.nn.Identity":
                    predictor_dim = (
                        self.cfg.encoder_emb_dim
                        + action_emb_dim * self.cfg.num_action_repeat
                        + self.cfg.proprio_emb_dim * self.cfg.num_proprio_repeat
                        + task_proj_dim
                    )
                else:
                    predictor_dim = (
                        self.cfg.projected_dim
                        + action_emb_dim * self.cfg.num_action_repeat
                        + raw_proprio_dim
                        + task_proj_dim
                    )
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=predictor_dim,
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path, weights_only=False)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path, weights_only=False)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    decoder_target = str(self.cfg.decoder.get("_target_", ""))
                    if "ActionConditionedProjectedVQVAEDecoder" in decoder_target:
                        # ACG-WM: action-conditioned emb_decoder; needs in_dim, emb_dim, action_dim
                        self.decoder = hydra.utils.instantiate(
                            self.cfg.decoder,
                            in_dim=self.cfg.projected_dim,
                            emb_dim=self.cfg.encoder_emb_dim,
                            action_dim=self.datasets["train"].action_dim,
                        )
                    elif "ProjectedVQVAEDecoder" in decoder_target:
                        # Two-stage decoder: needs both in_dim and emb_dim
                        self.decoder = hydra.utils.instantiate(
                            self.cfg.decoder,
                            in_dim=self.cfg.projected_dim,
                            emb_dim=self.cfg.encoder_emb_dim,
                        )
                    else:
                        self.decoder = hydra.utils.instantiate(
                            self.cfg.decoder,
                            emb_dim=self.cfg.encoder_emb_dim,
                        )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
                    
        # initialize decoder to embedding space
        emb_decoder_target = str(self.cfg.emb_decoder.get("_target_", ""))
        if emb_decoder_target == "models.decoder.emb_decoder_identity.EmbDecoderIdentity":
            self.emb_decoder = hydra.utils.instantiate(self.cfg.emb_decoder)
        else:
            self.emb_decoder = hydra.utils.instantiate(
                self.cfg.emb_decoder,
                in_dim=self.cfg.projected_dim,
                out_dim=self.cfg.encoder_emb_dim,
            )
                    
        # --- Build task_objectives from cfg.method.objectives ---
        _objectives = []
        self.alignment_projection = None  # backward-compat pointer

        if hasattr(self.cfg, "method") and hasattr(self.cfg.method, "objectives"):
            # Runtime dims available for partial constructors
            _runtime_kwargs = {}
            if hasattr(self.datasets['train'].dataset, 'states'):
                _runtime_kwargs['proprio_dim'] = self.datasets['train'].dataset.states.shape[-1]

            for obj_cfg in self.cfg.method.objectives:
                obj = hydra.utils.instantiate(obj_cfg)
                if isinstance(obj, functools.partial):
                    obj = obj(**_runtime_kwargs)
                obj = obj.to(self.accelerator.device)
                if any(p.requires_grad for p in obj.parameters()):
                    obj = self.accelerator.prepare(obj)
                _objectives.append(obj)
                log.info(f"Initialized objective: {type(obj).__name__} (loss_key={obj.loss_key}, weight={obj.loss_weight})")

            # backward-compat pointer: alignment_projection lives inside InfoNCEAlignmentObjective
            self.alignment_projection = next(
                (getattr(o, "alignment_projection", None) for o in _objectives
                 if hasattr(o, "alignment_projection")),
                None,
            )

        self.task_objectives = nn.ModuleList(_objectives)
        self.task_objective   = self.task_objectives[0] if self.task_objectives else None

        projector_target = str(self.cfg.projector.get("_target_", ""))
        raw_proprio_dim_model = self.datasets["train"].proprio_dim
        if projector_target == "torch.nn.Identity":
            raw_proprio_dim_model = self.cfg.proprio_emb_dim * self.cfg.num_proprio_repeat

        # Per-task action dims for action masking (multi-task configs only)
        _action_dims = None
        _train_dset = self.datasets["train"]
        if hasattr(_train_dset, "dataset") and hasattr(_train_dset.dataset, "sub_datasets"):
            _action_dims = [ds.action_dim for ds in _train_dset.dataset.sub_datasets]
            log.info(f"Multi-task action dims: {_action_dims} (passed to model for action masking)")

        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            train_decoder_grad=self.train_decoder_grad,
            train_emb_decoder_grad=self.train_emb_decoder_grad,
            emb_decoder=self.emb_decoder,
            post_concat_projection=self.post_concat_projection,
            alignment_projection=None,   # now owned by objectives
            task_objectives=list(self.task_objectives),
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            raw_proprio_dim=raw_proprio_dim_model,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            open_alignment=self.alignment_projection is not None,
            action_dims=_action_dims,
        )
        
        self.encoder, self.predictor, self.decoder, self.emb_decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder, self.emb_decoder
        )
        
        if hasattr(self.cfg, 'alignment'):
            if hasattr(self.cfg.alignment, 'state_consistency_loss_weight'):
                self.model.state_consistency_loss_weight = self.cfg.alignment.state_consistency_loss_weight
            if hasattr(self.cfg.alignment, 'alignment_regularization'):
                self.model.alignment_regularization = self.cfg.alignment.alignment_regularization
            if hasattr(self.cfg.alignment, 'latent_dynamics_loss_weight'):
                self.model.latent_dynamics_loss_weight = self.cfg.alignment.latent_dynamics_loss_weight
            elif hasattr(self.cfg.alignment, 'dynamics_7d_loss_weight'):  # Backward compatibility
                self.model.latent_dynamics_loss_weight = self.cfg.alignment.dynamics_7d_loss_weight
            else:
                self.model.latent_dynamics_loss_weight = 0.0  # Default: disabled
            
            # Dynamic ratio parameter
            if hasattr(self.cfg.alignment, 'dynamic_ratio'):
                self.model.dynamic_ratio = self.cfg.alignment.dynamic_ratio
            else:
                self.model.dynamic_ratio = 1.0  # Default: all 7 dims are dynamic
                
            # Alignment feature selection method
            if hasattr(self.cfg.alignment, 'alignment_feature_selection'):
                self.model.alignment_feature_selection = self.cfg.alignment.alignment_feature_selection
            else:
                self.model.alignment_feature_selection = 'isolate'  # Default: isolate first 64 dims

        if hasattr(self.cfg, "proprio_pred"):
            if hasattr(self.cfg.proprio_pred, "loss_type"):
                self.model.proprio_pred_loss_type = self.cfg.proprio_pred.loss_type
            if hasattr(self.cfg.proprio_pred, "space"):
                self.model.proprio_pred_space = self.cfg.proprio_pred.space
            if hasattr(self.cfg.proprio_pred, "weight"):
                self.model.proprio_pred_loss_weight = self.cfg.proprio_pred.weight
            if hasattr(self.cfg.proprio_pred, "infonce_temp"):
                self.model.proprio_infonce_temp = self.cfg.proprio_pred.infonce_temp
        self.model.emb_recon_weight = getattr(self.cfg, "emb_recon_weight", 1.0)
        
        self.model = self.model.to(self.accelerator.device)
        # open_alignment: True iff any objective owns an alignment_projection
        self.model.open_alignment = self.alignment_projection is not None
        model_ckpt = Path(self.cfg.resume_folder if self.cfg.resume_folder is not None else self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        
        if self.cfg.has_predictor:
            self.predictor_optimizer = torch.optim.AdamW(
                self.predictor.parameters(),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)
        else:
            self.decoder_optimizer = None

        if any(p.requires_grad for p in self.emb_decoder.parameters()):
            self.emb_decoder_optimizer = torch.optim.Adam(
                self.emb_decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.emb_decoder_optimizer = self.accelerator.prepare(self.emb_decoder_optimizer)
        else:
            self.emb_decoder_optimizer = None

        if any(p.requires_grad for p in self.post_concat_projection.parameters()):
            self.post_concat_projection_optimizer = torch.optim.Adam(
                self.post_concat_projection.parameters(),
                lr=self.cfg.training.decoder_lr,
            )
            self.post_concat_projection_optimizer = self.accelerator.prepare(
                self.post_concat_projection_optimizer
            )
        else:
            self.post_concat_projection_optimizer = None

        # Optimizer for all learnable objective parameters
        _obj_params = [p for o in self.task_objectives for p in o.parameters() if p.requires_grad]
        if _obj_params:
            self.alignment_projection_optimizer = torch.optim.Adam(
                _obj_params,
                lr=self.cfg.training.decoder_lr,
            )
            self.alignment_projection_optimizer = self.accelerator.prepare(
                self.alignment_projection_optimizer
            )
        else:
            self.alignment_projection_optimizer = None

        self.report_trainable_modules()

    def report_trainable_modules(self):
        if not self.accelerator.is_main_process:
            return
        modules = [
            ("encoder", self.encoder, self.encoder_optimizer),
            ("predictor", self.predictor, getattr(self, "predictor_optimizer", None)),
            ("post_concat_projection", self.post_concat_projection, self.post_concat_projection_optimizer),
            ("emb_decoder", self.emb_decoder, self.emb_decoder_optimizer),
            ("decoder", self.decoder, self.decoder_optimizer),
            ("action_encoder", self.action_encoder, getattr(self, "action_encoder_optimizer", None)),
            ("proprio_encoder", self.proprio_encoder, getattr(self, "action_encoder_optimizer", None)),
            ("alignment_projection", self.alignment_projection, self.alignment_projection_optimizer),
        ]
        print("Trainable module status:")
        for name, module, optimizer in modules:
            if module is None:
                print(f"- {name}: module=None")
                continue
            params = list(module.parameters())
            trainable = [p for p in params if p.requires_grad]
            has_optimizer = optimizer is not None
            print(
                f"- {name}: trainable_params={len(trainable)}/{len(params)}, "
                f"optimizer={'yes' if has_optimizer else 'no'}"
            )

    def report_grad_presence(self):
        if not self.accelerator.is_main_process or self._reported_grad_presence:
            return
        modules = [
            ("encoder", self.encoder),
            ("predictor", self.predictor),
            ("post_concat_projection", self.post_concat_projection),
            ("emb_decoder", self.emb_decoder),
            ("decoder", self.decoder),
            ("action_encoder", self.action_encoder),
            ("proprio_encoder", self.proprio_encoder),
            ("alignment_projection", self.alignment_projection),
        ]
        print("Gradient presence after backward:")
        for name, module in modules:
            if module is None:
                print(f"- {name}: module=None")
                continue
            has_grad = any(p.grad is not None for p in module.parameters())
            print(f"- {name}: grad={'yes' if has_grad else 'no'}")
        self._reported_grad_presence = True

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        export_cfg = getattr(self.cfg, "latent_export", None)
        if export_cfg and export_cfg.enabled:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.export_latents()
            self.accelerator.wait_for_everyone()
            return

        if getattr(self.cfg, "val_only", False):
            log.info("val_only mode: running evaluation and rollout only")
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            return

        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        init_epoch = self.epoch + 1  # epoch starts from 1
        
        # Run initial evaluation before training (epoch 0)
        if init_epoch == 1:  # Only run initial eval if starting from scratch
            self.epoch = 0
            log.info("Running initial evaluation before training...")
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
        
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only: launch planning jobs on the saved ckpt
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)
        if self.video_writer is not None:
            self.video_writer.close()

    def export_latents(self):
        """
        Export projected visual latents to an HDF5 file compatible with the latent diffusion planner.
        """
        export_cfg = self.cfg.latent_export
        resume_root = self.cfg.resume_folder
        if resume_root in (None, "none", "", "null"):
            raise ValueError("latent_export.enabled=true requires resume_folder to point to a trained model directory.")

        ckpt_path = Path(resume_root) / "checkpoints" / "model_latest.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

        output_path = export_cfg.output_path
        if output_path in (None, "none", "", "null"):
            output_path = Path(self.cfg.saved_folder) / "latent.hdf5"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        splits = list(export_cfg.splits)
        rgb_key = export_cfg.rgb_key
        grid_size = export_cfg.latent_grid_size
        if grid_size <= 0:
            raise ValueError(f"latent_grid_size must be positive, got {grid_size}.")

        mode = getattr(export_cfg, "mode", "trajectory").lower()
        if mode not in ("trajectory", "slice"):
            raise ValueError(f"latent_export.mode must be 'trajectory' or 'slice', got {mode}.")

        if mode == "trajectory":
            dataset_map = {
                "train": self.train_traj_dset,
                "valid": self.val_traj_dset,
            }
        else:
            dataset_map = {
                "train": self.datasets["train"],
                "valid": self.datasets["valid"],
            }

        # Ensure model weights are loaded from resume_folder checkpoint
        # Load checkpoint weights (ensures latest parameters even if init_models already restored them)
        self.load_ckpt(ckpt_path)
        module_names = [
            "encoder",
            "proprio_encoder",
            "action_encoder",
            "decoder",
            "post_concat_projection",
            "alignment_projection",
            "predictor",
        ]
        for name in module_names:
            module = getattr(self, name, None)
            if module is None:
                continue
            module = module.to(self.device)
            setattr(self, name, module)
            if hasattr(self.model, name):
                setattr(self.model, name, module)

        self.model.eval()
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        total_eps = 0
        min_z = None
        max_z = None

        with h5py.File(output_path, "w") as writer:
            data_grp = writer.create_group("data")
            data_grp.attrs["mode"] = mode
            for split in splits:
                dataset = dataset_map.get(split)
                if dataset is None:
                    log.warning(f"Unknown split '{split}' requested for latent export; skipping.")
                    continue

                for idx in tqdm(range(len(dataset)), desc=f"Exporting {split} latents"):
                    if mode == "trajectory":
                        obs, _, _, meta = dataset[idx]
                    else:
                        obs, _, _ = dataset[idx]
                        meta = {}
                    if "visual" not in obs:
                        raise KeyError("Observation dictionary missing 'visual' key required for latent export.")

                    obs_device = {
                        key: value.unsqueeze(0).to(self.device)
                        for key, value in obs.items()
                    }

                    with torch.no_grad():
                        projected = self.model.encode_obs_projected(obs_device)["projected"]  # (1, T, P, D)
                    projected = projected.squeeze(0).cpu().numpy()  # (T, P, D)

                    T, P, D = projected.shape
                    expected_patches = grid_size * grid_size
                    if P != expected_patches:
                        raise ValueError(
                            f"Projected feature count {P} does not match latent_grid_size {grid_size} (expected {expected_patches})."
                        )

                    latents = projected.reshape(T, grid_size, grid_size, D).transpose(0, 3, 1, 2)  # (T, D, H, W)

                    if mode == "trajectory":
                        group_name = f"{split}_episode_{idx:05d}"
                    else:
                        group_name = f"{split}_slice_{idx:06d}"
                    ep_grp = data_grp.create_group(group_name)
                    latent_grp = ep_grp.create_group("latent")
                    latent_grp.create_dataset(rgb_key, data=latents.astype(np.float32))

                    episode_min = float(latents.min())
                    episode_max = float(latents.max())
                    min_z = episode_min if min_z is None else min(min_z, episode_min)
                    max_z = episode_max if max_z is None else max(max_z, episode_max)
                    total_eps += 1

                    if meta and "shape" in meta:
                        ep_grp.attrs["shape"] = meta["shape"]

            data_grp.attrs["total"] = total_eps
            if min_z is not None and max_z is not None:
                data_grp.attrs["min_z"] = min_z
                data_grp.attrs["max_z"] = max_z

        torch.set_grad_enabled(prev_grad_state)
        log.info(f"Saved projected latents for {total_eps} episodes to {output_path}")

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        common_keys = z_pred.keys() & z_tgt.keys()
        for k in common_keys:
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        val_every_x_steps = getattr(self.cfg, "val_every_x_steps", None)
        val_max_batches = getattr(self.cfg, "val_max_batches", None)
        if isinstance(val_every_x_steps, str):
            val_every_x_steps = int(val_every_x_steps)
        if isinstance(val_max_batches, str):
            val_max_batches = int(val_max_batches)
        if val_every_x_steps is not None and val_every_x_steps < 1:
            val_every_x_steps = None
        if val_max_batches is not None and val_max_batches < 1:
            val_max_batches = None

        for i, data in enumerate(
            tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            
            # Pass state for alignment loss
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act, state
            )

            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
                if self.post_concat_projection_optimizer is not None:
                    self.post_concat_projection_optimizer.zero_grad()
                if self.emb_decoder_optimizer is not None:
                    self.emb_decoder_optimizer.zero_grad()
            if self.alignment_projection_optimizer is not None:
                self.alignment_projection_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            loss = loss.float()
            self.accelerator.backward(loss)
            self.report_grad_presence()

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
                if self.emb_decoder_optimizer is not None:
                    self.emb_decoder_optimizer.step()
                if self.post_concat_projection_optimizer is not None:
                    self.post_concat_projection_optimizer.step()
            if self.alignment_projection_optimizer is not None:
                self.alignment_projection_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            
            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    # Use projected features like in forward pass
                    z_gt, _ = self.model.encode(obs, act)
                    z_gt_projected = z_gt[:, :, :, :self.model.projected_dim]
                    z_tgt = slice_trajdict_with_t(
                        {"projected": z_gt_projected},
                        start_idx=self.model.num_pred,
                    )

                    z_obs_out = {
                        "projected": z_out[:, :, :, :self.model.projected_dim]
                    }

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            # Convert loss_components to the format expected by logs_update
            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)
            
            # Direct wandb logging for debugging
            if self.accelerator.is_main_process:
                wandb_log_dict = {f"train_{k}": v[0] for k, v in loss_components.items()}
                wandb_log_dict["epoch"] = self.epoch
                wandb_log_dict["step"] = i
                self.wandb_run.log(wandb_log_dict)

            if (
                val_every_x_steps is not None
                and (i + 1) % val_every_x_steps == 0
            ):
                log.info(
                    "Step-based val at train step %d (every %d)",
                    i + 1,
                    val_every_x_steps,
                )
                self.accelerator.wait_for_everyone()
                self.val(max_batches=val_max_batches, do_rollout=False)
                self.accelerator.wait_for_everyone()

    def val(self, max_batches=None, do_rollout=True):
        self.model.eval()
        val_max_batches = max_batches
        if val_max_batches is None:
            val_max_batches = getattr(self.cfg, "val_max_batches", None)
        if isinstance(val_max_batches, str):
            val_max_batches = int(val_max_batches)
        if val_max_batches is not None and val_max_batches < 1:
            val_max_batches = None

        if do_rollout and len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            import time as _time
            with torch.no_grad():
                num_rollout = getattr(self.cfg, "rollout_num", 10)
                _t0 = _time.time()
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, num_rollout=num_rollout, mode="train"
                )
                _t1 = _time.time()
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(
                    self.val_traj_dset, num_rollout=num_rollout, mode="val"
                )
                _t2 = _time.time()
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)
                log.info(f"Rollout time: train={_t1-_t0:.1f}s, val={_t2-_t1:.1f}s, total={_t2-_t0:.1f}s")

        # Flush async video writer BEFORE the validation DataLoader forks workers.
        # The writer's daemon thread may hold numpy/imageio internal locks; forking
        # while those locks are held causes the child workers to deadlock (0% CPU).
        if self.video_writer is not None:
            self.video_writer.flush()

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act, state
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()
            

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() if isinstance(value, torch.Tensor) else value
                for key, value in loss_components.items()
                if key not in ["visual_pred", "visual_tgt", "visual_reconstructed", "visual_original"]  # Exclude visualization data from metrics
            }

            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    # Use projected features like in forward pass
                    z_gt, _ = self.model.encode(obs, act)
                    z_tgt_tensor = z_gt[
                        :, self.model.num_pred :, :, : self.model.projected_dim
                    ]
                    
                    z_obs_out = {
                        "projected": z_out[:, :, :, : self.model.projected_dim]
                    }
                    z_tgt = {"projected": z_tgt_tensor}

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            # Convert loss_components to the format expected by logs_update
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)
            
            # Direct wandb logging for debugging
            if self.accelerator.is_main_process:
                wandb_log_dict = {f"val_{k}": v[0] for k, v in loss_components.items()}
                wandb_log_dict["epoch"] = self.epoch
                wandb_log_dict["step"] = i
                self.wandb_run.log(wandb_log_dict)
            if val_max_batches is not None and (i + 1) >= val_max_batches:
                break
                
        # Save three-way reconstruction visualizations during validation
        if visual_out is not None and visual_reconstructed is not None:
            self.save_threeway_comparison(
                    visual_tgt=obs["visual"],
                    visual_pred=visual_out,
                    visual_reconstructed=visual_reconstructed,
                    epoch=self.epoch,
                    batch_idx=i,
                    phase="train"
                )

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        rollout_parallel = getattr(self.cfg, "rollout_parallel", 8)
        if isinstance(rollout_parallel, str):
            rollout_parallel = int(rollout_parallel)
        if rollout_parallel is None or rollout_parallel < 1:
            rollout_parallel = 1
        rollout_batch = getattr(self.cfg, "rollout_batch", 4)
        if isinstance(rollout_batch, str):
            rollout_batch = int(rollout_batch)
        if rollout_batch is None or rollout_batch < 1:
            rollout_batch = 1
        rollout_save_workers = getattr(self.cfg, "rollout_save_workers", 2)
        if isinstance(rollout_save_workers, str):
            rollout_save_workers = int(rollout_save_workers)
        if rollout_save_workers is None or rollout_save_workers < 0:
            rollout_save_workers = 0
        rollout_debug = getattr(self.cfg, "rollout_debug", False)
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}
        save_executor = None
        save_futures = []
        if self.accelerator.is_main_process and rollout_save_workers > 0:
            save_executor = ThreadPoolExecutor(max_workers=rollout_save_workers)
        grid_sequences = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        rollout_meta = []
        prepared_samples = None
        rollout_fixed_indices = getattr(self.cfg, "rollout_fixed_indices", None)
        rollout_uniform = getattr(self.cfg, "rollout_uniform", False)
        is_multitask = hasattr(dset, 'num_tasks') and dset.num_tasks > 1
        if rollout_fixed_indices:
            traj_indices = [int(x) for x in rollout_fixed_indices]
            if num_rollout is not None:
                traj_indices = traj_indices[:num_rollout]
        elif rollout_uniform:
            if len(dset) <= 0:
                traj_indices = []
            elif is_multitask:
                # Task-balanced: linspace within each task's episode range
                samples_per_task = num_rollout // dset.num_tasks
                traj_indices = []
                for t in range(dset.num_tasks):
                    t_start = dset._cumulative[t]
                    t_end = dset._cumulative[t + 1] - 1
                    if t_end >= t_start and samples_per_task > 0:
                        task_ls = np.linspace(t_start, t_end, samples_per_task)
                        traj_indices.extend([int(round(x)) for x in task_ls])
                num_rollout = len(traj_indices)
                if rollout_debug and self.accelerator.is_main_process:
                    log.info(
                        "Task-balanced rollout: %d tasks × %d samples = %d rollouts",
                        dset.num_tasks, samples_per_task, num_rollout,
                    )
            else:
                linspace = np.linspace(0, len(dset) - 1, num_rollout)
                traj_indices = [int(round(x)) for x in linspace]
        else:
            traj_indices = None
        if rollout_parallel > 1 and traj_indices is not None:
            try:
                # ProcessPoolExecutor: each worker has its own memory space, which is
                # required because decord VideoReaders are NOT thread-safe.
                # Add per-future timeout to avoid hanging on slow/stuck workers.
                with ProcessPoolExecutor(
                    max_workers=rollout_parallel,
                    initializer=_init_rollout_worker,
                    initargs=(dset,),
                ) as ex:
                    futures = [
                        ex.submit(
                            _prepare_rollout_sample,
                            (traj_idx, rand_start_end, min_horizon, self.cfg.frameskip, self.cfg.training.seed + i),
                        )
                        for i, traj_idx in enumerate(traj_indices)
                    ]
                    prepared_samples = [f.result(timeout=120) for f in futures]
            except Exception as exc:
                log.warning("Parallel rollout sampling failed (%s), falling back to sequential", exc)
                prepared_samples = None

        for batch_start in range(0, num_rollout, rollout_batch):
            batch_items = list(range(batch_start, min(batch_start + rollout_batch, num_rollout)))
            if rollout_debug and self.accelerator.is_main_process:
                log.info(
                    "Rollout batch %d/%d (size=%d)",
                    batch_start // rollout_batch + 1,
                    (num_rollout + rollout_batch - 1) // rollout_batch,
                    len(batch_items),
                )
            for idx in batch_items:
                valid_traj = False
                while not valid_traj:
                    if prepared_samples is not None:
                        traj_idx, start, horizon, obs, act, state = prepared_samples[idx]
                        valid_traj = True
                    else:
                        if traj_indices is not None:
                            traj_idx = traj_indices[idx]
                        else:
                            traj_idx = np.random.randint(0, len(dset))
                        obs, act, state, _ = dset[traj_idx]
                        act = act.to(self.device)
                        if rand_start_end:
                            if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                                start = np.random.randint(
                                    0,
                                    obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                                )
                            else:
                                start = 0
                            max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                            if max_horizon > min_horizon:
                                valid_traj = True
                                horizon = np.random.randint(min_horizon, max_horizon + 1)
                        else:
                            valid_traj = True
                            start = 0
                            horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip
                meta_entry = {
                    "rollout_id": idx,
                    "traj_idx": int(traj_idx),
                    "start": int(start),
                    "horizon": int(horizon),
                }
                if is_multitask:
                    task_id, _, _ = dset._resolve(int(traj_idx))
                    meta_entry["task_id"] = task_id
                    meta_entry["task_name"] = dset.task_names[task_id]
                rollout_meta.append(meta_entry)

                if prepared_samples is None:
                    for k in obs.keys():
                        obs[k] = obs[k][
                            start :
                            start + horizon * self.cfg.frameskip + 1 :
                            self.cfg.frameskip
                        ]
                    act = act[start : start + horizon * self.cfg.frameskip]

                act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

                obs_g = {}
                for k in obs.keys():
                    obs_g[k] = obs[k].unsqueeze(0).to(self.device)
                actions = act.unsqueeze(0).to(self.device)  # needed in parallel path (sequential path already moved act to device at load time)
            
            _z = self.model.encode_to_projected(obs_g, torch.nn.functional.pad(actions, (0, 0, 1, 0), mode='constant', value=0))
            
            z_g = {}
            for k in _z.keys():
                z_g[k] = _z[k][:, -1:, ...]

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_dct, z = self.model.rollout(obs_0, actions)
                z_last = slice_trajdict_with_t(z_dct, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_last, z_g)
                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    z_emb = self.model.emb_decoder(z_dct["projected"])
                    pred_obs, _ = self.model.decode(z_emb)
                    pred_visual = pred_obs["visual"][0].cpu()
                    if pred_visual.shape[-2:] != obs["visual"].shape[-2:]:
                        pred_visual = torch.nn.functional.interpolate(
                            pred_visual,
                            size=obs["visual"].shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    imgs = torch.cat([obs["visual"], pred_visual], dim=0)
                    img_path = f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png"
                    if save_executor is not None:
                        imgs_cpu = imgs.detach().cpu()
                        save_futures.append(
                            save_executor.submit(
                                _save_image_async,
                                (imgs_cpu, obs["visual"].shape[0], img_path),
                            )
                        )
                    else:
                        self.plot_imgs(
                            imgs,
                            obs["visual"].shape[0],
                            img_path,
                        )
                    if (
                        self.video_save
                        and self.accelerator.is_main_process
                        and (self.epoch % self.video_save_every == 0)
                    ):
                        grid_sequences.setdefault(postfix, []).append(
                            {
                                "idx": idx,
                                "gt": obs["visual"].cpu(),
                                "pred": pred_visual.cpu(),
                            }
                        )
        if save_executor is not None:
            for fut in as_completed(save_futures):
                _ = fut.result()
            save_executor.shutdown(wait=True)

        if self.accelerator.is_main_process:
            if (
                self.video_save
                and (self.epoch % self.video_save_every == 0)
                and grid_sequences
            ):
                fps = self.video_fps
                if fps is None:
                    fps = max(1, int(30 / max(1, self.cfg.frameskip)))
                grid_cols = getattr(self.cfg, "video_grid_cols", None)
                grid_row_gap = getattr(self.cfg, "video_grid_row_gap", 4)
                if grid_row_gap is None or grid_row_gap < 0:
                    grid_row_gap = 4
                if grid_cols is None or grid_cols < 1:
                    grid_cols = None

                # Determine method label and color for prediction rows
                _folder = getattr(self.cfg, 'saved_folder', '') or ''
                _method_map = {
                    'tcwm':   ('TCWM Rollout',    (139, 92,  246)),
                    'dinowm': ('DINO-WM Rollout', (6,  182,  212)),
                    'tdmpc2': ('TDMPC2 Rollout',  (245, 158,  11)),
                }
                _pred_label, _pred_color = 'Rollout', (99, 102, 241)
                for _mk, (_ml, _mc) in _method_map.items():
                    if _mk in _folder.lower():
                        _pred_label, _pred_color = _ml, _mc
                        break
                _gt_color = (34, 197, 94)   # green-500
                _bar_h    = 28              # label bar height (px)
                _pair_gap = max(grid_row_gap, 6)  # gap between sample pairs

                # Cache header bars — same for every frame
                _bar_cache = {}

                for postfix, entries in grid_sequences.items():
                    if not entries:
                        continue
                    entries = sorted(entries, key=lambda x: x["idx"])
                    # Limit entries for video grid (video_max_rollouts)
                    max_entries = self.video_max_rollouts if self.video_max_rollouts else len(entries)
                    entries = entries[:max_entries]
                    if grid_cols is None:
                        grid_cols = min(len(entries), 4)
                    num_cols = min(grid_cols, len(entries))
                    num_chunks = (len(entries) + num_cols - 1) // num_cols
                    max_t = max(e["gt"].shape[0] for e in entries)
                    frames = []
                    for t in range(max_t):
                        rows = []
                        for chunk_start in range(0, len(entries), num_cols):
                            chunk = entries[chunk_start : chunk_start + num_cols]
                            gt_row = []
                            pred_row = []
                            for e in chunk:
                                gt_seq = e["gt"]
                                pred_seq = e["pred"]
                                idx_t = t if t < gt_seq.shape[0] else gt_seq.shape[0] - 1
                                idx_p = t if t < pred_seq.shape[0] else pred_seq.shape[0] - 1
                                # de-normalize [-1,1] → [0,1] (dataset applies Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]))
                                gt_row.append((gt_seq[idx_t].clamp(-1, 1) + 1.0) * 0.5)
                                pred_row.append((pred_seq[idx_p].clamp(-1, 1) + 1.0) * 0.5)
                            if len(chunk) < num_cols:
                                pad_count = num_cols - len(chunk)
                                pad = torch.zeros_like(gt_row[0])
                                for _ in range(pad_count):
                                    gt_row.append(pad)
                                    pred_row.append(pad)
                            gt_row = torch.cat(gt_row, dim=2)    # (C, H, W*cols)
                            pred_row = torch.cat(pred_row, dim=2) # (C, H, W*cols)

                            # Cached colored header bars (same every frame)
                            row_w = gt_row.shape[2]
                            if row_w not in _bar_cache:
                                gt_bar_np   = _render_header_bar(row_w, "Ground Truth", _gt_color,  bar_h=_bar_h)
                                pred_bar_np = _render_header_bar(row_w, _pred_label,    _pred_color, bar_h=_bar_h)
                                _bar_cache[row_w] = (
                                    torch.from_numpy(gt_bar_np).permute(2,0,1).float().div(255.0),
                                    torch.from_numpy(pred_bar_np).permute(2,0,1).float().div(255.0),
                                )
                            gt_bar, pred_bar = _bar_cache[row_w]

                            # Stack: [GT bar | GT frames | Pred bar | Pred frames]
                            rows.append(gt_bar)
                            rows.append(gt_row)
                            rows.append(pred_bar)
                            rows.append(pred_row)

                            # Gap between sample pairs (not after the last chunk)
                            if (chunk_start + num_cols) < len(entries):
                                gap_row = torch.zeros(
                                    (gt_row.shape[0], _pair_gap, gt_row.shape[2]),
                                    dtype=gt_row.dtype,
                                )
                                rows.append(gap_row)
                        grid = torch.cat(rows, dim=1)
                        if self.video_resize is not None:
                            # Scale width to video_resize * num_cols; height proportionally
                            target_w = self.video_resize * num_cols
                            target_h = int(round(grid.shape[1] * target_w / max(grid.shape[2], 1)))
                            grid = torch.nn.functional.interpolate(
                                grid.unsqueeze(0),
                                size=(target_h, target_w),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)
                        frame = (
                            (grid.clamp(0, 1) * 255)
                            .to(torch.uint8)
                            .permute(1, 2, 0)
                            .numpy()
                        )
                        frames.append(frame)
                    video_path = os.path.join(
                        plotting_dir, f"e{self.epoch}_{mode}_grid{postfix}.mp4"
                    )
                    self.video_writer.enqueue(video_path, frames, fps)
            meta_path = f"{plotting_dir}/e{self.epoch}_{mode}_rollout_indices.json"
            meta = {
                "epoch": int(self.epoch),
                "mode": mode,
                "seed": int(self.cfg.training.seed),
                "frameskip": int(self.cfg.frameskip),
                "num_hist": int(self.cfg.num_hist),
                "min_horizon": int(min_horizon),
                "rollouts": rollout_meta,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # Keep only the most recent 10 rollout epoch directories
        if self.accelerator.is_main_process:
            import shutil
            rollout_dirs = sorted(
                [p for p in Path("rollout_plots").iterdir() if p.is_dir() and p.name.startswith("e")],
                key=lambda p: int(p.name.lstrip("e").split("_")[0]),
            )
            for old_dir in rollout_dirs[:-10]:
                shutil.rmtree(old_dir)

        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        
        # Log training and validation losses if they exist
        train_loss = epoch_log.get('train_loss', 0.0)
        val_loss = epoch_log.get('val_loss', 0.0)
        log.info(f"Epoch {self.epoch}  Training loss: {train_loss:.4f}  \
                Validation loss: {val_loss:.4f}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    def save_threeway_comparison(self, visual_tgt, visual_pred, visual_reconstructed, epoch, batch_idx, phase="train"):
        """Save three-way comparison: ground_truth vs recon_from_pred vs recon_from_encoded"""
        if not self.accelerator.is_main_process:
            return
        from torchvision.utils import save_image
        num_samples = min(6, visual_tgt.shape[0])
        num_frames = visual_pred.shape[1]
        visual_tgt = visual_tgt[:num_samples]
        visual_pred = visual_pred[:num_samples]
        visual_reconstructed = visual_reconstructed[:num_samples]
        # Create comparison grid
        comparison = []
        for i in range(num_samples):
            for t in range(num_frames):
                comparison.append(visual_tgt[i, t+1])           # Ground truth
                comparison.append(visual_pred[i, t])          # Prediction
                comparison.append(visual_reconstructed[i, t+1]) # Reconstruction

        comparison_tensor = torch.stack(comparison, dim=0)

        # Normalize to [0, 1]
        if comparison_tensor.min() < 0:
            comparison_tensor = (comparison_tensor + 1) / 2
        elif comparison_tensor.max() > 1:
            comparison_tensor = comparison_tensor / 255.0

        comparison_tensor = torch.clamp(comparison_tensor, 0, 1)

        # Save
        save_dir = os.path.join(self.cfg.saved_folder, "reconstructions")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"epoch_{epoch:03d}_batch_{batch_idx:03d}_{phase}_threeway.png"
        filepath = os.path.join(save_dir, filename)

        save_image(
            comparison_tensor,
            filepath,
            nrow=3,  # 3 images per row
            normalize=False,
            padding=2,
            pad_value=1.0
        )


    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        # Ensure directory exists for all processes
        os.makedirs(os.path.dirname(img_name), exist_ok=True)
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train_robomimic_align_recon")
def main(cfg: OmegaConf):
    # Only auto-select GPUs if CUDA_VISIBLE_DEVICES is not already set
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        # Auto-select GPUs with lowest memory usage
        selected_gpus = auto_select_gpus(getattr(cfg, 'num_gpus', 1), getattr(cfg, 'min_free_memory_gb', 2.0))
    else:
        log.info(f"Using pre-configured CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
