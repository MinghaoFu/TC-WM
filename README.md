# Back to Parsimonious Latents: Learning Task-Centric World Models from Visual Foundations

What latent space should a world model learn for planning and control beyond visual?

[Minghao Fu](https://minghaofu.com/) &middot; [Fan Feng](https://fan-feng.com/) &middot; [Nicklas Hansen](https://www.nicklashansen.com/) &middot; [Biwei Huang](https://biweihuang.com/) &nbsp;|&nbsp; UC San Diego

[![arXiv](https://img.shields.io/badge/arXiv-2605.25620-b31b1b)](https://arxiv.org/abs/2605.25620)
[![Project](https://img.shields.io/badge/Project-Page-2563eb)](https://minghaofu.com/tc-wm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

<p align="center">
  <a href="https://minghaofu.com/tc-wm/"><img src="https://minghaofu.com/tc-wm/static/videos/tcwm_teaser_sm.gif" width="90%" alt="TC-WM — one-minute overview"></a>
</p>
<p align="center"><sub><b>&#9654; One-minute overview</b> &mdash; click for the full video (with audio) and an interactive demo on the <a href="https://minghaofu.com/tc-wm/">project page</a>.</sub></p>

## What

TC-WM treats a pretrained visual backbone (e.g. DINOv2) as a **semantic scaffold**, not the final state space. A **linear projection** compresses its embedding into a compact latent; a designated subspace is **aligned with proprioception** via InfoNCE; a **ViT** predicts latent dynamics; a **linear decoder** reconstructs the embedding to prevent collapse. The task-centric block is identifiable up to a simple transformation.

<p align="center"><img src="https://minghaofu.com/tc-wm/static/figures/png/network.png" width="100%"></p>
<p align="center"><sub><b>Architecture.</b> A frozen visual backbone is linearly projected into a compact latent; a designated subspace z<sup>s</sup> is aligned with proprioception via InfoNCE; a ViT predicts latent dynamics; a linear decoder reconstructs the embedding to prevent collapse.</sub></p>

Empirically, TC-WM enables zero-shot **test-time planning** across nine offline visual-control tasks — Maze, Wall, Push-T, Lift, Can, Square, Reacher, Cheetah, Hopper — beating DINO-WM on every LDP task and matching strong model-based baselines.

<p align="center"><img src="https://minghaofu.com/tc-wm/static/figures/png/world_model_variants.png" width="100%"></p>
<p align="center"><sub><b>Where TC-WM sits.</b> Generative (a) and latent (b) world models, embedding world models on frozen foundations (c), and TC-WM (d&ndash;e): a compact task-centric latent <i>inside</i> the embedding, with z<sup>s</sup>/z<sup>c</sup> aligned and split.</sub></p>

## Results

<p align="center"><img src="https://minghaofu.com/tc-wm/static/figures/png/plan_compare_cem_ldp.png" width="100%"></p>
<p align="center"><sub><b>Planning.</b> CEM on Maze / Wall / Push-T / Cheetah / Hopper; LDP on Lift / Can / Square. TC-WM matches strong model-based baselines and is the only method that surpasses DINO-WM on every manipulation (LDP) task.</sub></p>

<p align="center"><img src="https://minghaofu.com/tc-wm/static/figures/png/pred_loss.png" width="100%"></p>
<p align="center"><sub><b>World-model prediction.</b> Lowest latent-prediction error on nearly all tasks; competitive image reconstruction. Lower is better.</sub></p>

<p align="center"><img src="https://minghaofu.com/tc-wm/static/figures/png/teaser_five_panel.png" width="100%"></p>
<p align="center"><sub><b>Robomimic highlight.</b> Success rate, latent-rollout MSE, linear probes on <code>Lift</code> and <code>Can</code>, and anti-collapse: TC-WM avoids the latent collapse seen when rolling out directly on foundation embeddings.</sub></p>

## Install

```bash
conda env create -f environment.yml
conda activate tcwm
bash scripts/install_mujoco.sh     # MuJoCo 210 + LD_LIBRARY_PATH setup
pip install d4rl pymunk==6.* shapely scikit-image pygame   # env extras
```

Set the data root once:

```bash
export TCWM_DATA_ROOT=/path/to/data       # configs read this via ${oc.env:TCWM_DATA_ROOT,./data}
```

## Train

```bash
# Single task
python train.py --config-name=train_tcwm env=wall

# All TC-WM tasks (one GPU each, picks free GPUs automatically)
bash scripts/run_tcwm.sh
```

Configs live in `conf/` (Hydra). Key knobs: `env=`, `encoder=` (dino / vjepa / dinov3), `projected_dim`, `alignment_dim`, `training.epochs`. Offline run by default (`WANDB_MODE=offline`).

## Plan

```bash
python plan.py --config-name=plan_wall \
  ckpt_base_path=$TCWM_DATA_ROOT/checkpoints/wall_proj256 \
  n_evals=50
```

Available planners: CEM (`conf/planner/cem.yaml`), LDP (`conf/planner/ldp.yaml`), GD. Rollout videos auto-saved as `plan{batch}_{trial}_{success|failure}.mp4` in the Hydra run dir.

## Rollout (qualitative)

```bash
python rollout.py --config-name=train_tcwm \
  env=wall resume_folder=$TCWM_DATA_ROOT/checkpoints/wall_proj256 \
  +traj_idx=0 +output_dir=rollouts/wall_idx0 +fps=8
```

Saves three GIFs per trajectory: `{env}_orig.gif`, `{env}_recon.gif`, `{env}_pred.gif` &mdash; original observation, autoencoder reconstruction, open-loop TC-WM prediction.

## Repository layout

```
TC-WM/
├── train.py · plan.py · rollout.py · rollout_videos.py    # entry points
├── utils.py · custom_resolvers.py       # shared helpers + Hydra resolvers
├── scripts/         # train.sh, run_tcwm.sh, install_mujoco.sh
├── conf/           # Hydra configs (env / encoder / method / planner / decoder / ...)
├── models/         # VWorldModel: encoder · projector · predictor · decoder
├── env/            # Gym wrappers for Maze, Wall, Push-T, Robomimic, DMC
├── datasets/       # Trajectory loaders + preprocessor
├── planning/       # CEM, LDP, MPC, evaluator
├── metrics/  ·  distributed_fn/  ·  gpu_utils/
├── environment.yml · requirements.txt · default_config.yaml
└── assets/
```

## Checkpoints

Pretrained checkpoints will be released at [`MinghaoFu/TC-WM-checkpoints`](https://huggingface.co/MinghaoFu/TC-WM-checkpoints) on HuggingFace Hub. Each task ships with a single best-seed checkpoint; load with `resume_folder=<path>`.

## Citation

```bibtex
@article{fu2026tcwm,
  title   = {Back to Parsimonious Latents: Learning Task-Centric World Models from Visual Foundations},
  author  = {Fu, Minghao and Feng, Fan and Hansen, Nicklas and Huang, Biwei},
  journal = {arXiv preprint arXiv:2605.25620},
  year    = {2026}
}
```

## Contact

`m9fu [at] ucsd [dot] edu`
