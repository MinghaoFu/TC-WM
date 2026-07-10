import os
import gym
import json
import hydra
import random
import torch
import pickle
import wandb
wandb.login()  # reads ~/.netrc; set via 
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from datasets.preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import cv2

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "emb_decoder",
    "proprio_encoder",
    "action_encoder",
    "post_concat_projection",
    "task_objectives",
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states, *_ = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    # NEWT/TDMPC2 datasets have a +1 action shift: actions[t] produced states[t],
    # so to go from states[t] to states[t+1], we need actions[t+1].
    _NEWT_ENVS = {"reacher_easy", "cheetah_run", "hopper_hop"}

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # NEWT datasets: actions[t] produced states[t], so shift by +1
        act_shift = 1 if self.env_name in self._NEWT_ENVS else 0
        n_acts = self.frameskip * self.goal_H

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len + act_shift
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len - act_shift
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset + act_shift : offset + act_shift + n_acts]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        
        # ## 
        # from utils import move_to_device
        # from einops import rearrange, repeat
        
        # trans_obs_0 = move_to_device(
        #     self.planner.preprocessor.transform_obs(self.obs_0), self.planner.device
        # )
        # trans_obs_g = move_to_device(
        #     self.planner.preprocessor.transform_obs(self.obs_g), self.planner.device
        # )

        # for traj in range(actions.shape[0]):
        #     cur_trans_obs_0 = {
        #         key: repeat(
        #             arr[traj].unsqueeze(0), "1 ... -> n ...", n=1
        #         )
        #         for key, arr in trans_obs_0.items()
        #     }
        #     cur_trans_obs_g = {
        #         key: repeat(
        #             arr[traj].unsqueeze(0), "1 ... -> n ...", n=1
        #         )
        #         for key, arr in trans_obs_g.items()
        #     }
            
        #     i_z_obses, i_zs = self.wm.rollout(obs_0=cur_trans_obs_0, act=actions[traj].unsqueeze(0))
        #     env_obs_list = []
        #     for action_step in range(actions[traj].shape[1]):
        #         o, r, d, info = self.env.step(actions[:,action_step])
        #         env_obs_list.append(o["visual"])

        #     env_obs = np.concatenate(env_obs_list, axis=0)
            
        
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device) if hasattr(v, "to") else v
    result["epoch"] = payload.get("epoch", 0)
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, raw_proprio_dim, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Loaded checkpoint from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if "decoder" not in result:
        if getattr(train_cfg, "has_decoder", True):
            decoder_path = getattr(train_cfg.env, "decoder_path", None)
            if decoder_path is not None:
                base_path = os.path.dirname(os.path.abspath(__file__))
                decoder_path = os.path.join(base_path, decoder_path)
                ckpt = torch.load(decoder_path)
                result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else ckpt
            else:
                raise ValueError("Decoder not found in checkpoint and no decoder_path in config")
        else:
            result["decoder"] = None

    # emb_decoder — EmbDecoderIdentity for DINO-WM, EmbDecoderMLP for TCWM
    if "emb_decoder" not in result:
        result["emb_decoder"] = hydra.utils.instantiate(train_cfg.emb_decoder)

    # post_concat_projection — Identity for DINO-WM (no dim args), MLP for TCWM
    if "post_concat_projection" not in result:
        projector_target = str(train_cfg.projector.get("_target_", ""))
        if projector_target == "torch.nn.Identity":
            result["post_concat_projection"] = hydra.utils.instantiate(train_cfg.projector)
        else:
            projection_input_dim = (
                train_cfg.encoder_emb_dim
                + train_cfg.proprio_emb_dim * train_cfg.num_proprio_repeat
            )
            result["post_concat_projection"] = hydra.utils.instantiate(
                train_cfg.projector,
                in_features=projection_input_dim,
                out_features=train_cfg.projected_dim,
                act_embed_dim=train_cfg.action_emb_dim,
            )

    # task_objectives — nn.ModuleList in checkpoint; convert to plain list for VWorldModel
    task_objectives_ckpt = result.pop("task_objectives", None)
    task_objectives = list(task_objectives_ckpt) if task_objectives_ckpt is not None else []

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        emb_decoder=result["emb_decoder"],
        post_concat_projection=result["post_concat_projection"],
        task_objectives=task_objectives,
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        raw_proprio_dim=raw_proprio_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.eval()
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", 
            config=cfg_dict,
            entity=cfg_dict["wandb"]["entity"],
            resume="allow",
        )
    
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    model_path = cfg_dict["ckpt_base_path"]
    model_cfg = OmegaConf.load(os.path.join(model_path, "hydra.yaml"))

    seed(cfg_dict["seed"])
    slice_dsets, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    # Compute raw_proprio_dim matching train_uwm.py logic
    projector_target = str(model_cfg.projector.get("_target_", ""))
    if projector_target == "torch.nn.Identity":
        raw_proprio_dim = model_cfg.proprio_emb_dim * model_cfg.num_proprio_repeat
    else:
        raw_proprio_dim = slice_dsets["train"].proprio_dim

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, raw_proprio_dim, device=device)

    def _make_env():
        return gym.make(
            model_cfg.env.name,
            *model_cfg.env.args,
            disable_env_checker=True,
            **model_cfg.env.kwargs,
        )

    # use serial vector env for envs not compatible with SubprocVectorEnv
    if model_cfg.env.name in ("wall", "deformable_env", "reacher_easy", "cheetah_run", "hopper_hop", "point_maze", "pusht"):
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv([_make_env() for _ in range(cfg_dict["n_evals"])])
    else:
        env = SubprocVectorEnv([_make_env for _ in range(cfg_dict["n_evals"])])

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    if "wandb_logging" not in cfg_dict:
        cfg_dict["wandb_logging"] = True  # default for original plan.yaml
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
