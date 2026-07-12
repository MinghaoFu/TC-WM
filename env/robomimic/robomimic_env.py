import gym
import h5py
import numpy as np
from robomimic.config import config_factory
from robomimic.utils import env_utils, file_utils, obs_utils
from robosuite.wrappers import GymWrapper

def get_dataset_path(prefix, env_name):
    """Get dataset path for given environment name and data prefix"""
    dataset_paths = {
        "can-mh": f"{prefix}/can/mh/low_dim.hdf5",
        "can-ph": f"{prefix}/can/ph/low_dim.hdf5",
        "lift-mh": f"{prefix}/lift/mh/low_dim.hdf5",
        "lift-ph": f"{prefix}/lift/ph/low_dim.hdf5",
        "square-mh": f"{prefix}/square/mh/low_dim.hdf5",
        "square-ph": f"{prefix}/square/ph/low_dim.hdf5",
    }
    return dataset_paths[env_name]


class RobomimicEnv(gym.Env):
    def __init__(self, env_name, data_prefix="./data/robomimic",
                 img_size=224, camera_name="agentview"):
        super().__init__()
        self.env_name = env_name
        self.img_size = img_size
        self.camera_name = camera_name
        path = get_dataset_path(data_prefix, env_name)
        env_meta = file_utils.get_env_metadata_from_dataset(path)
        env = env_utils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=True,  # Enable offscreen rendering for images
            use_image_obs=False
        ).env
        env.ignore_done = False
        env._max_episode_steps = env.horizon
        keys = [
            "object-state",
            "robot0_joint_pos",
            "robot0_joint_pos_cos",
            "robot0_joint_pos_sin",
            "robot0_joint_vel",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel",
        ]
        self.env = GymWrapper(env, keys=keys)
        self._max_episode_steps = self.env.horizon
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def _unwrap_reset(self, result):
        if isinstance(result, tuple):
            return result[0]
        return result

    def _unwrap_step(self, result):
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return obs, reward, done, info
        return result

    def _get_image(self):
        """Render current frame as image (H, W, C) uint8."""
        img = self.env.env.sim.render(
            camera_name=self.camera_name,
            width=self.img_size,
            height=self.img_size,
            depth=False,
        )
        img = img[::-1]  # flip vertically (mujoco renders upside down)
        return img

    def _get_state(self):
        return self.env.env.sim.get_state().flatten()

    def _get_obs_flat(self):
        obs_dict = self.env.env._get_observations()
        return np.concatenate([obs_dict[k].flatten() for k in self.env.keys])

    def step(self, action):
        result = self.env.step(action)
        obs, reward, done, info = self._unwrap_step(result)
        if self.env._check_success():
            done = True
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        return self._unwrap_reset(result)

    def render(self, mode="human"):
        return self.env.render(mode)

    def update_env(self, env_info):
        pass

    def prepare(self, seed, init_state):
        np.random.seed(seed)
        self.reset()
        if init_state is not None:
            try:
                self.env.env.sim.set_state_from_flattened(init_state)
                self.env.env.sim.forward()
            except Exception:
                pass
        obs_flat = self._get_obs_flat()
        state = self._get_state()
        return obs_flat, state

    def step_multiple(self, actions):
        all_obs_flat = []
        all_rewards = []
        all_dones = []
        all_states = []
        all_images = []

        for t in range(len(actions)):
            obs, reward, done, info = self.step(actions[t])
            all_obs_flat.append(obs)
            all_rewards.append(reward)
            all_dones.append(done)
            all_states.append(self._get_state())
            all_images.append(self._get_image())

        rewards = np.array(all_rewards)
        dones = np.array(all_dones)
        infos = {"state": np.stack(all_states)}

        return {"visual": np.stack(all_images)}, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        obs_flat, state = self.prepare(seed, init_state)
        init_image = self._get_image()

        obses_dict, rewards, dones, infos = self.step_multiple(actions)

        # Prepend initial image
        obses_dict["visual"] = np.vstack([
            np.expand_dims(init_image, 0),
            obses_dict["visual"]
        ])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])

        return obses_dict, states, rewards

    def eval_state(self, goal_state, cur_state):
        success = self.env._check_success()
        return {"success": bool(success), "state_dist": 0.0}

    def sample_random_init_goal_states(self, seed):
        np.random.seed(seed)
        self.reset()
        init_state = self._get_state()
        self.reset()
        goal_state = self._get_state()
        return init_state, goal_state
