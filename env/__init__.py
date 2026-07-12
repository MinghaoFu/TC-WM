from gym.envs.registration import register

try:
    from .pointmaze import U_MAZE
    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
except ImportError:
    pass  # d4rl not installed; point_maze env unavailable

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="robomimic_can",
    entry_point="env.robomimic.robomimic_env:RobomimicCanEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
)

try:
    register(
        id="reacher_easy",
        entry_point="env.dmcontrol.reacher_easy_wrapper:ReacherEasyWrapper",
        max_episode_steps=500,
    )
    register(
        id="cheetah_run",
        entry_point="env.dmcontrol.cheetah_run_wrapper:CheetahRunWrapper",
        max_episode_steps=500,
    )
    register(
        id="hopper_hop",
        entry_point="env.dmcontrol.hopper_hop_wrapper:HopperHopWrapper",
        max_episode_steps=500,
    )
except Exception:
    pass
