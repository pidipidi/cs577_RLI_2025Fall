from gymnasium.envs.registration import register

register(
    id='reaching-v0',
    entry_point='assignment_2.envs.reaching_2d_env:ReachingEnv2D',
)

register(
    id='reaching-v1',
    entry_point='assignment_2.envs.reaching_2d_env_v1:ReachingEnv2D_v1'
)

register(
    id='KukaBulletEnv-v0',
    entry_point='assignment_2.envs.kukaGymEnv:KukaGymEnv',
    max_episode_steps=100,
    reward_threshold=1000.0,
)
