import gym
from a3c_learn import A3Cagent

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env_1:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)

if __name__ == '__main__':
    max_episode_num = 5000
    env_name = 'CartPoleContinuous-v0'
    agent = A3Cagent(env_name)
    agent.train(max_episode_num)
