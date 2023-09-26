import gym
from a2c_learn import ContinuousA2CAgent

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)


def main():
    env = gym.make('CartPoleContinuous-v0')
    agent = ContinuousA2CAgent(env)
    agent.train(1000)


if __name__ == '__main__':
    main()
