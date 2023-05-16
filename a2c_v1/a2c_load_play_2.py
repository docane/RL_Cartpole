import gym
import numpy as np
from a2c_v1.a2c_learn import ContinuousA2CAgent

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env_1:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)


def main():
    env_name = 'CartPoleContinuous-v0'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = ContinuousA2CAgent(action_size, max_action)
    agent.load_weights('./220824/a2c_7/')
    for _ in range(100):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while True:
            env.render()
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            state = np.reshape(state, [1, state_size])
            if done:
                break
    env.close()


if __name__ == '__main__':
    main()
