import gym
import numpy as np
from a2c_v1.a2c_learn import ContinuousA2CAgent
import math

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='switching_env_test:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)
theta_threshold_radians = 12 * 2 * math.pi / 360


def main():
    env_name = 'CartPoleContinuous-v0'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent1 = ContinuousA2CAgent(action_size, max_action)
    agent1.load_weights('./220923/a2c_4_2/')
    agent2 = ContinuousA2CAgent(action_size, max_action)
    agent2.load_weights('./220923/a2c_4_3/')
    agent3 = ContinuousA2CAgent(action_size, max_action)
    agent3.load_weights('./220923/a2c_4_4/')
    for _ in range(100):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while True:
            env.render()
            action1 = agent1.get_action(state)
            action2 = agent2.get_action(state)
            action3 = agent3.get_action(state)
            if state[0][2] < (-theta_threshold_radians * (1 / 3)):
                action = action1
            elif (-theta_threshold_radians * (1 / 3)) < state[0][2] < 0:
                ratio = (state[0][2] + (theta_threshold_radians * (1 / 3))) / (theta_threshold_radians * (1 / 3))
                action = ((action1 * (1 - ratio)) + (action2 * ratio)) / 2
            elif 0 < state[0][2] < (theta_threshold_radians * (1 / 3)):
                ratio = state[0][2] / (theta_threshold_radians * (1 / 3))
                action = ((action2 * (1 - ratio)) + (action3 * ratio)) / 2
            else:
                action = action3
            state, reward, done, _ = env.step(action)
            state = np.reshape(state, [1, state_size])
            if done:
                break
    env.close()


if __name__ == '__main__':
    main()
