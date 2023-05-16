import gym
import numpy as np
import tensorflow as tf
from a3c_learn import A3Cagent
import pylab

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env_1:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)


def main():
    env_name = 'CartPoleContinuous-v0'
    env = gym.make(env_name)
    agent = A3Cagent(env_name)
    agent.load_weights('./220824/a3c_7/')
    for _ in range(100):
        state = env.reset()
        while True:
            env.render()
            action = agent.global_actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
            action = np.clip(action, -1, 1)
            state, reward, done, _ = env.step(action)
            if done:
                break
    env.close()


if __name__ == '__main__':
    main()
