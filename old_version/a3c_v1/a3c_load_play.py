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

cart_position = []
cart_speed = []
pole_theta = []
pole_theta_dot = []
steps = []


def main():
    env_name = 'CartPoleContinuous-v0'
    env = gym.make(env_name)
    agent = A3Cagent(env_name)
    agent.load_weights('./220824/a3c_7/')
    state = env.reset()
    step = 0
    while True:
        env.render()
        action = agent.global_actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        action = np.clip(action, -1, 1)
        state, reward, done, _ = env.step(action)
        step += 1
        cart_position.append(state[0])
        cart_speed.append(state[1])
        pole_theta.append(state[2])
        pole_theta_dot.append(state[3])
        steps.append(step)
        if done:
            break
    env.close()

    pylab.subplot(2, 2, 1)
    pylab.plot(steps, cart_position)
    pylab.title('Cart Position')
    pylab.xlabel('step')
    pylab.ylim(-1, 1)
    pylab.subplot(2, 2, 2)
    pylab.plot(steps, cart_speed)
    pylab.title('Cart Speed')
    pylab.xlabel('step')
    pylab.ylim(-1, 1)
    pylab.subplot(2, 2, 3)
    pylab.plot(steps, pole_theta)
    pylab.title('Pole Theta')
    pylab.xlabel('step')
    pylab.ylim(-0.1, 0.1)
    pylab.subplot(2, 2, 4)
    pylab.plot(steps, pole_theta_dot)
    pylab.title('Pole Theta Dot')
    pylab.xlabel('step')
    pylab.ylim(-1, 1)
    pylab.tight_layout()
    pylab.savefig('./state')


if __name__ == '__main__':
    main()
