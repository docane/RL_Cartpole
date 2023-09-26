import gym
import numpy as np
from a2c_learn import ContinuousA2CAgent
import pylab

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env:ContinuousCartPoleEnv',
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
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent1 = ContinuousA2CAgent(action_size, max_action, state_size)
    agent1.load_weights('../trainmodel/220930/a2c_3/')
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    step = 0
    while True:
        env.render()
        action = agent1.get_action(state)
        state, reward, done, _ = env.step(action)
        step += 1
        state = np.reshape(state, [1, state_size])
        cart_position.append(state[0][0])
        cart_speed.append(state[0][1])
        pole_theta.append(state[0][2])
        pole_theta_dot.append(state[0][3])
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
    pylab.ylabel('rad', rotation=0)
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
