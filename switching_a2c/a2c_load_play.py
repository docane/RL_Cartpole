import gym
import numpy as np
from a2c_learn import ContinuousA2CAgent
import pylab
import math

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='switching_env_test:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)

cart_position = []
cart_speed = []
pole_theta = []
pole_theta_dot = []
steps = []
theta_threshold_radians = 12 * 2 * math.pi / 360


def main():
    env_name = 'CartPoleContinuous-v0'
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent1 = ContinuousA2CAgent(action_size, max_action)
    agent1.load_weights('./model/a2c_4_2/')
    agent2 = ContinuousA2CAgent(action_size, max_action)
    agent2.load_weights('./model/a2c_4_3/')
    agent3 = ContinuousA2CAgent(action_size, max_action)
    agent3.load_weights('./model/a2c_4_4/')
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    step = 0
    while True:
        env.render()
        action1 = agent1.get_action(state)
        action2 = agent2.get_action(state)
        action3 = agent3.get_action(state)
        if state[0][2] < (-theta_threshold_radians * (1 / 3)):
            action = action1
            print(f'각 = {state[0][2]}, step {step} = 1')
        elif (-theta_threshold_radians * (1 / 3)) < state[0][2] < 0:
            ratio = (state[0][2] + (theta_threshold_radians * (1 / 3))) / (theta_threshold_radians * (1 / 3))
            action = ((action1 * (1 - ratio)) + (action2 * ratio)) / 2
            print(f'각 = {state[0][2]}, step {step} = 2')
        elif 0 < state[0][2] < (theta_threshold_radians * (1 / 3)):
            ratio = state[0][2] / (theta_threshold_radians * (1 / 3))
            print(f'각 = {state[0][2]}, step {step} = 3')
            action = ((action2 * (1 - ratio)) + (action3 * ratio)) / 2
        else:
            action = action3
            print(f'각 = {state[0][2]}, step {step} = 4')
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
    pylab.savefig('./images/state.png')


if __name__ == '__main__':
    main()
