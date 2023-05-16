import gym
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


class Actor(Model):
    def __init__(self, action_size):
        super(Actor, self).__init__()
        self.actor_fc1 = Dense(24, activation='tanh')
        self.actor_mu = Dense(action_size)
        self.actor_sigma = Dense(action_size, activation='sigmoid')

    def call(self, x):
        actor_x = self.actor_fc1(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)
        return mu, sigma


gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env_2:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)

env = gym.make('CartPoleContinuous-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

actor = Actor(action_size)
state = env.reset()
state = np.reshape(state, [1, state_size])
mu, sigma = actor(state)
print(mu[0][0], sigma[0])
