import threading
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Lambda
from tensorflow_probability import distributions as tfd
from keras.optimizers import Adam
import os

plt.switch_backend('agg')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


class Actor(Model):
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()
        self.actor_fc1 = Dense(24, activation='tanh')
        self.actor_mu = Dense(action_dim, activation='linear')
        self.actor_sigma = Dense(action_dim, activation='sigmoid')

    def call(self, x):
        actor_x = self.actor_fc1(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)
        return [mu, sigma]


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1, activation='linear')

    def call(self, x):
        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return value


global_episode_count = 0
global_step = 0
global_episode_reward = []
score_avg = 0
scores = []
episodes = []


class A3Cagent(object):
    def __init__(self, env_name):
        self.env_name = env_name
        self.threads = 8
        env = gym.make(env_name)

        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.global_actor = Actor(self.action_dim, self.action_bound)
        self.global_critic = Critic()

        self.global_actor.build(input_shape=(None, self.state_dim))
        self.global_critic.build(input_shape=(None, self.state_dim))

    def load_weights(self, path):
        self.global_actor.load_weights(path + 'actor.tf')
        self.global_critic.load_weights(path + 'critic.tf')

    def train(self, max_episode_num):
        workers = []
        for i in range(self.threads):
            worker_name = 'worker%i' % i
            workers.append(
                A3Cworker(worker_name, self.env_name, self.global_actor, self.global_critic, max_episode_num))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class A3Cworker(threading.Thread):
    def __init__(self, worker_name, env_name, global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)
        self.gamma = 0.99
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001
        self.t_max = 4

        self.max_episode_num = max_episode_num

        self.env = gym.make(env_name)
        self.worker_name = worker_name

        self.global_actor = global_actor
        self.global_critic = global_critic

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.sigma_bound = [1e-2, 1.0]

        self.worker_actor = Actor(self.action_dim, self.action_bound)
        self.worker_critic = Critic()
        self.worker_actor.build(input_shape=(None, self.state_dim))
        self.worker_critic.build(input_shape=(None, self.state_dim))

        self.actor_optimizer = Adam(self.actor_learning_rate)
        self.critic_optimizer = Adam(self.critic_learning_rate)

        self.worker_actor.set_weights(self.global_actor.get_weights())
        self.worker_critic.set_weights(self.global_critic.get_weights())

    def log_pdf(self, mu, sigma, action):
        sigma = tf.clip_by_value(sigma, self.sigma_bound[0], self.sigma_bound[1])
        var = sigma ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_action(self, state):
        mu, sigma = self.worker_actor(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action

    def actor_learn(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, sigma = self.global_actor(states, training=True)
            log_policy_pdf = self.log_pdf(mu, sigma, actions)
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)

        grads = tape.gradient(loss, self.global_actor.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 20)
        self.actor_optimizer.apply_gradients(zip(grads, self.global_actor.trainable_variables))

    def critic_learn(self, states, n_step_td_targets):
        with tf.GradientTape() as tape:
            td_hat = self.global_critic(states, training=True)
            loss = tf.reduce_mean(tf.square(n_step_td_targets - td_hat))

        grads = tape.gradient(loss, self.global_critic.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 20)
        self.critic_optimizer.apply_gradients(zip(grads, self.global_critic.trainable_variables))

    def n_step_td_target(self, rewards, next_v_value, done):
        y_i = np.zeros(rewards.shape)
        cumulative = 0
        if not done:
            cumulative = next_v_value
        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma * cumulative + rewards[k]
            y_i[k] = cumulative
        return y_i

    @staticmethod
    def unpack_batch(batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)
        return unpack

    def run(self):
        global global_episode_count, global_step
        global global_episode_reward
        global score_avg
        scores = []
        episodes = []
        episode_count = 0

        print(self.worker_name, 'starts ---')

        while global_episode_count <= int(self.max_episode_num):
            batch_state, batch_action, batch_reward = [], [], []
            step, episode_reward, done = 0, 0, False
            state = self.env.reset()
            while not done:
                action = self.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                reward = 0
                dist = tfd.Normal(loc=0, scale=0.25)
                reward += dist.prob(next_state[1]) / 47.7
                dist = tfd.Normal(loc=0, scale=0.015)
                reward += dist.prob(next_state[2]) / 265
                dist = tfd.Normal(loc=0, scale=0.05)
                reward += dist.prob(next_state[3]) / 159.4

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                reward = np.reshape(reward, [1, 1])

                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append(reward)

                state = next_state
                step += 1

                if len(batch_state) == self.t_max or done:
                    states = self.unpack_batch(batch_state)
                    actions = self.unpack_batch(batch_action)
                    rewards = self.unpack_batch(batch_reward)
                    batch_state, batch_action, batch_reward = [], [], []

                    next_state = np.reshape(next_state, [1, self.state_dim])
                    next_v_value = self.worker_critic(tf.convert_to_tensor(next_state, dtype=tf.float32))
                    n_step_td_targets = self.n_step_td_target(rewards, next_v_value.numpy(), done)
                    v_values = self.worker_critic(tf.convert_to_tensor(states, dtype=tf.float32))
                    advantages = n_step_td_targets - v_values

                    self.critic_learn(states, n_step_td_targets)
                    self.actor_learn(states, actions, advantages)

                    self.worker_actor.set_weights(self.global_actor.get_weights())
                    self.worker_critic.set_weights(self.global_critic.get_weights())

                    global_step += 1

                if done:
                    global_episode_count += 1
                    episode_count += 1
                    score_avg = 0.9 * score_avg + 0.1 * episode_reward if score_avg != 0 else episode_reward
                    log = 'Worker name : {:8} | '.format(self.worker_name)
                    log += 'Episode : {:4d} | '.format(global_episode_count)
                    log += 'Reward : {:5.1f} | '.format(episode_reward)
                    log += 'Score Avg : {:.1f}'.format(score_avg)
                    print(log)
                    scores.append(score_avg)
                    episodes.append(episode_count)
                    if episode_reward == 500:
                        self.global_actor.save_weights('./actor.tf')
                        self.global_critic.save_weights('./critic.tf')
                    if score_avg > 400:
                        plt.cla()
                        plt.plot(episodes, scores, 'b')
                        plt.suptitle('A3C')
                        plt.xlabel("episode")
                        plt.ylabel("average score")
                        plt.savefig(f"./graph_{self.worker_name}.png")
                        global_episode_reward.append(episode_reward)
                        sys.exit()
