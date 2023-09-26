import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow_probability import distributions as tfd
import sys
import pylab
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


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


class Critic(Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_value = Dense(1)

    def call(self, x):
        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_value(critic_x)
        return value


# 카트폴 예제에서의 연속적 액터-크리틱(A2C) 에이전트
class ContinuousA2CAgent:
    def __init__(self, env):
        self.render = False

        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_learning_rate = 0.0001
        self.critic_learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.actor = Actor(self.action_size)
        self.critic = Critic()
        self.actor.build(input_shape=(None, self.state_size))
        self.critic.build(input_shape=(None, self.state_size))

        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.actor_optimizer = Adam(lr=self.actor_learning_rate, clipnorm=1.0)
        self.critic_optimizer = Adam(lr=self.critic_learning_rate, clipnorm=1.0)

    def load_weights(self, path):
        self.actor.load_weights(path + 'actor.tf')
        self.critic.load_weights(path + 'critic.tf')

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma = self.actor(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    def actor_learn(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            mu, sigma = self.actor(state)
            value = self.critic(state)
            next_value = self.critic(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            advantage = tf.stop_gradient(target - value[0])
            dist = tfd.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_mean(cross_entropy * advantage)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

    def critic_learn(self, state, reward, next_state, done):
        with tf.GradientTape() as tape:
            value = self.critic(state)
            next_value = self.critic(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 가치 신경망 오류 함수 구하기
            loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            loss = tf.reduce_mean(loss)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

    def train(self, max_episode):
        scores, episodes = [], []
        score_avg = 0
        for e in range(max_episode):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward

                # reward = 0
                # if abs(next_state[0][2]) < 0.02:
                #     reward += 0.1
                # if abs(next_state[0][3]) < 0.1:
                #     reward += 0.05
                reward = 0
                dist = tfd.Normal(loc=0, scale=0.25)
                reward += dist.prob(next_state[0][1]) / 47.7
                dist = tfd.Normal(loc=0, scale=0.015)
                reward += dist.prob(next_state[0][2]) / 265
                dist = tfd.Normal(loc=0, scale=0.05)
                reward += dist.prob(next_state[0][3]) / 159.4

                self.actor_learn(state, action, reward, next_state, done)
                self.critic_learn(state, reward, next_state, done)
                state = next_state
                self.env.render()

                if done:
                    score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                    print("episode: {:3d} | score: {:3.0f} | score avg: {:3.2f}".format(e, score, score_avg))

                    scores.append(score_avg)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.suptitle('A2C')
                    pylab.xlabel("episode")
                    pylab.ylabel("average score")
                    pylab.savefig("./graph.png")

                    # 이동 평균이 400 이상일 때 종료
                    if score_avg > 400:
                        self.actor.save_weights('./actor.tf')
                        self.critic.save_weights('./critic.tf')
                        sys.exit()
