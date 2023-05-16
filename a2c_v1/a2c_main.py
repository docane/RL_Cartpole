import gym
from a2c_v1.a2c_learn import ContinuousA2CAgent
import numpy as np
import pylab
import sys
from tensorflow_probability import distributions as tfd
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env_2:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)

env = gym.make('CartPoleContinuous-v0')

# 환경으로부터 상태와 행동의 크기를 받아옴
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = env.action_space.high[0]

# 액터-크리틱(A2C) 에이전트 생성
agent = ContinuousA2CAgent(action_size, max_action)
scores, episodes = [], []
score_avg = 0
num_episode = 1000
train_done = False
for e in range(num_episode):
    done = False
    score = 0
    loss_list, sigma_list = [], []
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        score += reward
        # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
        # reward = 0.1 if not done or score == 500 else -1
        reward = 0
        dist = tfd.Normal(loc=0, scale=0.25)
        reward += dist.prob(next_state[0][1]) / 47.7
        dist = tfd.Normal(loc=0, scale=0.015)
        reward += dist.prob(next_state[0][2]) / 265
        dist = tfd.Normal(loc=0, scale=0.05)
        reward += dist.prob(next_state[0][3]) / 159.4
        # 매 타임스텝마다 학습
        loss, sigma = agent.train_model(state, action, reward, next_state, done)
        loss_list.append(loss)
        sigma_list.append(sigma)
        state = next_state
        env.render()

        if done:
            # 에피소드마다 학습 결과 출력
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f} | sigma: {:.3f}".format(
                e, score_avg, np.mean(loss_list), np.mean(sigma)))

            scores.append(score_avg)
            episodes.append(e)
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("average score")
            pylab.savefig("./graph.png")

            # 이동 평균이 400 이상일 때 종료
            if score_avg > 400:
                agent.model.save_weights("./model.tf")
                sys.exit()
