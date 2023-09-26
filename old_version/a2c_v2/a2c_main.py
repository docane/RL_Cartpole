import gym
from a2c_learn import ContinuousA2CAgent
import numpy as np
import pylab
import sys

gym.envs.register(
    id='CartPoleContinuous-v0',
    entry_point='env:ContinuousCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0)

env = gym.make('CartPoleContinuous-v0')

# 환경으로부터 상태와 행동의 크기를 받아옴
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
max_action = env.action_space.high[0]

# 액터-크리틱(A2C) 에이전트 생성
agent = ContinuousA2CAgent(action_size, max_action, state_size)
scores, episodes = [], []
score_avg = 0
num_episode = 1000
train_done = False
for e in range(num_episode):
    done = False
    score = 0
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
        if abs(next_state[0][2]) < 0.02:
            reward += 0.1
        if abs(next_state[0][3]) < 0.1:
            reward += 0.05
        # 매 타임스텝마다 학습
        agent.actor_learn(state, action, reward, next_state, done)
        agent.critic_learn(state, reward, next_state, done)
        state = next_state
        env.render()

        if done:
            # 에피소드마다 학습 결과 출력
            score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
            print("episode: {:3d} | score: {:3.0f} | score avg: {:3.2f}".format(e, score, score_avg))

            scores.append(score_avg)
            episodes.append(e)
            pylab.plot(episodes, scores, 'b')
            pylab.xlabel("episode")
            pylab.ylabel("average score")
            pylab.savefig("./graph.png")

            # 이동 평균이 400 이상일 때 종료
            if score_avg > 400:
                agent.actor.save_weights('./actor.tf')
                agent.critic.save_weights('./critic.tf')
                sys.exit()
