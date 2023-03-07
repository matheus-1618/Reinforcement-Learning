import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

q_table = loadtxt('./data/q-table-cliffwalking.csv', delimiter=',')
sarsa_table = loadtxt('./data/sarsa-table-cliffwalking.csv', delimiter=',')


env = gym.make("CliffWalking-v0", render_mode="human").env

print("Running 10 times CliffWalking")
qlearning_actions, sarsa_actions = [],[]
for i in range(10):
    #Q_learning
    (state, _) = env.reset()
    rewards_q = 0
    actions_q = 0
    done = False

    while not done:
        #print(state)
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)

        rewards_q = rewards_q + reward
        actions_q = actions_q + 1

    #Sarsa
    (state, _) = env.reset()
    rewards_sarsa = 0
    actions_sarsa = 0
    done = False
    old_state = 0

    while not done:
        old_state = state
        #print(state)
        action = np.argmax(sarsa_table[state])
        state, reward, done, truncated, info = env.step(action)

        #avoiding bug of misdirection
        if old_state == state:
            print("Ops, bug detected! Retraining Sarsa.")
            env = gym.make("CliffWalking-v0").env
            sarsa = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=10000)
            sarsa_table,sarsa_rewards = sarsa.train('data/sarsa-table-cliffwalking.csv', 'results/actions_cliffwalking_sarsa')

            print("Reseting env    ")
            env = gym.make("CliffWalking-v0", render_mode="human").env
            (state, _) = env.reset()
            rewards_sarsa = 0
            actions_sarsa = 0

        rewards_sarsa = rewards_sarsa + reward
        actions_sarsa = actions_sarsa + 1
    qlearning_actions.append(actions_q)
    sarsa_actions.append(actions_sarsa)

print(f"In 10 times, the mean of actions taken by Q_learning was: {np.mean(qlearning_actions)}")
print(f"In 10 times, the mean of actions taken by Sarsa was: {np.mean(sarsa_actions)}")