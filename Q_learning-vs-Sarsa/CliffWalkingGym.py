import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from Sarsa import Sarsa
from QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

env = gym.make("CliffWalking-v0").env

def mean(r):
    out = []
    a = list(np.arange(0,len(r),10))
    for i in range(1,len(a)):
        out.append(np.mean(r[a[i-1]:a[i]]))
    return out

def specific_plot(qlearning, sarsa):
    r1,r2 = mean(qlearning), mean(sarsa)
    plt.plot(range(len(r1)),r1,'b',label="Q-Learning")
    plt.plot(range(len(r2)),r2,'g', label="Sarsa")
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes | Sarsa vs Q Learning | Cliff Walking')
    plt.legend(loc="best")
    plt.xlim(-5,305)
    #plt.figtext(0.5, 0, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig("Sarsa_vs_Qlearning|CliffWalking"+".jpg")     
    plt.close()

print("training QLearning\n")
qlearn = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=10000)
q_table,q_rewards = qlearn.train('data/q-table-cliffwalking.csv', 'results/actions_cliffwalking_qlearning')
#q_table = loadtxt('data/q-table-taxi-driver.csv', delimiter=',')

print("training Sarsa\n")
sarsa = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=10000)
sarsa_table,sarsa_rewards = sarsa.train('data/sarsa-table-cliffwalking.csv', 'results/actions_cliffwalking_sarsa')

specific_plot(q_rewards, sarsa_rewards)

env = gym.make("CliffWalking-v0", render_mode="human").env

#Q_learning
(state, _) = env.reset()
rewards_q = 0
actions_q = 0
done = False

while not done:
    print(state)
    action = np.argmax(q_table[state])
    state, reward, done, truncated, info = env.step(action)

    rewards_q = rewards_q + reward
    actions_q = actions_q + 1

#Sarsa
(state, _) = env.reset()
rewards_sarsa = 0
actions_sarsa = 0
done = False

while not done:
    print(state)
    action = np.argmax(sarsa_table[state])
    state, reward, done, truncated, info = env.step(action)

    #avoiding bugs
    if actions_sarsa > 50:
        (state, _) = env.reset()
        rewards_sarsa = 0
        actions_sarsa = 0

    rewards_sarsa = rewards_sarsa + reward
    actions_sarsa = actions_sarsa + 1

print("\n")
print("Actions taken for Q learning: {}".format(actions_q))
print("Rewards for Q Learning: {}".format(rewards_q))
print("\n")
print("Actions taken for Sarsa: {}".format(actions_sarsa))
print("Rewards for Sarsa: {}".format(rewards_sarsa))
