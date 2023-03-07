import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from IPython.display import clear_output
from time import sleep

q_table = loadtxt('./data/q-table-taxi-driver.csv', delimiter=',')
sarsa_table = loadtxt('./data/sarsa-table-taxi-driver.csv', delimiter=',')

env = gym.make("Taxi-v3", render_mode='human').env

print("Running 10 times Taxi Driver")
qlearning_epochs, sarsa_epochs = [],[]
for i in range(10):
    #QLearning game   
    (state, _) = env.reset()
    epochs_q, penalties_q, reward = 0, 0, 0
    done = False
    frames_q = [] # for animation

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)

        if reward == -10:
            penalties_q += 1

        # Put each rendered frame into dict for animation
        frames_q.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs_q += 1

    #Sarsa game 
    (state, _) = env.reset()
    epochs_sarsa, penalties_sarsa, reward = 0, 0, 0
    done = False
    frames_sarsa = [] # for animation
    
    while not done:
        #print("state: "+ str(state))
        action = np.argmax(sarsa_table[state])
        state, reward, done, truncated, info = env.step(action)

        if reward == -10:
            penalties_q += 1

        #Avoid bugs
        if epochs_sarsa > 100:
            (state, _) = env.reset()
            epochs_sarsa, penalties_sarsa, reward = 0, 0, 0
            frames_sarsa = [] # for animation


        # Put each rendered frame into dict for animation
        frames_sarsa.append({
            'frame': env.render(),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs_sarsa += 1

    qlearning_epochs.append(epochs_q)
    sarsa_epochs.append(epochs_sarsa)

print(f"In 10 times, the mean of actions taken by Q_learning was: {np.mean(qlearning_epochs)}")
print(f"In 10 times, the mean of actions taken by Sarsa was: {np.mean(sarsa_epochs)}")