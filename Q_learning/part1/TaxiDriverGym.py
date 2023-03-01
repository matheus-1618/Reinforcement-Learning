import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode='ansi').env

def specific_plot(r1, r2, r3):
    plt.scatter(range(len(r1)),r1,label="alpha=.1, gamma=.1, epsilon=.1", alpha=0.8)
    plt.scatter(range(len(r2)),r2,label="alpha=.1, gamma=.5, epsilon=.7", marker='s', alpha=0.8)
    plt.scatter(range(len(r3)),r3,label="alpha=.9, gamma=.9, epsilon=.9", marker='^', alpha=0.8)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.legend(loc="best")
    plt.xlim(-5,65)
    plt.savefig("Final"+".jpg")     
    plt.close()

# only execute the following lines if you want to create a new q-table
qlearn = QLearning(env, alpha=.1, gamma=.1, epsilon=.1, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table,r1  = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver2')

qlearn = QLearning(env, alpha=.1, gamma=.5, epsilon=.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table,r2  = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver2')

qlearn = QLearning(env, alpha=.9, gamma=.9, epsilon=.9, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table,r3  = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver2')

specific_plot(r1, r2, r3)
#q_table = loadtxt('data/q-table-taxi-driver.csv', delimiter=',')


(state, _) = env.reset()
epochs, penalties, reward = 0, 0, 0
done = False
frames = [] # for animation
    
while (not done) and (epochs < 100):
    action = np.argmax(q_table[state])
    state, reward, done, t, info = env.step(action)

    if reward == -10:
        penalties += 1

    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        }
    )
    epochs += 1

from IPython.display import clear_output
from time import sleep

clear_output(wait=True)

def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        #print(frame['frame'].getvalue())
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)
        
print_frames(frames)

print("\n")
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))