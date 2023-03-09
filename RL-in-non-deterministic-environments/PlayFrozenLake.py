from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from Sarsa import Sarsa
from numpy import loadtxt
import warnings
warnings.simplefilter("ignore")
from utils import *

q_table = loadtxt('data/q-table.csv', delimiter=',')

env = gym.make('FrozenLake-v1',map_name="8x8",is_slippery=True, render_mode='human').env
(state, _) = env.reset()
epochs = 0
rewards = 0
done = False
    
while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _, info = env.step(action)
    rewards += reward
    epochs += 1

print("\n")
print("Timesteps taken: {}".format(epochs))
print("Rewards: {}".format(rewards))
