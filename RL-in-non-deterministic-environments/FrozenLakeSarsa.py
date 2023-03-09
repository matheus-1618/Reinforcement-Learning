from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from Sarsa import Sarsa
from numpy import loadtxt
import warnings
warnings.simplefilter("ignore")
from utils import *

# exemplo de ambiente nao determinístico
env = gym.make('FrozenLake-v1',map_name="8x8",is_slippery=True, render_mode='ansi').env

# only execute the following lines if you want to create a new q-table
qlearn1 = Sarsa(env, alpha=0.4, gamma=0.95, epsilon=0.95, epsilon_min=0.0001, epsilon_dec=0.9999, episodes=20000)
q_table1,r1 = qlearn1.train('data/q-table1.csv','results/frozen_lake_sarsa')

env.reset()
qlearn2 = Sarsa(env, alpha=0.2, gamma=0.95, epsilon=0.95, epsilon_min=0.0001, epsilon_dec=0.9999, episodes=20000)
q_table2,r2 = qlearn2.train('data/q-table2.csv','results/frozen_lake_sarsa')

env.reset()
qlearn3 = Sarsa(env, alpha=0.1, gamma=0.95, epsilon=0.95, epsilon_min=0.0001, epsilon_dec=0.9999, episodes=20000)
q_table3,r3 = qlearn3.train('data/q-table3.csv','results/frozen_lake_sarsa')

env.reset()
qlearn = Sarsa(env, alpha=0.05, gamma=0.95, epsilon=0.95, epsilon_min=0.0001, epsilon_dec=0.9999, episodes=20000)
q_table,r4 = qlearn.train('data/q-table.csv','results/frozen_lake_sarsa')

general_plot(r1,r2,r3,r4)

plt.suptitle(f'# Rewards vs Episodes | Sarsa | Frozen Lake',fontsize=13)
plt.subplot(221)
specific_plot(r1,'alpha=0.4, gamma=0.95','red')
plt.subplot(222)
specific_plot(r2,'alpha=0.2, gamma=0.95','yellow')
plt.subplot(223)
specific_plot(r3,'alpha=0.1, gamma=0.95','blue')
plt.subplot(224)
specific_plot(r4,'alpha=0.05, gamma=0.95','green')

plt.tight_layout(pad=3.0)
plt.savefig(f"Sarsa_FrozenLake_specific"+".jpg")  