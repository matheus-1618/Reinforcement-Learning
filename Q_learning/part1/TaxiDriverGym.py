import random
from IPython.display import clear_output
import gymnasium as gym
import numpy as np
from QLearning import QLearning
from numpy import loadtxt
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode='ansi').env

def specific_plot(r1, r2, r3):
    plt.scatter(range(len(r1)),r1,label="alpha=.1, gamma=.1, epsilon=.5", alpha=0.8)
    plt.scatter(range(len(r2)),r2,label="alpha=.5, gamma=.5, epsilon=.7", marker='s', alpha=0.8)
    plt.scatter(range(len(r3)),r3,label="alpha=.9, gamma=.9, epsilon=.9", marker='^', alpha=0.8)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.legend(loc="best")
    plt.xlim(-5,105)
    txt='''Figure 1. #Rewards x Episodes.\nRewards are used as 
    values ​​on the y-axis, as they are values ​​with a tendency to 
    vary greatly according to the choice of hyperparameters and 
    with a greater sensitivity range for evaluating the model
     than the simple amount of actions performed in each episode. 
     In the figure, three triples of hyperparameter values ​​of 
     the Q-learning reinforcement learning model are presented, 
     these being α (learning rate), relative to the weight that 
     a new learning (future value) has in relation to the current 
     value, that is , its importance in the current composition; 
     γ (discount factor) that measures the importance of future 
     rewards in relation to the current one and ε , which is 
     the model exploration factor, essential to select the best 
     action to be taken with a higher success rate. In this sense,
      three triples of these values ​​were chosen, one with low 
      values ​​(valuing less the importance of values, rewards and 
      future actions in relation to the current moment), high 
      values ​​(valuing more the importance of values, rewards and 
      future actions in relation to the current moment) and 
      intermediate values ​​between these. From the figure, it can 
      be seen that the one with high values ​​(triangular points) 
      has a tendency to converge to rewards close to zero in a 
      smaller number of episodes, since as future actions have 
      high importance in their composition, they end up making 
      their distribution more uniform. action through greater
       knowledge of subsequent actions and thus facilitate 
       better decision-making earlier. On the other hand, the one 
       with lower values ​​(circular dots), do not show a clear 
       convergence trend and show less uniformity of behavior
        than other elements in the figure, due to the fact that 
        the actions taken have their weighting very little 
        dependent on actions and states futures which impacts on
         greater randomness and choice of less complex patterns
          that may make the model less efficient in terms of fewer
           episodes.'''
    #plt.figtext(0.5, 0, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig("Final"+".jpg")     
    plt.close()

# only execute the following lines if you want to create a new q-table
qlearn = QLearning(env, alpha=.1, gamma=.1, epsilon=.5, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
q_table,r1  = qlearn.train('data/q-table-taxi-driver.csv', 'results/actions_taxidriver2')

qlearn = QLearning(env, alpha=.5, gamma=.5, epsilon=.7, epsilon_min=0.05, epsilon_dec=0.99, episodes=5000)
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