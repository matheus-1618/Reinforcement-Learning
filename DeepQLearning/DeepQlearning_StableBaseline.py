import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import io
import os
import glob
import torch
import base64
import stable_baselines3
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.results_plotter import ts2xy, load_results
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
import gym
from gym import spaces

class DeepQlearningBase:
  def train(self):
    nn_layers = [64,64] #This is the configuration of your neural network. Currently, we have two layers, each consisting of 64 neurons.
                        #If you want three layers with 64 neurons each, set the value to [64,64,64] and so on.

    learning_rate = 0.001 #This is the step-size with which the gradient descent is carried out.
                          #Tip: Use smaller step-sizes for larger networks.
                        

    self.log_dir = "/tmp/gym/"
    os.makedirs(self.log_dir, exist_ok=True)

    # Create environment
    env = gym.make('LunarLander-v2')

    callback = EvalCallback(env,log_path = self.log_dir, deterministic=True)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=nn_layers)
    self.model = DQN("MlpPolicy", env,policy_kwargs = policy_kwargs,
                learning_rate=learning_rate,
                batch_size=1, 
                buffer_size=1, 
                learning_starts=1, 
                gamma=0.99, 
                tau = 1, 
                target_update_interval=1, 
                train_freq=(1,"step"), 
                max_grad_norm = 10,
                exploration_initial_eps = 1, 
                exploration_fraction = 0.5, 
                gradient_steps = 1, 
                seed = 1, 
                verbose=0) 

    self.model.learn(total_timesteps=100000, log_interval=10,callback=callback)

  def save_model(self,name):
    self.model.save(name)

  def execute(self):
    env = (gym.make("LunarLander-v2",render_mode="human"))
    (state,_) = env.reset()
    while True:
      env.render()
      action, _states = self.model.predict(state, deterministic=True)
      state, reward, done, info,_ = env.step(action)
      if done:
        break
  def plot(self):
    x, y = ts2xy(load_results(self.log_dir), 'timesteps')  # Organising the logged results in to a clean format for plotting.
    plt.plot(x,y)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.grid(True)
    plt.savefig(f"results/Stable_model.jpg")     
    plt.close()