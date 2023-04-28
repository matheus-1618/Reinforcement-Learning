import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
from DoubleDeepQLearning import DoubleDeepQLearning
import matplotlib.pyplot as plt

import argparse

import warnings
warnings.filterwarnings("ignore")

def Model(env):        
    model = Sequential()
    model.add(Dense(512, activation=relu, input_dim=env.observation_space.shape[0]))
    model.add(Dense(256, activation=relu))
    model.add(Dense(env.action_space.n, activation=linear))
    return model

def plot_reward_per_episode(rewards ,fileName):  
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('# Rewards')
    plt.title('# Rewards vs Episodes')
    plt.grid(True)
    plt.savefig(f"{fileName}.jpg")     
    plt.close()
 
def main(args):

    # Files names:
    plot_rewards_file =  "results/LunarLander_rewards_per_episode_best"
    model_filename = "data/model_lunarlander_best"
   
    if(args.train):
        best_model = None
        best_reward = 0
        rewards = 0

        # ------------- Train Model --------------- 
        
        # Instância do Ambiente
        env = gym.make('LunarLander-v2')
        np.random.seed(42)
        
        # Parâmetros
        gamma = 0.99 
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_dec = 0.99
        episodes = 500
        batch_size = 64
        memory = deque(maxlen=500000)
        max_steps = 1500
        
        # Cria modelo de Rede Neural
        model = Model(env)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

        target = Model(env)
        target.summary()
        target.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
 
        # Treinando modelo com Rede Neural

        if(args.algorithm == 1):

            print("\n > Utilizando o DoubleDeepQLearning implementado \n")
            DDQN =   DoubleDeepQLearning(env, gamma, epsilon, epsilon_min, epsilon_dec, episodes, batch_size, memory, model,max_steps,target,50)
            rewards = DDQN.train()
            # Grafico de rewards
            plot_reward_per_episode(rewards, plot_rewards_file)
            model.save(model_filename)

        elif(args.algorithm == 0):
            
            print("\n > Utilizando o DoubleDeepQLearning de um módulo python \n") 
    else:

        # ------- Use a existing model -----------
        env = gym.make('LunarLander-v2', render_mode="human")
        np.random.seed(42)
        
        state , _ = env.reset()
        model = keras.models.load_model(model_filename, compile=False)
        done = False
        rewards = 0
        steps = 0

        while not done and steps < 1500:
            state = np.array(state).reshape(1,env.observation_space.shape[0])
            actions = model.predict(state)
            action = np.argmax(actions[0]) 
            state, reward, terminal, truncated, _  = env.step(action)
            rewards += reward
            env.render()
            steps += 1

if __name__ == "__main__":

    # Argumentos do terminal
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help= "Train model or use existing template")
    parser.add_argument('-a', '--algorithm', type = int , help = 'Select model for train [library model - 0 or myModel - 1]' )  
    args, unknown_args = parser.parse_known_args()
    
    if(args.train):
        print("\n---------------------------------")
        print("-----------TRAIN MODEL-----------")
        print("---------------------------------\n")
    else:
        print("\n---------------------------------")
        print("-----------USING MODEL-----------")
        print("---------------------------------\n")

    # Funcao principal
    main(args)
