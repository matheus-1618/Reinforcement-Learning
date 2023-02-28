import gymnasium as gym
from pettingzoo.classic import tictactoe_v3
import random
from utils import *


def play_random_agent(agent, obs, env):
    x = env.action_space(agent).sample()
    while obs['action_mask'][x] != 1:
        x = env.action_space(agent).sample()
    return x

def play_min_max_agent(agent, obs):
    if len(get_available_slots(obs['action_mask'])) == 9:
        return random.randint(0,8)
    matrix = fill_matrix(obs)
    result = minMax(matrix,len(get_available_slots(obs['action_mask'])),True)
    keys = [k for k, v in positions.items() if v == [result[0],result[1]]]
    return int(keys[0])

def play_human_agent(agent,obs):
    print("Available positions: ",get_available_slots(obs['action_mask']))
    while True:
        try:
            a = int(input("Movement: "))
            if a in get_available_slots(obs['action_mask']):
                return a
            else:
                print("Bad movement")
        except:
            print("Bad movement")

def minmax_agent_vs_minmax_agent():
    a = 0
    for i in range(0,100):
        env = tictactoe_v3.env(render_mode='human')
        env.reset()
        not_finish = True
        while not_finish:
            for agent in ['player_1','player_2']:
                observation, reward, termination, truncation, info = env.last() 
                if termination or truncation:
                    not_finish = False
                else:
                    if agent == 'player_1':
                        action = play_min_max_agent(agent,observation)  # this is where you would insert your policy/algorithm
                        
                    else:
                        action = play_min_max_agent(agent,observation) # TODO change
                    print(f'play: ',action)
                    env.step(action)
        if env.rewards['player_1'] == -1:
            a+=1
        print(env.rewards)
    print("Losts: ",a)

def minmax_agent_vs_random_agent():
    a = 0
    for i in range(0,1000):
        env = tictactoe_v3.env(render_mode='human')
        env.reset()
        not_finish = True
        while not_finish:
            for agent in ['player_1','player_2']:
                observation, reward, termination, truncation, info = env.last() 
                if termination or truncation:
                    not_finish = False
                else:
                    if agent == 'player_1':
                        action = play_min_max_agent(agent,observation)  # this is where you would insert your policy/algorithm
                        
                    else:
                        action = play_random_agent(agent,observation,env) # TODO change
                    print(f'play: ',action)
                    env.step(action)
        if env.rewards['player_1'] == -1:
            a+=1
        print(env.rewards)
    print("Losts: ",a)

def minmax_agent_vs_human():
    env = tictactoe_v3.env(render_mode='human')
    env.reset()
    not_finish = True
    while not_finish:
        for agent in ['player_1','player_2']:
            observation, reward, termination, truncation, info = env.last() 
            if termination or truncation:
                not_finish = False
            else:
                if agent == 'player_1':
                    action = play_min_max_agent(agent,observation)  # this is where you would insert your policy/algorithm
                    
                else:
                    action = play_human_agent(agent,observation) 
                print(f'play: ',action)
                env.step(action)
    if env.rewards['player_1'] == 1:
        print("You Loose")
    if env.rewards['player_1'] == 0:
        print("Draw")
    
if __name__ == "__main__":
    print("Wich game do you want to see/play?\n1-minmax_agent_vs_minmax_agent\n2-minmax_agent_vs_random_agent\n3-minmax_agent_vs_Human")
    ans = int(input("Answer: "))
    if ans == 1:
        print("Showing 100 games")
        minmax_agent_vs_minmax_agent()
    elif ans == 2:
        print("Showing 1000 games")
        minmax_agent_vs_random_agent()
    elif ans == 3:
        minmax_agent_vs_human()
    else:
        print("Bad Answer")
        