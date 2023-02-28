import math
import numpy as np

positions = {'0':[0,0],'1':[1,0],'2':[2,0],'3':[0,1],'4':[1,1],'5':[2,1],'6':[0,2],'7':[1,2],'8':[2,2]}


def fill_matrix(obs):
    template = np.zeros((3,3))
    movements = obs['action_mask']
    for i in range(len(movements)):
        if movements[i] == 0:
            a = positions[str(i)]
            if obs['observation'][a[1]][a[0]][0] == 1:
                template[tuple(a)] = 1
            else:
                template[tuple(a)] = -1
    return template

def get_available_slots(action_mask):
    out = []
    for i in range(len(action_mask)):
        if action_mask[i] == 1:
            out.append(i)
    return out

def evaluate(state, agent) :
    for row in range(3) :    
        if (state[row][0] == state[row][1] and state[row][1] == state[row][2]) :       
            if (state[row][0] == agent) :
                return 1
            elif (state[row][0] == -agent) :
                return -1
    for col in range(3) :
        if (state[0][col] == state[1][col] and state[1][col] == state[2][col]) :
            if (state[0][col] == agent) :
                return 1
            elif (state[0][col] == -agent) :
                return -1
    if (state[0][0] == state[1][1] and state[1][1] == state[2][2]) :
        if (state[0][0] == agent) :
            return 1
        elif (state[0][0] == -agent) :
            return -1
    if (state[0][2] == state[1][1] and state[1][1] == state[2][0]) :
        if (state[0][2] == agent) :
            return 1
        elif (state[0][2] == -agent) :
            return -1
    return 0

def available_slots(state):
    slots = []
    for x, row in enumerate(state):
        for y, slot in enumerate(row):
            if slot == 0:
                slots.append([x, y])

    return slots

def simulate_state(state,maximizingPlayer,x,y):
    if maximizingPlayer:
        state[x][y] = 1
    else:
        state[x][y] = -1
    return state

def undo_state(state,x,y):
    state[x][y] = 0
    return state

def minMax(state, depth, maximizingPlayer):
    if depth == 0 or evaluate(state,1) != 0:
        score = evaluate(state,1)
        return [-1, -1, score]
        
    if maximizingPlayer:
        best = [-1, -1, -math.inf]
    else:
        best = [-1, -1, +math.inf]

    for slot in available_slots(state):
        state = simulate_state(state,maximizingPlayer,slot[0], slot[1])
        score = minMax(state, depth - 1, not maximizingPlayer)
        state = undo_state(state,slot[0], slot[1])
        score[0], score[1] = slot[0], slot[1]

        if maximizingPlayer:
            if score[2] > best[2]:
                best = score
        else:
            if score[2] < best[2]:
                best = score 
    return best