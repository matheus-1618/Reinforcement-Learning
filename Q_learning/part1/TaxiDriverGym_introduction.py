import gymnasium as gym

env = gym.make("Taxi-v3", render_mode='ansi').env

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print('\n\n')

state = env.reset()
print(state[0])
print(env.render())

# escolhe uma acao aleatoria
action = env.action_space.sample()
print(action)

# executa a acao
state, reward, done, truncated, info = env.step(action)
print(state)
print(env.render())

# executa a acao ir para north
state, reward, done, truncated, info = env.step(1)
print(env.render())

# executa a acao ir para south
state, reward, done, truncated, info = env.step(0)
print(env.render())

# actions:
# 0 = south
# 1 = north
# 2 = east
# 3 = west
# 4 = pickup
# 5 = dropoff

env.close()