import random as rd
from environment import GridWorld


debug_mode = False
show_graph_every = 1


env = GridWorld(show_graph_every, debug_mode)



state = env.reset()
while True:
    action = rd.randint(1,7)
    state, reward, total_reward, new_state, done = env.step(action)
    if done == True:
        state = env.reset() 