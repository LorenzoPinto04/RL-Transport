import frame_runner as fr, math, random, pandas as pd, numpy as np

def random_spawn(x, y):
    return (random.randint(0, x - 2), random.randint(0, y - 2))


def create_grid(n_x, n_y, agent_pos, target_pos, transport_timetable, timestep):
    grid = np.zeros((n_y, n_x))
    for i, row in transport_timetable.iterrows():
        if row[0] == timestep:
            grid[row[3]-1][row[2]-1] = 3
    grid[agent_pos[1]][agent_pos[0]] = 1
    grid[target_pos[1]][target_pos[0]] = 2
    return grid


def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def get_mean_pos(name, timestep):
    df = pd.read_csv('data/test_data.csv')
    try:
        mean_pos = (
         int(df.loc[((df['0'] == timestep) & (df['1'] == name))]['2']), int(df.loc[((df['0'] == timestep) & (df['1'] == name))]['3']))
        return mean_pos
    except:
        return (0, 0)


class GridWorld:

    def __init__(self, show_graph_every=1, debug_mode=False):
        self.n_x = 80
        self.n_y = 50
        self.done = False
        self.reward = 0
        self.total_reward = 0
        self.agent_pos = random_spawn(self.n_x, self.n_y)
        self.target_pos = random_spawn(self.n_x, self.n_y)
        self.timestep = 0
        self.target_achieved = 0
        self.target_missed = 0
        self.n_iteration = 1
        self.transport_timetable = pd.read_csv('data/test_data.csv')
        self.final_reward = 0
        self.transport_used = False
        #self.max_steps = n_x + n_y
        self.max_steps = 60
        
        # visualization attributes 
        self.show_graph_every = show_graph_every
        self.frame_second = 3
        self.debug_mode = debug_mode

    def move_right(self):
        if self.debug_mode:
            print('move right')
        x = self.agent_pos[0]
        if x < self.n_x - 1:
            self.agent_pos = (
             x + 1, self.agent_pos[1])
        else:
            self.reward -= 2

    def move_left(self):
        if self.debug_mode:
            print('move left')
        x = self.agent_pos[0]
        if x > 0:
            self.agent_pos = (
             x - 1, self.agent_pos[1])
        else:
            self.reward -= 2

    def move_down(self):
        if self.debug_mode:
            print('move down')
        y = self.agent_pos[1]
        if y < self.n_y - 1:
            self.agent_pos = (
             self.agent_pos[0], y + 1)
        else:
            self.reward -= 2

    def move_up(self):
        if self.debug_mode:
            print('move up')
        y = self.agent_pos[1]
        if y > 0:
            self.agent_pos = (
             self.agent_pos[0], y - 1)
        else:
            self.reward -= 2

    def take_mean(self, name):
        if distance(self.agent_pos, get_mean_pos(name, self.timestep)) == 1:
            #self.extra_reward = 100
            print('-------------------------------------------------------------------------------------------ON A MEAN')
            self.agent_pos = get_mean_pos(name, self.timestep + 1)
        elif distance(self.agent_pos, get_mean_pos(name, self.timestep)) < 1:
            self.agent_pos = get_mean_pos(name, self.timestep + 1)
        else:
            self.reward -= 1

            
            
    # ------------------------ AI control ------------------------

    # 0 do nothing
    # 1 move right
    # 2 move left
    # 3 move up
    # 4 move down 
    # 5 bus_1
    # 6 bus_2
    # 7 train_1
    # 8 train_2            
       
            
    def reset(self):
        self.transport_used = False
        self.final_reward = 0
        self.total_reward = 0
        self.timestep = 0
        self.agent_pos = random_spawn(self.n_x, self.n_y)
        self.target_pos = random_spawn(self.n_x, self.n_y)
        self.grid = create_grid(self.n_x, self.n_y, self.agent_pos, self.target_pos, self.transport_timetable, self.timestep)
        state = self.grid
        return state

    def step(self, action):
        state = self.grid
        self.timestep += 1
        self.done = False
        self.reward = 0
        
        distance_0 = distance(self.agent_pos, self.target_pos)
        
        if action == 0:
            self.action_performed = 'Wait'
            self.reward -= .1
        elif action == 1:
            self.action_performed = 'Move right'
            self.move_right()
            self.reward -= 0.3
        elif action == 2:
            self.action_performed = 'Move left'
            self.move_left()
            self.reward -= 0.3
        elif action == 3:
            self.action_performed = 'Move down'
            self.move_down()
            self.reward -= 0.3
        elif action == 4:
            self.action_performed = 'Move up'
            self.move_up()
            self.reward -= 0.3
        elif action == 5:
            self.action_performed = 'Take bus 1'
            self.take_mean('bus_1')
        elif action == 6:
            self.action_performed = 'Take bus 2'
            self.take_mean('bus_2')
        elif action == 7:
            self.action_performed = 'Take train 1'
            self.take_mean('train_1')
        else:
            return
            
            
            
        self.grid = create_grid(self.n_x, self.n_y, self.agent_pos, self.target_pos, self.transport_timetable, self.timestep)
        
        distance_1 = distance(self.agent_pos, self.target_pos)
        
        delta_distance = distance_0 - distance_1
        
        if -1 < delta_distance < 1:
            delta_distance = np.sign(delta_distance)

        delta_distance = delta_distance ** 3
        self.reward += delta_distance
        
        self.final_reward = 2000 * (1/(self.timestep+50))
        
        new_state = self.grid
        self.run_frame()
        self.total_reward += self.reward
        return (state, self.reward, self.total_reward, new_state, self.done)

        
    def run_frame(self):
        if self.show_graph_every != False and self.n_iteration % self.show_graph_every == 0:
            fr.runner((self.grid), (self.agent_pos), (self.target_pos), (self.timestep), (self.transport_timetable),
                  (self.target_missed), (self.target_achieved), (self.n_iteration),
                  (self.frame_second), (self.reward), (self.final_reward), (self.action_performed),
                  (self.total_reward), debug_mode=(self.debug_mode))
        if self.agent_pos == self.target_pos:
            self.reward += self.total_reward
            self.target_achieved += 1
            self.done = True
            self.n_iteration += 1
            print('------------------------------------------------[WIN] Target achieved')
        elif self.timestep == self.max_steps:
            self.reward -= 5
            self.target_missed += 1
            self.done = True
            self.n_iteration += 1
