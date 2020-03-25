# MODULES
import pygame
import pandas as pd

def runner(grid, agent_pos, target_pos, timestep, transport_timetable, n_missed, n_achieved, n_iteration, frame_second, reward_action, possible_reward, action_performed, total_reward, debug_mode = False):
    agent_pos_x = agent_pos[0]
    agent_pos_y = agent_pos[1]
    target_pos_x = target_pos[0]
    target_pos_y = target_pos[1]


    # PARAMETERS

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)

    # WIDTH and HEIGHT of each grid location
    WIDTH = 13
    HEIGHT = 13
    # margin between each cell
    MARGIN = 5

    WINDOW_SIZE = [1500, 1000]

    def plot_grid(grid):
        for row in range(grid.shape[0]):
            for column in range(grid.shape[1]):
                color = WHITE
                if grid[row][column] == 1:
                    color = GREEN
                elif grid[row][column] == 2:
                    color = RED
                elif grid[row][column] == 3:
                    color = BLACK
                pygame.draw.rect(screen,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])   


    def write(text, x, y):
        font = pygame.font.Font('freesansbold.ttf', 15) 
        text = font.render(text, True, WHITE, RED) 
        textRect = text.get_rect()  
        textRect.center = (x, y) 
        screen.blit(text, textRect)    


    # dataset creator
    # RUN ONLY IF YOU WANT TO CREATE A NEW ROUTE
    # CODE 

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("ENVIRONMENT")
    clock = pygame.time.Clock()
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            pygame.quit()
    screen.fill(BLACK) 
    plot_grid(grid)
    write('Time: '+str(timestep), 100, 10)
    write('Iteration: '+str(n_iteration), 200, 10)
    write('Missed: '+str(n_missed), 300, 10)
    write('Achieved: '+str(n_achieved), 400, 10)
    write('Reward_action: '+str(reward_action)[:5], 550, 10)
    write('Total_reword: '+str(total_reward)[:5], 700, 10)
    write('Possible_final_reward: '+str(possible_reward)[:4], 900, 10)
    write('Action: '+str(action_performed), 1100, 10)
    clock.tick(frame_second)
    pygame.display.flip()
    next_iter = False
    if debug_mode == True:
        while next_iter == False:
            event = pygame.event.wait()
            if event.type == pygame.KEYDOWN:
                next_iter = True
    return 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    