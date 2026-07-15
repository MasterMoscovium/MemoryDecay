import os
import time

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_maze(maze, agent_pos=None, dynamic_obstacles=None):
    """
    Prints the maze using emojis.
    maze: 2D list where '0' = wall, '1' = path, '2' = target
    agent_pos: tuple (r, c)
    dynamic_obstacles: list of tuples [(r, c)]
    """
    maze_copy = [row[:] for row in maze]
    
    # Place dynamic obstacles
    if dynamic_obstacles:
        for r, c in dynamic_obstacles:
            if 0 <= r < len(maze) and 0 <= c < len(maze[0]):
                maze_copy[r][c] = 'D' # D for Dynamic Obstacle
            
    # Place agent
    if agent_pos:
        r, c = agent_pos
        if 0 <= r < len(maze) and 0 <= c < len(maze[0]):
            maze_copy[r][c] = 'A'
        
    for i in range(len(maze_copy)):
        for j in range(len(maze_copy[0])):
            if maze_copy[i][j] == '0':
                maze_copy[i][j] = '🟦'
            elif maze_copy[i][j] == '1':
                maze_copy[i][j] = '🟨'
            elif maze_copy[i][j] == '2':
                maze_copy[i][j] = '🎯'
            elif maze_copy[i][j] == 'D':
                maze_copy[i][j] = '🟥' # Red block for dynamic obstacle
            elif maze_copy[i][j] == 'A':
                maze_copy[i][j] = '🤖' # Robot for agent
                
    clear_console()
    print('\n'.join(''.join(row) for row in maze_copy))
    print()

def animate_path(maze, path, dynamic_obstacle_states, delay=0.2):
    """
    path: list of (r, c) tuples
    dynamic_obstacle_states: list of lists of (r, c) for each step
    """
    for i, step in enumerate(path):
        dyn_obs = dynamic_obstacle_states[i] if i < len(dynamic_obstacle_states) else []
        print_maze(maze, agent_pos=step, dynamic_obstacles=dyn_obs)
        time.sleep(delay)
