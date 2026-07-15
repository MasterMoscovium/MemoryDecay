import random
import copy

class MazeEnv:
    def __init__(self):
        # 11x11 maze
        # 0: wall, 1: path, 2: target
        self.static_maze = [
            ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0'],
            ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0'],
            ['0', '1', '0', '1', '1', '0', '1', '1', '0', '1', '0'],
            ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0'],
            ['0', '1', '1', '0', '1', '0', '1', '0', '1', '1', '0'],
            ['0', '1', '1', '1', '1', '0', '1', '1', '1', '1', '0'],
            ['0', '1', '1', '0', '1', '0', '1', '0', '1', '1', '0'],
            ['0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0'],
            ['0', '1', '0', '1', '1', '0', '1', '1', '0', '1', '0'],
            ['0', '1', '1', '1', '1', '1', '1', '1', '1', '2', '0'],
            ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']
        ]
        self.start_pos = (1, 1)
        self.target_pos = (9, 9)
        self.rows = len(self.static_maze)
        self.cols = len(self.static_maze[0])
        
        # Potential locations for dynamic obstacles (choke points)
        self.potential_dynamic_obstacles = [(1, 5), (3, 5), (7, 5), (9, 5)]
        
        self.dynamic_obstacles = []
        self.current_step = 0
        self.update_interval = 5 # Dynamic obstacles change every 5 steps
        
        self._build_distance_map()
        self.reset()
        
    def _build_distance_map(self):
        self.distance_map = {}
        for r in range(self.rows):
            for c in range(self.cols):
                self.distance_map[(r, c)] = float('inf')
                
        queue = [(self.target_pos[0], self.target_pos[1], 0)]
        self.distance_map[self.target_pos] = 0
        visited = {self.target_pos}
        
        while queue:
            r, c, dist = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.static_maze[nr][nc] != '0':
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        self.distance_map[(nr, nc)] = dist + 1
                        queue.append((nr, nc, dist + 1))
                        
    def reset(self):
        self.agent_pos = self.start_pos
        self.current_step = 0
        self._update_dynamic_obstacles()
        return self.get_state()
        
    def _update_dynamic_obstacles(self):
        """Randomly select a subset of potential locations to have dynamic obstacles."""
        self.dynamic_obstacles = random.sample(
            self.potential_dynamic_obstacles, 
            k=random.randint(1, 2) # 1 to 2 active dynamic obstacles
        )
        

        
    def get_state(self):
        return self.agent_pos
        
    def get_dynamic_obstacles(self):
        return self.dynamic_obstacles
        
    def step(self, action):
        """
        Actions: 0: Up, 1: Right, 2: Down, 3: Left
        Returns: next_state, reward, done
        """
        r, c = self.agent_pos
        next_r, next_c = r, c
        
        if action == 0: # Up
            next_r -= 1
        elif action == 1: # Right
            next_c += 1
        elif action == 2: # Down
            next_r += 1
        elif action == 3: # Left
            next_c -= 1
            
        # Check bounds and static walls
        hit_wall = False
        if (next_r < 0 or next_r >= self.rows or 
            next_c < 0 or next_c >= self.cols or 
            self.static_maze[next_r][next_c] == '0'):
            hit_wall = True
            
        # Check dynamic obstacles
        hit_dynamic = (next_r, next_c) in self.dynamic_obstacles
        
        if hit_wall:
            # Stay in place
            reward = -10 # Penalty for hitting static wall
            done = False
        elif hit_dynamic:
            reward = -100 # Penalty for hitting dynamic obstacle (balanced to allow exploration)
            done = True # Instant iteration end as requested
        else:
            prev_dist = self.distance_map.get((r, c), float('inf'))
            next_dist = self.distance_map.get((next_r, next_c), float('inf'))
            
            self.agent_pos = (next_r, next_c)
            if self.agent_pos == self.target_pos:
                reward = 100
                done = True
            else:
                # Reward shaping: +5 for moving closer, -5 for moving further
                shaping = (prev_dist - next_dist) * 5
                reward = -1 + shaping
                done = False
                
        self.current_step += 1
        if self.current_step % self.update_interval == 0:
            self._update_dynamic_obstacles()
            
        return self.get_state(), reward, done, {"hit_dynamic": hit_dynamic}

    def get_valid_actions(self, state_or_pos=None):
        if state_or_pos is None:
            r, c = self.agent_pos
        else:
            # If it's a state tuple ((r, c), obs)
            if isinstance(state_or_pos, tuple) and len(state_or_pos) == 2 and isinstance(state_or_pos[0], tuple):
                r, c = state_or_pos[0]
            else:
                r, c = state_or_pos
                
        valid_actions = []
        # Up
        if r > 0 and self.static_maze[r-1][c] != '0': valid_actions.append(0)
        # Right
        if c < self.cols-1 and self.static_maze[r][c+1] != '0': valid_actions.append(1)
        # Down
        if r < self.rows-1 and self.static_maze[r+1][c] != '0': valid_actions.append(2)
        # Left
        if c > 0 and self.static_maze[r][c-1] != '0': valid_actions.append(3)
        return valid_actions
