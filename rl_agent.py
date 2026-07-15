import random
import math
import numpy as np

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, 
                 epsilon_start=1.0, epsilon_end=0.01, decay_type='none'):
        self.actions = actions # list of valid action indices, e.g., [0, 1, 2, 3]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_type = decay_type
        
        self.q_table = {}
        
    def get_q_values(self, state):
        if state not in self.q_table:
            # Initialize with small random values to break ties
            self.q_table[state] = {a: random.uniform(-0.01, 0.01) for a in self.actions}
        return self.q_table[state]

    def choose_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = self.actions
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.get_q_values(state)
            # Find the action in valid_actions with the max Q-value
            max_q = float('-inf')
            best_actions = []
            for a in valid_actions:
                if q_values[a] > max_q:
                    max_q = q_values[a]
                    best_actions = [a]
                elif q_values[a] == max_q:
                    best_actions.append(a)
            
            return random.choice(best_actions)
            
    def learn(self, state, action, reward, next_state, next_valid_actions=None, done=False, instant_penalty=False):
        if next_valid_actions is None:
            next_valid_actions = self.actions
            
        q_values = self.get_q_values(state)
        
        if instant_penalty:
            q_values[action] = -999.0
            return
            
        if done:
            max_next_q = 0.0
        else:
            next_q_values = self.get_q_values(next_state)
            max_next_q = float('-inf')
            for a in next_valid_actions:
                if next_q_values[a] > max_next_q:
                    max_next_q = next_q_values[a]
            if max_next_q == float('-inf'):
                max_next_q = 0.0
            
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - q_values[action]
        q_values[action] += self.lr * td_error
        
    def update_epsilon(self, episode, max_episodes):
        if self.decay_type == 'none':
            self.epsilon = self.epsilon_start
        elif self.decay_type == 'linear':
            # Linear decay
            decay_rate = (self.epsilon_start - self.epsilon_end) / max_episodes
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_rate)
        elif self.decay_type == 'exponential':
            # Exponential decay
            decay_rate = math.pow(self.epsilon_end / self.epsilon_start, 1.0 / max_episodes)
            self.epsilon = max(self.epsilon_end, self.epsilon * decay_rate)
        elif self.decay_type == 'cosine':
            # Cosine decay
            fraction = episode / max_episodes
            self.epsilon = self.epsilon_end + 0.5 * (self.epsilon_start - self.epsilon_end) * (1 + math.cos(fraction * math.pi))
            
    def wipe_memory(self, iteration, max_iterations):
        if self.decay_type == 'none' or iteration == 0:
            forget_fraction = 0.0
        elif self.decay_type == 'linear':
            forget_fraction = iteration / max_iterations
        elif self.decay_type == 'exponential':
            forget_fraction = 1.0 - math.exp(-2 * iteration / max_iterations)
        elif self.decay_type == 'cosine':
            fraction = iteration / max_iterations
            forget_fraction = 0.5 * (1 - math.cos(fraction * math.pi))
            
        if forget_fraction > 0:
            import random
            num_to_wipe = int(len(self.q_table) * forget_fraction)
            if num_to_wipe > 0:
                keys = list(self.q_table.keys())
                random.shuffle(keys)
                for k in keys[:num_to_wipe]:
                    for a in self.actions:
                        self.q_table[k][a] = 0.0
            
    def reset_agent(self):
        self.q_table = {}
        self.epsilon = self.epsilon_start
