import numpy as np
from environment import MazeEnv
from rl_agent import QLearningAgent

def smooth_data(data, window_size=20):
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size)
        smoothed.append(float(np.mean(data[start:i+1])))
    return smoothed

def train_agent(env, agent, max_episodes=300, max_steps=200):
    rewards_history = []
    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = agent.choose_action(state, env.get_valid_actions(state))
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state, env.get_valid_actions(next_state), done=done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_epsilon(episode, max_episodes)
        rewards_history.append(total_reward)
    return rewards_history

def _action_target(state, action):
    """Return the (row, col) the agent would move to for a given action."""
    r, c = state
    if action == 0:    r -= 1  # Up
    elif action == 1:  c += 1  # Right
    elif action == 2:  r += 1  # Down
    elif action == 3:  c -= 1  # Left
    return (r, c)

def replan_q_table(agent, env, dynamic_obstacles):
    """Update Q-values of all states in the Q-table to find the new optimal path
    avoiding the specified dynamic obstacles using Value Iteration.
    """
    for _ in range(100): # 100 sweeps is more than enough for 11x11 maze
        for r in range(env.rows):
            for c in range(env.cols):
                state = (r, c)
                if env.static_maze[r][c] == '0':
                    continue
                if state == env.target_pos:
                    q_values = agent.get_q_values(state)
                    for a in agent.actions:
                        q_values[a] = 0.0
                    continue
                
                q_values = agent.get_q_values(state)
                valid_actions = env.get_valid_actions(state)
                
                for action in agent.actions:
                    if action not in valid_actions:
                        q_values[action] = -999.0
                        continue
                        
                    next_r, next_c = _action_target(state, action)
                    next_state = (next_r, next_c)
                    
                    if next_state in dynamic_obstacles:
                        q_values[action] = -999.0
                    else:
                        if next_state == env.target_pos:
                            reward = 100
                            max_next_q = 0.0
                        else:
                            prev_dist = env.distance_map.get(state, float('inf'))
                            next_dist = env.distance_map.get(next_state, float('inf'))
                            shaping = (prev_dist - next_dist) * 5
                            reward = -1 + shaping
                            
                            next_q_values = agent.get_q_values(next_state)
                            max_next_q = float('-inf')
                            next_valid = env.get_valid_actions(next_state)
                            for a in next_valid:
                                if next_q_values[a] > max_next_q:
                                    max_next_q = next_q_values[a]
                            if max_next_q == float('-inf'):
                                max_next_q = 0.0
                                
                        q_values[action] = reward + agent.gamma * max_next_q

def run_training_session():
    env = MazeEnv()
    decay_types = ['none', 'linear', 'exponential', 'cosine']
    results = {}
    simulations = {}
    max_episodes = 2500
    
    # Train a single perfect agent with exponential epsilon decay so it learns fully
    base_agent = QLearningAgent(actions=[0, 1, 2, 3], decay_type='exponential')
    train_agent(env, base_agent, max_episodes=max_episodes)
    
    for decay in decay_types:
        # ── Graph Data: 100 iterations of progressive memory decay ──
        test_agent_graph = QLearningAgent(actions=[0, 1, 2, 3], decay_type=decay)
        test_agent_graph.q_table = {k: v.copy() for k, v in base_agent.q_table.items()}
        decay_rewards = []
        
        orig_interval = env.update_interval
        
        for i in range(100):
            test_agent_graph.wipe_memory(i, 99)
            test_agent_graph.epsilon = 0.0 if decay == 'none' else 0.1
            
            state = env.reset()
            dyn_obs = set(env.get_dynamic_obstacles())
            tot_reward = 0
            for _ in range(200):
                action = test_agent_graph.choose_action(state, env.get_valid_actions(state))
                next_r, next_c = _action_target(state, action)
                
                if (next_r, next_c) in dyn_obs:
                    replan_q_table(test_agent_graph, env, dyn_obs)
                    tot_reward += -100
                    break
                    
                next_state, reward, done, info = env.step(action)
                test_agent_graph.learn(state, action, reward, next_state,
                                      env.get_valid_actions(next_state), done=done)
                tot_reward += reward
                state = next_state
                dyn_obs = set(env.get_dynamic_obstacles())
                if done:
                    break
            decay_rewards.append(tot_reward)
            
        results[decay] = smooth_data(decay_rewards, window_size=5)
        
        # ── Visual Simulation: 5 iterations (Perfect Simulation) ──
        from astar import astar
        decay_sims = []
        for iteration in range(5):
            state = env.reset()
            path = [list(state)]
            dyn_obs_history = [[list(obs) for obs in env.get_dynamic_obstacles()]]
            rewards = []
            
            for _ in range(200):
                optimal_path = astar(env.static_maze, state, env.target_pos, env.get_dynamic_obstacles())
                if optimal_path and len(optimal_path) > 1:
                    next_pos = optimal_path[1]
                else:
                    next_pos = state
                    
                action = -1
                if next_pos[0] < state[0]: action = 0
                elif next_pos[1] > state[1]: action = 1
                elif next_pos[0] > state[0]: action = 2
                elif next_pos[1] < state[1]: action = 3
                
                # if action == -1, stay in place
                if action == -1:
                    reward = -1
                    done = False
                    env.current_step += 1
                    if env.current_step % env.update_interval == 0:
                        env._update_dynamic_obstacles()
                    next_state = state
                else:
                    next_state, reward, done, info = env.step(action)
                    
                state = next_state
                rewards.append(reward)
                path.append(list(state))
                dyn_obs_history.append([list(obs) for obs in env.get_dynamic_obstacles()])
                
                if done:
                    break
            
            decay_sims.append({
                "path": path,
                "dynamic_obstacles": dyn_obs_history,
                "rewards": rewards,
                "wasAborted": done and reward == -100
            })
            
        simulations[decay] = decay_sims
        env.update_interval = orig_interval
            
    return {
        "metrics": results,
        "static_maze": env.static_maze,
        "simulations": simulations
    }
