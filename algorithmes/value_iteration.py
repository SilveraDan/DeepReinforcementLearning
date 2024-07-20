import numpy as np
import yaml
import sys
import os

# Ajouter le répertoire principal au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les environnements
from environnements.rps import TwoRoundRPS
from environnements.montyhall1 import MontyHall1
from environnements.montyhall2 import MontyHall2
from environnements.lineworld import LineWorld
from environnements.gridworld import GridWorld

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transitions(env, state, action):
    current_state = state  # Save the current state
    env.state = state  # Set to the current state
    if hasattr(env, 'selected_doors'):
        current_selected_doors = env.selected_doors.copy()  # Save current selected doors for MontyHall2
    next_state, reward, done = env.step(action)
    prob = 1.0  # Assuming deterministic transitions for simplicity
    env.state = current_state  # Reset state to the original after simulating step
    if hasattr(env, 'selected_doors'):
        env.selected_doors = current_selected_doors  # Reset selected doors for MontyHall2
    if next_state == 'terminal':
        next_state_index = len(env.states)
    else:
        next_state_index = env.states.index(next_state) if next_state in env.states else len(env.states)
    return [(prob, next_state_index, reward)]

class ValueIteration:
    def __init__(self, env, discount_factor=0.99, theta=0.0001, max_iterations=1000, epsilon=0.1, max_steps=100):
        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.epsilon = epsilon  # Epsilon for exploration
        self.max_steps = max_steps  # Max steps to detect looping
        self.value_table = np.zeros(len(env.states) + 1)  # Adding 1 for terminal state
        self.action_counts = np.zeros((len(env.states) + 1, self.env.action_space.n))  # Tracking action counts

    def run(self):
        iteration = 0
        while iteration < self.max_iterations:
            delta = 0
            for state in range(len(self.env.states)):
                v = self.value_table[state]
                max_value = float('-inf')
                for action in range(self.env.action_space.n):
                    transitions = get_transitions(self.env, self.env.states[state], action)
                    value = sum([prob * (reward + self.discount_factor * self.value_table[next_state])
                                 for prob, next_state, reward in transitions])
                    # Penalize repeated actions
                    value -= self.action_counts[state][action] * 0.1  # Penalty factor
                    max_value = max(max_value, value)
                self.value_table[state] = max_value
                delta = max(delta, abs(v - self.value_table[state]))
            iteration += 1
            if delta < self.theta:
                break
        return self.value_table

    def extract_policy(self):
        policy = np.zeros(len(self.env.states) + 1, dtype=int)  # Adding 1 for terminal state
        for state in range(len(self.env.states)):
            max_value = float('-inf')
            best_action = None
            for action in range(self.env.action_space.n):
                transitions = get_transitions(self.env, self.env.states[state], action)
                value = sum([prob * (reward + self.discount_factor * self.value_table[next_state])
                             for prob, next_state, reward in transitions])
                # Penalize repeated actions
                value -= self.action_counts[state][action] * 0.1  # Penalty factor
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[state] = best_action
        return policy

    def epsilon_greedy_policy(self, policy):
        def policy_fn(state):
            if np.random.rand() < self.epsilon:
                return np.random.choice(self.env.action_space.n)
            else:
                return policy[state]
        return policy_fn

    def update_action_counts(self, state, action):
        self.action_counts[state][action] += 1

def visualize_policy(env, policy_fn, vi, max_steps=100):
    state = env.reset()
    steps = 0
    visited_states_actions = set()
    while True:
        state_index = env.states.index(state) if state in env.states else len(env.states)
        action = policy_fn(state_index)
        
        # Check for loops and force exploration if stuck
        if (state_index, action) in visited_states_actions:
            action = np.random.choice(env.action_space.n)
        
        visited_states_actions.add((state_index, action))
        
        next_state, reward, done = env.step(action)
        env.render()
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        # Update action counts
        vi.update_action_counts(state_index, action)
        if next_state == 'terminal':
            break
        state = next_state
        steps += 1
        if done or steps >= max_steps:
            print(f"Episode finished in {steps} steps.")
            break
        if steps >= vi.max_steps:
            print(f"Detected potential loop at step {steps}. Changing strategy.")
            state = env.reset()
            steps = 0
            visited_states_actions.clear()

if __name__ == '__main__':
    # Chemin vers le fichier de configuration
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')

    # Charger la configuration
    config = load_config(config_path)

    # Tester l'algorithme pour LineWorld
    lineworld_config = config['LineWorld']
    env = LineWorld(lineworld_config)
    vi = ValueIteration(env)
    value_table = vi.run()
    policy = vi.extract_policy()
    epsilon_greedy_policy = vi.epsilon_greedy_policy(policy)
    print("LineWorld Value Table:", value_table)
    print("LineWorld Policy:", policy)
    visualize_policy(env, epsilon_greedy_policy, vi)

    # Tester l'algorithme pour GridWorld
    gridworld_config = config['GridWorld']
    env = GridWorld(gridworld_config)
    vi = ValueIteration(env)
    value_table = vi.run()
    policy = vi.extract_policy()
    epsilon_greedy_policy = vi.epsilon_greedy_policy(policy)
    print("GridWorld Value Table:", value_table)
    print("GridWorld Policy:", policy)
    visualize_policy(env, epsilon_greedy_policy, vi)

    # Tester l'algorithme pour TwoRoundRPS
    tworoundrps_config = config['TwoRoundRPS']
    env = TwoRoundRPS(tworoundrps_config)
    vi = ValueIteration(env)
    value_table = vi.run()
    policy = vi.extract_policy()
    epsilon_greedy_policy = vi.epsilon_greedy_policy(policy)
    print("TwoRoundRPS Value Table:", value_table)
    print("TwoRoundRPS Policy:", policy)
    visualize_policy(env, epsilon_greedy_policy, vi)

    # Tester l'algorithme pour MontyHall1
    montyhall1_config = config['MontyHall1']
    env = MontyHall1(montyhall1_config)
    vi = ValueIteration(env)
    value_table = vi.run()
    policy = vi.extract_policy()
    epsilon_greedy_policy = vi.epsilon_greedy_policy(policy)
    print("MontyHall1 Value Table:", value_table)
    print("MontyHall1 Policy:", policy)
    visualize_policy(env, epsilon_greedy_policy, vi)

    # Tester l'algorithme pour MontyHall2
    montyhall2_config = config['MontyHall2']
    env = MontyHall2(montyhall2_config)
    vi = ValueIteration(env)
    value_table = vi.run()
    policy = vi.extract_policy()
    epsilon_greedy_policy = vi.epsilon_greedy_policy(policy)
    print("MontyHall2 Value Table:", value_table)
    print("MontyHall2 Policy:", policy)
    visualize_policy(env, epsilon_greedy_policy, vi)
