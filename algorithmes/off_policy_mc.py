import numpy as np
import random
from collections import defaultdict, deque
import yaml
import sys
import os

# Ajouter le répertoire principal au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer les environnements
from environnements.rps import TwoRoundRPS
from environnements.montyhall1 import MontyHall1
from environnements.montyhall2 import MontyHall2
from environnements.lineworld2 import LineWorld
from environnements.gridworld2 import GridWorld

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def epsilon_greedy_policy(Q, state, n_actions, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(n_actions))
    else:
        return np.argmax(Q[state])

def detect_loop(steps, max_repeated_states=10):
    if len(steps) < max_repeated_states:
        return False
    recent_states = list(steps)
    return len(set(recent_states)) < len(recent_states)

def off_policy_mc_control(env, num_episodes, gamma=0.99, epsilon=0.2, epsilon_decay=0.99, min_epsilon=0.05, max_steps=100, max_repeated_states=10):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    target_policy = defaultdict(int)
    state_action_counts = defaultdict(lambda: defaultdict(int))
    
    for ep in range(num_episodes):
        episode = []
        state = env.reset()
        current_epsilon = max(min_epsilon, epsilon * (epsilon_decay ** ep))
        steps = deque(maxlen=max_repeated_states)
        steps.append(state)
        
        for t in range(max_steps):
            action = epsilon_greedy_policy(Q, state, env.action_space.n, current_epsilon)
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            else:
                next_state, reward, done = step_result

            # Pénalité pour chaque étape pour encourager une solution rapide
            reward -= 0.01

            episode.append((state, action, reward))
            steps.append(next_state)
            state_action_counts[state][action] += 1
            if done or detect_loop(steps, max_repeated_states):
                break
            state = next_state
        
        G = 0
        W = 1.0
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            target_policy[state] = np.argmax(Q[state])
            if action != target_policy[state]:
                break
            W *= 1.0 / ((current_epsilon / env.action_space.n) + (1 - current_epsilon) * (action == target_policy[state]))
            if W == 0:
                break
    
    return Q, target_policy

def visualize_policy(env, policy):
    state = env.reset()
    steps = 0
    while True:
        action = policy[state]
        step_result = env.step(action)
        if len(step_result) == 4:
            next_state, reward, done, _ = step_result
        else:
            next_state, reward, done = step_result
        if hasattr(env, 'render'):
            env.render()
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        state = next_state
        steps += 1
        if done or steps > 100:
            print(f"Episode finished in {steps} steps.")
            break

if __name__ == '__main__':
    # Chemin vers le fichier de configuration
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')

    # Charger la configuration
    config = load_config(config_path)

    # Tester l'algorithme pour LineWorld
    lineworld_config = config['LineWorld']
    env = LineWorld(lineworld_config)
    Q, policy = off_policy_mc_control(env, num_episodes=1000)
    print("LineWorld Q-values:", Q)
    print("LineWorld Policy:", policy)
    visualize_policy(env, policy)

    # Tester l'algorithme pour GridWorld
    gridworld_config = config['GridWorld']
    env = GridWorld(gridworld_config)
    Q, policy = off_policy_mc_control(env, num_episodes=1000)
    print("GridWorld Q-values:", Q)
    print("GridWorld Policy:", policy)
    visualize_policy(env, policy)

    # Tester l'algorithme pour TwoRoundRPS
    tworoundrps_config = config['TwoRoundRPS']
    env = TwoRoundRPS(tworoundrps_config)
    Q, policy = off_policy_mc_control(env, num_episodes=2000, gamma=0.99, epsilon=0.2, epsilon_decay=0.99, min_epsilon=0.05)
    print("TwoRoundRPS Q-values:", Q)
    print("TwoRoundRPS Policy:", policy)
    visualize_policy(env, policy)

    # Tester l'algorithme pour MontyHall1
    montyhall1_config = config['MontyHall1']
    env = MontyHall1(montyhall1_config)
    Q, policy = off_policy_mc_control(env, num_episodes=1000)
    print("MontyHall1 Q-values:", Q)
    print("MontyHall1 Policy:", policy)
    visualize_policy(env, policy)

    # Tester l'algorithme pour MontyHall2
    montyhall2_config = config['MontyHall2']
    env = MontyHall2(montyhall2_config)
    Q, policy = off_policy_mc_control(env, num_episodes=1000)
    print("MontyHall2 Q-values:", Q)
    print("MontyHall2 Policy:", policy)
    visualize_policy(env, policy)
