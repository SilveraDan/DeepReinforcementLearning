import numpy as np
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environnements.lineworld import LineWorld
from environnements.gridworld import GridWorld
from environnements.rps import TwoRoundRPS
from environnements.montyhall1 import MontyHall1
from environnements.montyhall2 import MontyHall2

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def policy_evaluation(policy, P, S, R, gamma=0.999, theta=1e-6):
    V = np.zeros(len(S))
    while True:
        delta = 0
        for s in range(len(S)):
            v = sum(policy[s][a] * sum(P[s, a, s_p, r] * (R[r] + gamma * V[s_p])
                                        for s_p in range(len(S))
                                        for r in range(len(R)))
                    for a in range(len(policy[s])))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

def policy_improvement(policy, V, P, S, A, R, gamma=0.999):
    new_policy = np.zeros_like(policy)
    policy_stable = True
    for s in range(len(S)):
        action_values = np.array([sum(P[s, a, s_p, r] * (R[r] + gamma * V[s_p])
                                      for s_p in range(len(S))
                                      for r in range(len(R)))
                                  for a in range(len(A))])
        best_action = np.argmax(action_values)
        new_policy[s, best_action] = 1.0
        if np.argmax(policy[s]) != best_action:
            policy_stable = False
    return new_policy, policy_stable

def play_game(env, policy, P, R, T, state_space):
    state = env.reset()
    total_reward = 0
    steps = 0
    visited_states = []
    while True:
        env.render()
        state_idx = state_space.index(state) if state in state_space else len(state_space) - 1
        action = np.argmax(policy[state_idx])
        next_state, reward, done = env.step(action)
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        state = next_state
        total_reward += reward
        steps += 1
        visited_states.append(state)
        if done or steps > 100 or detect_loop(visited_states):
            break
    return steps, total_reward

def detect_loop(visited_states, threshold=10):
    if len(visited_states) < threshold:
        return False
    return len(set(visited_states[-threshold:])) == 1

def create_transition_matrix(env, state_space, action_space):
    num_states = len(state_space)
    num_actions = len(action_space)
    num_rewards = len(env.rewards)

    P = np.zeros((num_states, num_actions, num_states + 1, num_rewards))

    for s in range(num_states):
        for a in range(num_actions):
            env.reset()
            env.state = state_space[s]
            next_state, reward, done = env.step(a)
            next_state_idx = state_space.index(next_state) if next_state in state_space else num_states
            reward_idx = env.rewards.index(reward)
            P[s, a, next_state_idx, reward_idx] = 1.0

    return P

def policy_iteration(env_class, config, game_name):
    env = env_class(config)
    S = config['states']
    A = config['actions']
    R = config['rewards']
    T = config['terminals']

    if game_name in ["LineWorld", "GridWorld"]:
        P = env.transition_matrix
    else:
        P = create_transition_matrix(env, S, A)

    # Initial policy: favor switching in Monty Hall problems
    if game_name.startswith("MontyHall"):
        policy = np.zeros((len(S), len(A)))
        policy[:, 1] = 1.0  # Always switch
    else:
        policy = np.ones((len(S), len(A))) / len(A)

    iteration = 0
    while True:
        V = policy_evaluation(policy, P, S, R)
        new_policy, policy_stable = policy_improvement(policy, V, P, S, A, R)
        iteration += 1
        print(f"Iteration: {iteration}")
        steps, total_reward = play_game(env, new_policy, P, R, T, S)
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward}")
        if policy_stable or iteration >= 1:
            break
        policy = new_policy
    return policy, V

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    config = load_config(config_path)

    print("Running policy iteration for LineWorld")
    lineworld_config = config['LineWorld']
    policy, V = policy_iteration(LineWorld, lineworld_config, "LineWorld")
    print("Optimal Policy for LineWorld:", policy)
    print("Value Function for LineWorld:", V)

    print("Running policy iteration for GridWorld")
    gridworld_config = config['GridWorld']
    policy, V = policy_iteration(GridWorld, gridworld_config, "GridWorld")
    print("Optimal Policy for GridWorld:", policy)
    print("Value Function for GridWorld:", V)

    print("Running policy iteration for TwoRoundRPS")
    tworoundrps_config = config['TwoRoundRPS']
    policy, V = policy_iteration(TwoRoundRPS, tworoundrps_config, "TwoRoundRPS")
    print("Optimal Policy for TwoRoundRPS:", policy)
    print("Value Function for TwoRoundRPS:", V)

    print("Running policy iteration for MontyHall1")
    montyhall1_config = config['MontyHall1']
    policy, V = policy_iteration(MontyHall1, montyhall1_config, "MontyHall1")
    print("Optimal Policy for MontyHall1:", policy)
    print("Value Function for MontyHall1:", V)

    print("Running policy iteration for MontyHall2")
    montyhall2_config = config['MontyHall2']
    policy, V = policy_iteration(MontyHall2, montyhall2_config, "MontyHall2")
    print("Optimal Policy for MontyHall2:", policy)
