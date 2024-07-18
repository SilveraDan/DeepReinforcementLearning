import numpy as np
from environnements.lineworld import play_game


def policy_evaluation(policy, P, S, R, gamma=0.999, theta=1e-6):
    V = np.random.random(len(S))
    while True:
        delta = 0
        for s in S:
            v = sum(policy[s][a] * sum(P[s, a, s_p, r] * (R[r] + gamma * V[s_p])
                                        for s_p in S
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
    for s in S:
        action_values = np.array([sum(P[s, a, s_p, r] * (R[r] + gamma * V[s_p])
                                      for s_p in S
                                      for r in range(len(R)))
                                  for a in A])
        best_action = np.argmax(action_values)
        new_policy[s, best_action] = 1.0
        if np.argmax(policy[s]) != best_action:
            policy_stable = False
    return new_policy, policy_stable


def policy_iteration(P, S, A, R, gamma=0.999):
    # Initialize policy as a 2D array with uniform distribution over actions
    policy = np.ones((len(S), len(A))) / len(A)
    iteration = 0
    while True:
        V = policy_evaluation(policy, P, S, R, gamma)
        new_policy, policy_stable = policy_improvement(policy, V, P, S, A, R, gamma)
        iteration += 1
        print(f"Iteration: {iteration}")
        steps, total_reward = play_game(policy, P, R)
        print(f"Steps: {steps}")
        print(f"Total Reward: {total_reward}")
        if policy_stable:
            break
        policy = new_policy
    return policy, V

