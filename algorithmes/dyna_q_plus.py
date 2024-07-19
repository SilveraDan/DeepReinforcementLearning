import numpy as np
import random


def choose_action(state, epsilon, Q, A):
    if random.uniform(0, 1) < epsilon:
        return random.choice(A)
    else:
        return np.argmax(Q[state])


def dyna_q_plus(S, A, R, P, T, n_episodes, n_planning_steps, alpha, gamma, k, epsilon, start_state):
    q_table = np.zeros((len(S), len(A)))
    policy = np.ones((len(S), len(A))) / len(A)
    model = {}
    time_since_last_visit = np.zeros((len(S), len(A)))
    timestep = 0

    for episode in range(n_episodes):
        state = start_state

        while state not in T:
            action = choose_action(state, epsilon, q_table, A)

            # Obtenir la récompense et le nouvel état
            next_state = np.argmax(np.sum(P[state, action, :, :], axis=1))
            reward_index = np.argmax(P[state, action, next_state, :])
            reward = R[reward_index]

            # Mise à jour de la Q-table
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

            # Mise à jour du modèle
            model[(state, action)] = (reward, next_state)
            time_since_last_visit[state, action] = timestep

            state = next_state
            timestep += 1

            # Planification (n_planning_steps mises à jour avec le modèle)
            for _ in range(n_planning_steps):
                if model:
                    s, a = random.choice(list(model.keys()))
                    r, s_prime = model[(s, a)]
                    bonus = k * np.sqrt(timestep - time_since_last_visit[s, a])
                    q_table[s, a] += alpha * (r + bonus + gamma * np.max(q_table[s_prime]) - q_table[s, a])

    for state in range(len(S)):
        best_action = np.argmax(q_table[state])
        policy[state] = np.zeros(len(A))
        policy[state][best_action] = 1.0

    return policy, q_table
