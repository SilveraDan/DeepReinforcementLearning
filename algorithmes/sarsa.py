import numpy as np
import random
from environnements.lineworld import play_game as play_game_lineworld
from environnements.gridworld import play_game as play_game_gridworld


def choose_action(state, epsilon, Q, A):
    if random.uniform(0, 1) < epsilon:
        return random.choice(A)
    else:
        return np.argmax(Q[state])


def sarsa(game, S, A, R, P, T, num_episodes, alpha, gamma, epsilon, start_state):
    # Initialisation de la fonction de valeur d'action Q
    Q = np.zeros((len(S), len(A)))
    policy = np.ones((len(S), len(A))) / len(A)

    # Fonction pour choisir une action en utilisant une politique epsilon-greedy
    for episode in range(num_episodes):
        state = start_state
        action = choose_action(state, epsilon, Q, A)

        while state not in T:  # États terminaux
            next_state = np.argmax(np.sum(P[state, action, :, :], axis=1))
            reward_index = np.argmax(P[state, action, next_state, :])
            reward = R[reward_index]
            next_action = choose_action(next_state, epsilon, Q, A)

            # Mise à jour de Q selon la formule Sarsa
            Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action

    # Mise à jour de la politique pour être gloutonne par rapport à Q
    for state in range(len(S)):
        best_action = np.argmax(Q[state])
        policy[state] = np.zeros(len(A))
        policy[state][best_action] = 1.0
    #
    # match game:
    #     case "lineworld":
    #         steps, total_reward = play_game_lineworld(policy, P, R, T)
    #     case "gridworld":
    #         steps, total_reward = play_game_gridworld(policy, P, R, T)

    # print(f"Steps: {steps}")
    # print(f"Total Reward: {total_reward}")

    return policy, Q
