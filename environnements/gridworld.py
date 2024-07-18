import numpy as np


def create_gridworld(S, A, R):
    grid_size = int(np.sqrt(len(S)))
    p = np.zeros((len(S), len(A), len(S), len(R)))

    # Définir les mouvements possibles pour chaque action
    movements = {
        0: (-1, 0),  # left
        1: (1, 0),   # right
        2: (0, 1),   # down
        3: (0, -1)   # up
    }

    for s in range(len(S)):
        x, y = s // grid_size, s % grid_size

        if 0 < x < grid_size - 1 and 0 < y < grid_size - 1:
            for action, (dx, dy) in movements.items():
                next_x, next_y = x + dx, y + dy
                if 0 <= next_x < grid_size and 0 <= next_y < grid_size:
                    next_state = next_x * grid_size + next_y
                    if (next_x == 0 or next_x == grid_size - 1 or
                        next_y == 0 or next_y == grid_size - 1):
                        reward = -1
                    else:
                        reward = 1 if (next_x, next_y) == (3, 3) else 0
                    p[s, action, next_state, R.index(reward)] = 1.0

    return p


def display_state(state):
    gridworld = [['_' for _ in range(5)] for _ in range(5)]
    x, y = divmod(state, 5)
    gridworld[x][y] = 'X'
    for row in gridworld:
        print(" ".join(row))


def play_game(policy, P, R, T, start_state=12):
    state = start_state
    total_reward = 0
    steps = [state]
    display_state(state)  # Display initial state

    while state not in T:  # terminal states
        print('*' * 30)
        action = np.argmax(policy[state])
        next_state = np.argmax(np.sum(P[state, action, :, :], axis=1))
        reward_index = np.argmax(P[state, action, next_state, :])
        reward = R[reward_index]
        total_reward += reward
        steps.append(next_state)
        state = next_state
        display_state(state)  # Display state after action
        print('*' * 30)

    return steps, total_reward
