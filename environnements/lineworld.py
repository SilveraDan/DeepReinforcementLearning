import numpy as np

def create_lineworld(S, A, R):
    p = np.zeros((len(S), len(A), len(S), len(R)))

    # Simplify the nested loops by directly calculating transitions and rewards
    for s in range(1, len(S) - 1):
        if s in [1, 2]:
            p[s, 1, s + 1, R.index(0)] = 1.0  # Move right with no reward
        if s in [2, 3]:
            p[s, 0, s - 1, R.index(0)] = 1.0  # Move left with no reward

    # Specific transitions with rewards
    p[3, 1, 4, R.index(1)] = 1.0  # Move right from state 3 to terminal state 4 with reward
    p[1, 0, 0, R.index(-1)] = 1.0  # Move left from state 1 to terminal state 0 with penalty
    return p


def display_state(state, length=5):
    lineworld = ['_' for _ in range(length)]
    lineworld[state] = 'X'
    print("".join(lineworld))


def play_game(policy, P, R, start_state=2):
    state = start_state
    total_reward = 0
    steps = [state]
    display_state(state)  # Display initial state
    while state not in [0, 4]:  # terminal states
        action = np.argmax(policy[state])
        next_state = np.argmax(np.sum(P[state, action, :, :], axis=1))  # Correct the next state calculation
        reward_index = np.argmax(P[state, action, next_state, :])
        reward = R[reward_index]
        total_reward += reward
        steps.append(next_state)
        state = next_state
        display_state(state)  # Display state after action
    return steps, total_reward
