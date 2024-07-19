from collections import namedtuple
import numpy as np

class LineWorld:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.transition_matrix = self.create_lineworld()
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.action_space.n = len(self.actions)

    def create_lineworld(self):
        p = np.zeros((len(self.states), len(self.actions), len(self.states), len(self.rewards)))

        # Simplify the nested loops by directly calculating transitions and rewards
        for s in range(1, len(self.states) - 1):
            if s in [1, 2]:
                p[s, 1, s + 1, self.rewards.index(0)] = 1.0  # Move right with no reward
            if s in [2, 3]:
                p[s, 0, s - 1, self.rewards.index(0)] = 1.0  # Move left with no reward

        # Specific transitions with rewards
        p[3, 1, 4, self.rewards.index(1)] = 1.0  # Move right from state 3 to terminal state 4 with reward
        p[1, 0, 0, self.rewards.index(-1)] = 1.0  # Move left from state 1 to terminal state 0 with penalty
        return p

    def reset(self):
        self.state = 2  # Start state
        return self.state

    def step(self, action):
        next_state = np.argmax(np.sum(self.transition_matrix[self.state, action, :, :], axis=1))
        reward_index = np.argmax(self.transition_matrix[self.state, action, next_state, :])
        reward = self.rewards[reward_index]
        done = next_state in self.terminals
        self.state = next_state
        return self.state, reward, done

    def render(self):
        lineworld = ['_' for _ in range(5)]
        lineworld[self.state] = 'X'
        print("".join(lineworld))
