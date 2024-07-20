from collections import namedtuple
import numpy as np

class GridWorld:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.grid_size = int(np.sqrt(len(self.states)))
        self.transition_matrix = self.create_gridworld()
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.action_space.n = len(self.actions)
        self.state = None

    def create_gridworld(self):
        p = np.zeros((len(self.states), len(self.actions), len(self.states), len(self.rewards)))

        for s in range(len(self.states)):
            if s in self.terminals:
                continue

            row, col = divmod(s, self.grid_size)

            for a in range(len(self.actions)):
                if a == 0 and col > 0:  # Left
                    next_state = s - 1
                elif a == 1 and col < self.grid_size - 1:  # Right
                    next_state = s + 1
                elif a == 2 and row < self.grid_size - 1:  # Down
                    next_state = s + self.grid_size
                elif a == 3 and row > 0:  # Up
                    next_state = s - self.grid_size
                else:
                    next_state = s

                if next_state in self.terminals:
                    reward = self.rewards[2]
                elif next_state in [0, 4, 20, 24]:
                    reward = self.rewards[0]
                else:
                    reward = self.rewards[1]

                reward_index = self.rewards.index(reward)
                p[s, a, next_state, reward_index] = 1.0

        return p

    def reset(self):
        self.state = 12  # Start state in the middle
        return self.state

    def step(self, action):
        next_state = np.argmax(np.sum(self.transition_matrix[self.state, action, :, :], axis=1))
        reward_index = np.argmax(self.transition_matrix[self.state, action, next_state, :])
        reward = self.rewards[reward_index]
        done = next_state in self.terminals
        self.state = next_state
        return self.state, reward, done

    def render(self):
        grid = ['_' for _ in range(len(self.states))]
        grid[self.state] = 'X'
        for i in range(self.grid_size):
            print(" ".join(grid[i * self.grid_size:(i + 1) * self.grid_size]))
        print("\n")
