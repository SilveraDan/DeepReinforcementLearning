from collections import namedtuple
import numpy as np

class GridWorld:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.transition_matrix = self.create_gridworld()
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.action_space.n = len(self.actions)

    def create_gridworld(self):
        grid_size = int(np.sqrt(len(self.states)))
        p = np.zeros((len(self.states), len(self.actions), len(self.states), len(self.rewards)))

        # Définir les mouvements possibles pour chaque action
        movements = {
            0: (-1, 0),  # left
            1: (1, 0),   # right
            2: (0, 1),   # down
            3: (0, -1)   # up
        }

        for s in range(len(self.states)):
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
                        p[s, action, next_state, self.rewards.index(reward)] = 1.0

        return p

    def reset(self):
        self.state = 12  # Start state
        return self.state

    def step(self, action):
        next_state = np.argmax(np.sum(self.transition_matrix[self.state, action, :, :], axis=1))
        reward_index = np.argmax(self.transition_matrix[self.state, action, next_state, :])
        reward = self.rewards[reward_index]
        done = next_state in self.terminals
        self.state = next_state
        return self.state, reward, done

    def render(self):
        gridworld = [['_' for _ in range(5)] for _ in range(5)]
        x, y = divmod(self.state, 5)
        gridworld[x][y] = 'X'
        for row in gridworld:
            print(" ".join(row))
