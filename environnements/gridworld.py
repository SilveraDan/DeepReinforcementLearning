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
        self.scored = 0
        self.state = 6
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
        self.state = 6  # Start state
        return self.state

    def num_states(self) -> int:
        return len(self.states)

    def num_actions(self) -> int:
        return len(self.actions)

    def num_rewards(self) -> int:
        return len(self.rewards)

    def reward(self, i: int) -> float:
        return self.rewards[i]

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.transition_matrix[s, a, s_p, r_index]

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return self.state

    def display(self):
        grid = ['_' for _ in range(len(self.states))]  # Étape 1
        for i in range(len(self.states)):  # Étape 2
            if i == self.state:  # Si la case est celle du joueur
                grid[i] = 'P'
            elif i in self.terminals:  # Si la case est une case terminale (but)
                if i == 18:
                    grid[i] = '0'
                else:
                    grid[i] = 'X'
        for i in range(self.grid_size):  # Étape 3
            print(" ".join(grid[i * self.grid_size:(i + 1) * self.grid_size]))
        print("\n")

    def is_forbidden(self, action: int) -> int:
        return not action in self.actions

    def is_game_over(self) -> bool:
        is_end = self.state in self.terminals
        return is_end

    def step(self, action: int):
        if action == 0:
            self.state -= 1
        elif action == 1:
            self.state += 1
        elif action == 2:
            self.state += 5
        elif action == 3:
            self.state -= 5

    def score(self):
        if self.state in self.terminals:
            if self.state == 18:
                self.scored = self.rewards[2]
            else:
                self.scored = self.rewards[0]
        return self.scored

    def available_actions(self):
        return self.actions