from collections import namedtuple
import numpy as np

class LineWorld:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.transition_matrix = self.create_lineworld()
        self.scored = 0
        self.state = 2
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
        lineworld = ['_' for _ in range(5)]
        lineworld[self.state] = 'X'
        print("".join(lineworld))

    def is_forbidden(self, action: int) -> int:
        return not action in self.actions

    def is_game_over(self) -> bool:
        is_end = self.state in self.terminals
        return is_end

    def step(self, action: int):
        if action == 1:
            self.state += 1
        if action == 0:
            self.state -= 1
    def score(self):
        if self.state == 4:
            self.scored = self.rewards[2]
        if self.state == 0:
            self.scored = self.rewards[0]
        return self.scored

    def available_actions(self):
        return self.actions