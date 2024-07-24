import numpy as np
import random
from collections import namedtuple

class TwoRoundRPS:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.transition_matrix = self.create_rps()
        self.scored = 0
        self.state = 'initial'
        self.rounds_played = 0
        
        # Add action_space and state_space attributes
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.state_space = namedtuple('StateSpace', ['n'])
        self.action_space.n = len(self.actions)
        self.state_space.n = len(self.states)

    def create_rps(self):
        p = np.zeros((len(self.states), len(self.actions), len(self.states), len(self.rewards)))

        # Defining transitions for the first round
        p[self.states.index('initial'), self.actions.index(0), self.states.index('first_round_rock'), self.rewards.index(0)] = 1.0
        p[self.states.index('initial'), self.actions.index(1), self.states.index('first_round_paper'), self.rewards.index(0)] = 1.0
        p[self.states.index('initial'), self.actions.index(2), self.states.index('first_round_scissors'), self.rewards.index(0)] = 1.0

        # Defining transitions for the second round which directly transitions to terminal state with win/lose/draw rewards
        for first_round_state in ['first_round_rock', 'first_round_paper', 'first_round_scissors']:
            p[self.states.index(first_round_state), self.actions.index(0), self.states.index('terminal'), self.rewards.index(1)] = 1.0
            p[self.states.index(first_round_state), self.actions.index(1), self.states.index('terminal'), self.rewards.index(1)] = 1.0
            p[self.states.index(first_round_state), self.actions.index(2), self.states.index('terminal'), self.rewards.index(1)] = 1.0

        return p

    def reset(self):
        self.state = 'initial'
        self.rounds_played = 0
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

    def state_id(self) -> int:
        return self.states.index(self.state)

    def display(self):
        print(f"Current State: {self.state}")

    def is_forbidden(self, action: int) -> int:
        return not action in self.actions

    def is_game_over(self) -> bool:
        return self.state in self.terminals

    def step(self, action: int):
        if self.is_game_over():
            return self.state, 0, True, {}

        if self.state == 'initial':
            if action == 0:
                self.state = 'first_round_rock'
            elif action == 1:
                self.state = 'first_round_paper'
            elif action == 2:
                self.state = 'first_round_scissors'
            reward = 0
            done = False
        else:
            opponent_action = random.choice(self.actions)
            if action == opponent_action:
                reward = self.rewards[1]  # Draw
            elif (action == self.actions[0] and opponent_action == self.actions[2]) or \
                 (action == self.actions[1] and opponent_action == self.actions[0]) or \
                 (action == self.actions[2] and opponent_action == self.actions[1]):
                reward = self.rewards[2]  # Win
            else:
                reward = self.rewards[0]  # Lose
            self.state = 'terminal'
            done = True
        return self.state, reward, done, {}

    def score(self):
        if self.state == 'terminal':
            return self.scored
        return 0

    def available_actions(self):
        return self.actions
