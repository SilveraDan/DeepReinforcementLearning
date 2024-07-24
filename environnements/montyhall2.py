import numpy as np
from collections import namedtuple

class MontyHall2:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.transition_matrix = self.create_montyhall2()
        self.scored = 0
        self.state = None

        # Ajout des attributs action_space et state_space
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.state_space = namedtuple('StateSpace', ['n'])
        self.action_space.n = len(self.actions)
        self.state_space.n = len(self.states)
        self.reset()
    
    def create_montyhall2(self):
        num_states = len(self.states)
        p = np.zeros((num_states, len(self.actions), num_states, len(self.rewards)))

        for s in range(num_states):
            for a in range(len(self.actions)):
                self.doors = ['goat'] * 4 + ['car']
                np.random.shuffle(self.doors)
                selected_door = self.states[s]
                revealed_doors = self.reveal_doors(selected_door)
                if a == 1:  # Switch
                    remaining_doors = [door for door in self.states if door != selected_door and door not in revealed_doors]
                    next_state = np.random.choice(remaining_doors)
                else:  # Stay
                    next_state = selected_door

                reward = self.rewards[1] if self.doors[self.states.index(next_state)] == 'car' else self.rewards[0]
                next_state_idx = self.states.index(next_state)
                reward_idx = self.rewards.index(reward)
                p[s, a, next_state_idx, reward_idx] = 1.0

        return p
    
    def reveal_doors(self, selected_door):
        available_doors = [door for door in self.states if door != selected_door and self.doors[self.states.index(door)] == 'goat']
        return np.random.choice(available_doors, 3, replace=False)
    
    def reset(self):
        self.doors = ['goat'] * 4 + ['car']
        np.random.shuffle(self.doors)
        self.state = np.random.choice(self.states)
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
        print(f"Current State: {self.state}, Doors: {self.doors}")

    def is_forbidden(self, action: int) -> int:
        return not action in self.actions

    def is_game_over(self) -> bool:
        return self.state in self.terminals

    def step(self, action: int):
        if self.is_game_over():
            return self.state, 0, True, {}

        revealed_doors = self.reveal_doors(self.state)
        if action == 1:  # Switch
            remaining_doors = [door for door in self.states if door != self.state and door not in revealed_doors]
            next_state = np.random.choice(remaining_doors)
        else:  # Stay
            next_state = self.state

        reward = self.rewards[1] if self.doors[self.states.index(next_state)] == 'car' else self.rewards[0]
        self.state = next_state
        done = self.state in self.terminals
        return self.state, reward, done, {}

    def score(self):
        if self.state in self.terminals:
            return self.scored
        return 0

    def available_actions(self):
        return self.actions
