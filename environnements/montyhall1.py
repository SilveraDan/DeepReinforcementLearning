import numpy as np
from collections import namedtuple

class MontyHall1:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.scored = 0
        self.doors = [0, 0, 1]
        np.random.shuffle(self.doors)

        self.state = np.random.choice(self.states)
        self.nb_steps = 0
        self.action_choose = None

        # Ajout des attributs action_space et state_space
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.state_space = namedtuple('StateSpace', ['n'])
        self.action_space.n = len(self.actions)
        self.state_space.n = len(self.states)


    def create_montyhall1(self):
        num_states = len(self.states)
        p = np.zeros((num_states, len(self.actions), num_states, len(self.rewards)))

        for s in range(num_states):
            for a in range(len(self.actions)):
                selected_door = self.states[s]
                revealed_door = self.reveal_door()
                if a == 1:  # Switch
                    next_state = [door for door in self.states if door != selected_door and door != revealed_door][0]
                else:  # Stay
                    next_state = selected_door

                reward = self.rewards[1] if self.doors[self.state] == 1 else self.rewards[0]
                next_state_idx = self.states.index(next_state)
                reward_idx = self.rewards.index(reward)
                p[s, a, next_state_idx, reward_idx] = 1.0

        return p

    def reveal_door(self):
        original_value = self.doors[self.state]
        available_doors = [i for i in range(len(self.doors)) if i != self.state and self.doors[i] == 0]
        door_to_reveal = np.random.choice(available_doors)
        self.doors.pop(door_to_reveal)
        self.state = self.doors.index(original_value)

        return door_to_reveal

    def reset(self):
        self.doors = [0, 0, 1]
        np.random.shuffle(self.doors)
        self.state = np.random.choice(self.states)
        self.nb_steps = 0
        self.action_choose = None
        self.scored = 0

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
        return self.nb_steps in self.terminals

    def step(self, action: int):
        self.reveal_door()
        if action == 1:  # Switch
            available_doors = [i for i in range(len(self.doors)) if i != self.state]
            next_state = np.random.choice(available_doors)
        else:  # Stay
            next_state = self.state

        self.nb_steps += 1
        self.action_choose = action
        self.state = next_state

    def score(self):
        if self.nb_steps in self.terminals:
            if self.doors[self.state] == 1:
                self.scored = self.rewards[1]
            else:
                self.scored = self.rewards[0]
        return 0

    def available_actions(self):
        return self.actions
