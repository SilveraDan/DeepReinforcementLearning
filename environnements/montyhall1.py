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
        self.available_doors = [0, 1, 2]
        np.random.shuffle(self.doors)

        self.state = np.random.choice(self.states)
        self.nb_steps = 0
        self.action_choose = None

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
        available_doors_to_choose = [i for i in self.available_doors if i != self.state and self.doors[i] == 0]
        door_to_reveal = np.random.choice(available_doors_to_choose)
        self.available_doors.remove(door_to_reveal)

    def reset(self):
        self.doors = [0, 0, 1]
        self.available_doors = [0, 1, 2]
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
        return self.state


    def is_forbidden(self, action: int) -> int:
        return not action in self.actions

    def is_game_over(self) -> bool:
        return self.nb_steps in self.terminals

    def step(self, action: int):
        self.reveal_door()
        if action == 1:  # Switch
            available_doors_to_choose = [door for door in self.available_doors if door != self.state]
            self.state = np.random.choice(available_doors_to_choose)

        self.nb_steps += 1
        self.action_choose = action
        self.scored = self.score()

    def score(self):
        if self.nb_steps in self.terminals:
            if self.doors[self.state] == 1:
                return self.rewards[1]
            else:
                return self.rewards[0]
        else:
            return 0

    def available_actions(self):
        return self.actions

    def display(self):
        if self.nb_steps == 0:
            doors_display = ['X' if i != self.state else 'O' for i in range(len(self.doors))]
            print(f"Doors: {doors_display}")
        else:
            action_str = "Switch" if self.action_choose == 1 else "Stay"
            result_str = "Won" if self.doors[self.state] == 1 else "Lost"
            print(f"Action: {action_str}")
            print(f"Result: {result_str}")
