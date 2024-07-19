import numpy as np
import random
from collections import namedtuple
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

class MontyHall2:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.reset()
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.action_space.n = len(self.actions)
    
    def reset(self):
        self.doors = ['goat'] * 4 + ['car']
        random.shuffle(self.doors)
        self.selected_doors = [random.choice(self.states)]
        self.revealed_doors = self.reveal_doors()
        self.state = (tuple(self.selected_doors), tuple(self.revealed_doors))
        return self.state
    
    def reveal_doors(self):
        available_doors = [door for door in self.states if door not in self.selected_doors and self.doors[self.states.index(door)] == 'goat']
        return random.sample(available_doors, 3)
    
    def step(self, action):
        if action == 1:  # Switch
            remaining_doors = [door for door in self.states if door not in self.selected_doors and door not in self.revealed_doors]
            self.selected_doors.append(random.choice(remaining_doors))
        reward = self.rewards[1] if self.doors[self.states.index(self.selected_doors[-1])] == 'car' else self.rewards[0]
        done = len(self.selected_doors) == 2
        if done:
            self.state = 'terminal'
        else:
            self.revealed_doors = self.reveal_doors()
            self.state = (tuple(self.selected_doors), tuple(self.revealed_doors))
        return self.state, reward, done

    def render(self):
        print(f"Doors: {self.doors}, Selected Doors: {self.selected_doors}, Revealed Doors: {self.revealed_doors}")

# Charger la configuration
config = load_config()
montyhall2_config = config['MontyHall2']

# Tester et visualiser MontyHall2
if __name__ == "__main__":
    env = MontyHall2(montyhall2_config)
    state = env.reset()
    print(f"Initial State: {state}")
    done = False
    while not done:
        action = random.choice(env.actions)
        next_state, reward, done = env.step(action)
        env.render()
