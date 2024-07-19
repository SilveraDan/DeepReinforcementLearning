import numpy as np
import random
from collections import namedtuple
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

class MontyHall1:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.reset()
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.action_space.n = len(self.actions)
    
    def reset(self):
        self.doors = ['goat', 'goat', 'car']
        random.shuffle(self.doors)
        self.selected_door = random.choice(self.states)
        self.revealed_door = self.reveal_door()
        self.state = (self.selected_door, self.revealed_door)
        return self.state
    
    def reveal_door(self):
        available_doors = [door for door in self.states if door != self.selected_door and self.doors[self.states.index(door)] == 'goat']
        return random.choice(available_doors)
    
    def step(self, action):
        if action == 1:  # Switch
            self.selected_door = [door for door in self.states if door != self.selected_door and door != self.revealed_door][0]
        reward = self.rewards[1] if self.doors[self.states.index(self.selected_door)] == 'car' else self.rewards[0]
        done = True
        self.state = 'terminal'
        return self.state, reward, done

    def render(self):
        print(f"Doors: {self.doors}, Selected Door: {self.selected_door}, Revealed Door: {self.revealed_door}")

# Charger la configuration
config = load_config()
montyhall1_config = config['MontyHall1']

# Tester et visualiser MontyHall1
if __name__ == "__main__":
    env = MontyHall1(montyhall1_config)
    state = env.reset()
    print(f"Initial State: {state}")
    done = False
    while not done:
        action = random.choice(env.actions)
        next_state, reward, done = env.step(action)
        env.render()
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        state = next_state
