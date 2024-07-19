import numpy as np
import random
from collections import namedtuple, defaultdict
import yaml

# Fonction pour charger la configuration
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Classe TwoRoundRPS
class TwoRoundRPS:
    def __init__(self, config):
        self.states = config['states']
        self.actions = config['actions']
        self.rewards = config['rewards']
        self.terminals = config['terminals']
        self.reset()
        self.action_space = namedtuple('ActionSpace', ['n'])
        self.action_space.n = len(self.actions)
    
    def reset(self):
        self.state = 'initial'
        self.opponent_moves = [random.choice(self.actions), None]
        return self.state
    
    def step(self, action):
        move = self.actions[action]
        if self.state == 'initial':
            if move == 0:
                self.state = 'first_round_rock'
            elif move == 1:
                self.state = 'first_round_paper'
            elif move == 2:
                self.state = 'first_round_scissors'
            reward = 0
            done = False
        else:
            opponent_move = self.opponent_moves[0]
            if move == opponent_move:
                reward = self.rewards[1]  # Draw
            elif (move == self.actions[0] and opponent_move == self.actions[2]) or \
                 (move == self.actions[1] and opponent_move == self.actions[0]) or \
                 (move == self.actions[2] and opponent_move == self.actions[1]):
                reward = self.rewards[2]  # Win
            else:
                reward = self.rewards[0]  # Lose
            self.state = 'terminal'
            done = True
        return self.state, reward, done

    def render(self):
        print(f"State: {self.state}, Opponent Moves: {self.opponent_moves}")

# Charger la configuration
config = load_config('config.yaml')
tworoundrps_config = config['TwoRoundRPS']

# Tester et visualiser TwoRoundRPS
env = TwoRoundRPS(tworoundrps_config)
state = env.reset()
print(f"Initial State: {state}")
done = False
while not done:
    action = random.choice(env.actions)
    next_state, reward, done = env.step(action)
    env.render()
    print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
    state = next_state
