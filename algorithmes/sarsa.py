import numpy as np
from tqdm import tqdm
import os
import yaml
import random
import secret_envs_wrapper
import pickle
import environnements.lineworld2 as lw
from utils import load_config, calcul_policy, play_a_game_by_Pi

congig_file = "../config.yaml"


def choose_action(Q, s, available_actions, epsilon):
    if random.uniform(0, 1) < epsilon:
        a = random.choice(available_actions)
    else:
        q_s = [Q[s][a] for a in available_actions]
        best_a_index = np.argmax(q_s)
        a = available_actions[best_a_index]
    return a


def sarsa(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 100000, Q={}):
    for _ in tqdm(range(nb_iter)):
        # Initialize S
        env.reset()
        s = env.state_id()
        available_actions = env.available_actions()
        Q = update_Q(Q, s, available_actions)
        # Choose A from S using policy derived from Q
        a = choose_action(Q, s, available_actions, epsilon)
        while not env.is_game_over():
            #observe R and S'
            prev_score = env.score()
            env.step(a)
            new_score = env.score()
            reward = new_score - prev_score
            s_prime = env.state_id()
            # Choose A' from S' using policy derived from Q
            available_actions_prime = env.available_actions()
            Q = update_Q(Q, s_prime, available_actions_prime)
            a_prime = choose_action(Q, s_prime, available_actions_prime, epsilon)
            if not env.is_game_over():
                q_s_prime = Q[s_prime][a_prime]
                Q[s][a] = Q[s][a] + alpha * (reward + gamma * q_s_prime - Q[s][a])
            else:
                Q[s][a] = Q[s][a] + alpha * (reward)
            s = s_prime
            a = a_prime
    return Q

def update_Q(Q,s,aa):
    if s not in Q:
        Q[s] = {}
        for a in aa:
            Q[s][a] = np.random.random()
    return Q




if __name__ == '__main_random_game__':
    env0 = secret_envs_wrapper.SecretEnv3()
    while not env0.is_game_over():
        env0.display()
        a = random.choice(env0.available_actions())
        print(env0.state_id())
        env0.step(a)
        print(env0.score())

if __name__ == '__main__':
    config_lineworld = load_config(congig_file, "LineWorld")
    game = "lineworld"
    S = config_lineworld["states"]
    A = config_lineworld["actions"]
    R = config_lineworld["rewards"]
    T = config_lineworld["terminals"]
    lineworld_env = lw.LineWorld(config_lineworld)
    Q_optimal = sarsa(lineworld_env, 0.1, 0.1, 0.9,50000)
    Pi = calcul_policy(Q_optimal)
    lineworld_test = lw.LineWorld(config_lineworld)
    play_a_game_by_Pi(lineworld_test, Pi)

if __name__ == '__main__env':
    Q_optimal = sarsa(secret_envs_wrapper.SecretEnv0(), 0.1, 0.1, 0.9,5000)
    Pi = calcul_policy(Q_optimal)
    env = secret_envs_wrapper.SecretEnv0()
    play_a_game_by_Pi(env,Pi)