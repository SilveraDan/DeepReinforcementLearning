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
        q_s = [Q[s, a] for a in available_actions]
        best_a_index = np.argmax(q_s)
        a = available_actions[best_a_index]
    return a


def init_Q(env, Q):
    for s in range(env.num_states()):
        for a in range(env.num_actions()):
            Q[s, a] = np.random.random()
    return Q


def dyna_q(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 100000, n_planning = 100):
    Q = np.zeros((env.num_states(), env.num_actions()))
    Q = init_Q(env, Q)
    model = {}
    s_observed = []
    a_taken = []
    for _ in range(nb_iter):
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            s_observed.append(s)
            available_actions = env.available_actions()
            a = choose_action(Q, s, available_actions, epsilon)
            a_taken.append(a)
            prev_score = env.score()
            env.step(a)
            new_score = env.score()
            reward = new_score - prev_score
            s_prime = env.state_id()
            available_actions_prime = env.available_actions()
            q_s_prime = [Q[s_prime, a_p] for a_p in available_actions_prime]
            index_best_reward_after_move = np.argmax(q_s_prime)
            action = available_actions_prime[index_best_reward_after_move]
            best_move = Q[s_prime, action]
            Q[s, a] = Q[s, a] + alpha * (reward + gamma * best_move - Q[s, a])
            model[(s, action)] = (reward, s_prime)
            for _ in range(n_planning):
                s, a = random.choice(list(model.keys()))
                reward, s_prime = model[(s, a)]
                available_actions_prime = env.available_actions()
                q_s_prime = [Q[s_prime][a_p] for a_p in available_actions_prime]
                index_best_reward_after_move = np.argmax(q_s_prime)
                action = available_actions_prime[index_best_reward_after_move]
                best_move = Q[s_prime][action]
                Q[s, a] = Q[s, a] + alpha * (reward + gamma * best_move - Q[s, a])
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
    Q_optimal = dyna_q(lineworld_env, 0.1, 0.1, 0.9,500)
    Pi = calcul_policy(Q_optimal)
    lineworld_test = lw.LineWorld(config_lineworld)
    play_a_game_by_Pi(lineworld_test, Pi)

if __name__ == '__main__kdf':
    Q_optimal = dyna_q(secret_envs_wrapper.SecretEnv0(), 0.1, 0.1, 0.9,5000)
    Pi = calcul_policy(Q_optimal)
    env = secret_envs_wrapper.SecretEnv0()
    play_a_game_by_Pi(env, Pi)
