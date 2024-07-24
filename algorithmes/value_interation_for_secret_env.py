import numpy as np
from tqdm import tqdm
import os
import yaml
import random
import secret_envs_wrapper
import pickle
import environnements.lineworld2 as lw
import environnements.gridworld2 as gw
from utils import load_config, calcul_policy, play_a_game_by_Pi, choose_action, update_Q, observe_R_S_prime, save_results_to_pickle

congig_file = "../config.yaml"


def value_interation(env, theta, gamma, data_dict):
    delta = 0
    V = np.random.rand(env.num_states())

    while True:
        delta = 0
        for s in range(env.num_states()):
            v = V[s]
            filtered_dico = {k: v for k, v in data_dict.items() if v[0] == s}
            if not filtered_dico: continue
            sums = []
            for value in filtered_dico.values():
                s, a, s_prime, reward, p = value
                sum_expected_value = p * (reward + gamma * V[s_prime])
                sums.append(sum_expected_value)
            best_expected = max(sums)
            V[s] = best_expected
            if delta > np.abs(v - V[s]):
                delta = np.abs(v - V[s])
        if delta < theta:
            break
    Pi = compute_policy(V,data_dict,gamma,V)
    return Pi

def compute_policy(s,data_dict,gamma,V):
    Pi = {}
    for s in range(len(V)):
        Pi[s] = find_arg_max(s,data_dict,gamma,V)
    return  Pi

def find_arg_max(s,data_dict,gamma,V):
    filtered_dico = {k: v for k, v in data_dict.items() if v[0] == s}
    if not filtered_dico: return 0
    sums = []
    actions = []
    for key, value in filtered_dico.items():
        s, a, s_prime, reward, p = value
        sum_expected_value = p * (reward + gamma * V[s_prime])
        sums.append(sum_expected_value)
        actions.append(a)
    best_arg_expected = np.argmax(sums)
    best_action = actions[best_arg_expected]
    return best_action

def extract_data(env):
    dict = {}
    i = 0
    for s in tqdm(range(env.num_states())):
        for a in range(env.num_actions()):
            for s_prime in range(s, env.num_states()):
                for r in range(env.num_rewards()):
                    p = env.p(s, a, s_prime, r)
                    if p > 0.0:
                        dict[i] = (s, a, s_prime, r, p)
                    i += 1
    return dict

def play_game(game, parameters, results_path):
    theta = parameters["theta"]
    gamma = parameters["gamma"]
    data_dict = {}

    match game:
        case "LineWorld":
            config = load_config(congig_file, game)
            env = lw.LineWorld(config)
        case "GridWorld":
            config = load_config(congig_file, game)
            env = gw.GridWorld(config)
        case "SecretEnv0":
            env = secret_envs_wrapper.SecretEnv0()
        case "SecretEnv1":
            env = secret_envs_wrapper.SecretEnv1()
        case "SecretEnv2":
            env = secret_envs_wrapper.SecretEnv2()
        case _:
            print("Game not found")
            return 0
    if not os.path.exists(game + '.pkl'):
        data_dict = (extract_data(env))
        print('Created data_dict')
        with open(game + '.pkl', 'wb') as fichier:
            pickle.dump(data_dict, fichier)
    else:
        print('Loading data_dict')
        with open(game + '.pkl', 'rb') as fichier:
            data_dict = pickle.load(fichier)
    Pi = value_interation(env, theta, gamma, data_dict)
    Q = {}
    env.reset()
    score = env.score()
    save_results_to_pickle(Q, Pi, score, results_path)
    play_a_game_by_Pi(env, Pi)

if __name__ == '__main__':
    game = "SecretEnv1"
    parameters = {"theta": 0.00001, "gamma": 0.999}
    results_path = f"results/{game}_value_iteration.pkl"
    play_game(game, parameters, results_path)




