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


def value_interation(env, theta, gamma, date_dict):
    delta = 0
    V = np.random.rand(env.num_states())

    for i in range(1000):
        delta = 0
        for s in range(env.num_states()):
            v = V[s]
            filtered_dico = {k: v for k, v in data_dict.items() if v[0] == s}
            if not filtered_dico: continue
            sums = []
            for key, value in filtered_dico.items():
                s, a, s_prime, reward, p = value
                sum_expected_value = p * (reward + gamma * V[s_prime])
                sums.append(sum_expected_value)
            best_expected = max(sums)
            V[s] = best_expected
            if delta > np.abs(v - V[s]):
                delta = np.abs(v - V[s])
        if delta < theta:
            break
    Pi = compute_policy(V,date_dict,gamma,V)
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


if __name__ == '__main4__':
    config_lineworld = load_config(congig_file, "LineWorld")
    lineworld_env = lw.LineWorld(config_lineworld)
    Pi = value_interation(lineworld_env, 1e-3, 0.95)
    lineworld_test = lw.LineWorld(config_lineworld)
    play_a_game_by_Pi(lineworld_test, Pi)

if __name__ == '__main__':
    with open('mon_dictionnaire.pkl', 'rb') as fichier:
        data_dict = pickle.load(fichier)
    Pi = value_interation(secret_envs_wrapper.SecretEnv0(), 0.001, 0.95, data_dict)
    env = secret_envs_wrapper.SecretEnv0()
    play_a_game_by_Pi(env, Pi)

if __name__ == '__main4__':
    mon_dictionnaire = (extract_data(secret_envs_wrapper.SecretEnv0()))

    # Sauvegarde du dictionnaire dans un fichier
    with open('mon_dictionnaire.pkl', 'wb') as fichier:
        pickle.dump(mon_dictionnaire, fichier)
    # Chargement du dictionnaire depuis le fichier
    with open('mon_dictionnaire.pkl', 'rb') as fichier:
        mon_dictionnaire_chargé = pickle.load(fichier)

    print(mon_dictionnaire_chargé)

if __name__ == '__main4__':
    with open('mon_dictionnaire.pkl', 'rb') as fichier:
        data_dict = pickle.load(fichier)
    actions = np.zeros(len(data_dict))
    for key, value in data_dict.items():
        s, a, s_prime, reward, p = value
        num1 = a
        if isinstance(num1, int):
            print(f"{num1} est un int")
        elif isinstance(num1, float):
            print(f"{num1} est un float")
        actions = np.append(actions, int(a))
    for i in range(len(actions)):
        best_action = actions[i]
        print(best_action)
        num1 = best_action
        if isinstance(num1, int):
            print(f"{num1} est un int")
        elif isinstance(num1, float):
            print(f"{num1} est un float")