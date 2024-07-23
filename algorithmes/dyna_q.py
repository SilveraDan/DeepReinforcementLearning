import numpy as np
from tqdm import tqdm
import random
import secret_envs_wrapper
import environnements.lineworld as lw
import environnements.gridworld2 as gw
from utils import load_config, calcul_policy, play_a_game_by_Pi, observe_R_S_prime, save_results_to_pickle


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


def calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha):
    q_s_prime = [Q[s_prime, a_p] for a_p in available_actions_prime]
    best_move = np.max(q_s_prime)
    Q[s, a] = Q[s, a] + alpha * (reward + gamma * best_move - Q[s, a])
    return Q[s, a]


def dyna_q(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 100000, n_planning=10):
    Q = np.zeros((env.num_states(), env.num_actions()))
    Q = init_Q(env, Q)
    model = {}
    for _ in tqdm(range(nb_iter)):
        env.reset()
        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions()
            # Choose A from S using policy derived from Q
            a = choose_action(Q, s, available_actions, epsilon)
            # Take action A, observe R, S'
            reward, s_prime, available_actions_prime = observe_R_S_prime(env, a)
            Q[s, a] = calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha)
            model[(s, a)] = (reward, s_prime)
            for _ in range(n_planning):
                s, a = random.choice(list(model.keys()))
                reward, s_prime = model[(s, a)]
                available_actions_prime = env.available_actions()
                Q[s, a] = calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha)
    return Q


def play_game(game, parameters, results_path):
    if "SecretEnv" not in game:
        config = load_config(congig_file, game)
    alpha = parameters["alpha"]
    epsilon = parameters["epsilon"]
    gamma = parameters["gamma"]
    nb_iter = parameters["nb_iter"]
    n_planning = parameters["n_planning"]
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
    Q_optimal = dyna_q(env, alpha, epsilon, gamma, nb_iter, n_planning)
    Pi = calcul_policy(Q_optimal)
    env.reset()
    save_results_to_pickle(Q_optimal, Pi, results_path)
    #play_a_game_by_Pi(env, Pi)


if __name__ == '__main__':
    game = "SecretEnv0"
    parameters = {
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 0.999,
        "nb_iter": 1000,
        "n_planning": 10
    }
    results_path = f"../results/{game}_dyna_q.pkl"
    play_game(game, parameters, results_path)
