import numpy as np
from tqdm import tqdm
import secret_envs_wrapper
import environnements.lineworld as lw
import environnements.gridworld2 as gw
from utils import load_config, calcul_policy, play_a_game_by_Pi, choose_action, update_Q, save_results_to_pickle


congig_file = "../config.yaml"


def calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha, env):
    if not env.is_game_over():
        q_s_prime = [Q[s_prime][a_p] for a_p in available_actions_prime]
        best_move = np.max(q_s_prime)  # Directement obtenir la meilleure récompense après le mouvement
        Q[s][a] += alpha * (reward + gamma * best_move - Q[s][a])
    else:
        Q[s][a] += alpha * reward
    return Q[s][a]


def observe_R_S_prime(env, a):
    prev_score = env.score()
    env.step(a)
    new_score = env.score()
    reward = new_score - prev_score
    s_prime = env.state_id()
    available_actions_prime = env.available_actions()
    return reward, s_prime, available_actions_prime


def q_learning(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 100000):
    Q = {}
    # Loop for each episode
    for _ in tqdm(range(nb_iter)):
        #Initialize S
        env.reset()
        # Loop for each step epiosde
        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions()
            Q = update_Q(Q, s, available_actions, env)
            # Choose A from S using policy derived from Q
            a = choose_action(Q, s, available_actions, epsilon)
            # Take action A, observe R, S'
            reward, s_prime, available_actions_prime = observe_R_S_prime(env, a)
            Q = update_Q(Q, s_prime, available_actions_prime, env)
            if not env.is_game_over():
                # Calcul Q(s,a)
                Q[s][a] = calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha, env)
            else:
                Q[s][a] = Q[s][a] + alpha * reward
    return Q


def play_game(game, parameters, results_path):
    if "SecretEnv" not in game:
        config = load_config(congig_file, game)
    alpha = parameters["alpha"]
    epsilon = parameters["epsilon"]
    gamma = parameters["gamma"]
    nb_iter = parameters["nb_iter"]
    match game:
        case "LineWorld":
            env = lw.LineWorld(config)
        case "GridWorld":
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
    Q_optimal = q_learning(env, alpha, epsilon, gamma, nb_iter)
    Pi = calcul_policy(Q_optimal)
    env.reset()
    save_results_to_pickle(Q_optimal, Pi, results_path)
    play_a_game_by_Pi(env, Pi)


if __name__ == '__main__':
    game = "SecretEnv0"
    parameters = {"alpha": 0.1, "epsilon": 0.1, "gamma": 0.999, "nb_iter": 1000}
    results_path = f"results/{game}_q_learning.pkl"
    play_game(game, parameters, results_path)



