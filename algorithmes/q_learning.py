import numpy as np
from tqdm import tqdm
import secret_envs_wrapper
import environnements.lineworld as lw
import environnements.gridworld2 as gw
import environnements.montyhall1 as mh1
import environnements.montyhall2 as mh2
import utils as ut


congig_file = "./config.yaml"


def calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha, env):
    q_s_prime = [Q[s_prime][a_p] for a_p in available_actions_prime]
    best_move = np.max(q_s_prime)  # Directement obtenir la meilleure récompense après le mouvement
    Q[s][a] += alpha * (reward + gamma * best_move - Q[s][a])
    return Q[s][a]


def q_learning(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 100000):
    Q = {}
    # Loop for each episode
    for _ in range(nb_iter):
        #Initialize S
        env.reset()
        # Loop for each step epiosde
        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions()
            Q = ut.update_Q(Q, s, available_actions, env)
            # Choose A from S using policy derived from Q
            a = ut.choose_action(Q, s, available_actions, epsilon)
            # Take action A, observe R, S'
            reward, s_prime, available_actions_prime = ut.observe_R_S_prime(env, a)
            Q = ut.update_Q(Q, s_prime, available_actions_prime, env)
            if not env.is_game_over():
                # Calcul Q(s,a)
                Q[s][a] = calcul_Q(Q, s, s_prime, a, reward, available_actions_prime, gamma, alpha, env)
            else:
                Q[s][a] = Q[s][a] + alpha * reward
    return Q


def play_game(game, parameters, results_path):
    alpha = parameters["alpha"]
    epsilon = parameters["epsilon"]
    gamma = parameters["gamma"]
    nb_iter = parameters["nb_iter"]
    match game:
        case "LineWorld":
            config = ut.load_config(congig_file, game)
            env = lw.LineWorld(config)
        case "GridWorld":
            config = ut.load_config(congig_file, game)
            env = gw.GridWorld(config)
        case "MontyHall1":
            config = ut.load_config(congig_file, game)
            env = mh1.MontyHall1(config)
        case "MontyHall2":
            config = ut.load_config(congig_file, game)
            env = mh2.MontyHall2(config)
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
    Pi = ut.calcul_policy(Q_optimal)
    score = env.score()
    ut.save_results_to_pickle(parameters, Q_optimal, Pi, score, results_path)
    # if game == "MontyHall1":
    #     #ut.play_montyhall1(env, Pi)
    # if game == "MontyHall2":
    #     #ut.play_montyhall2(env, Pi)
    # else:
    #     #env.reset()
    #     #ut.play_a_game_by_Pi(env, Pi)


if __name__ == '__main__':
    game = "MontyHall1"
    parameters = {"alpha": 0.1, "epsilon": 0.1, "gamma": 0.999, "nb_iter": 10000}
    results_path = f"../results/{game}_q_learning.pkl"
    play_game(game, parameters, results_path)



