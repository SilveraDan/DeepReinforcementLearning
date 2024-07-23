from tqdm import tqdm
import secret_envs_wrapper
import environnements.lineworld as lw
from utils import load_config, calcul_policy, play_a_game_by_Pi, choose_action, update_Q, observe_R_S_prime, save_results_to_pickle
import environnements.gridworld as gw


congig_file = "./config.yaml"


def sarsa(env, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 0.999, nb_iter: int = 500):
    Q = {}
    # Loop for each episode
    for _ in tqdm(range(nb_iter)):
        # Initialize S
        env.reset()
        s = env.state_id()
        available_actions = env.available_actions()
        Q = update_Q(Q, s, available_actions, env)
        # Choose A from S using policy derived from Q
        a = choose_action(Q, s, available_actions, epsilon)
        while not env.is_game_over():
            # Take action A, observe R, S'
            reward, s_prime, available_actions_prime = observe_R_S_prime(env, a)
            Q = update_Q(Q, s_prime, available_actions_prime, env)
            # Choose A' from S' using policy derived from Q
            a_prime = choose_action(Q, s_prime, available_actions_prime, epsilon)
            # Q(s,a) <- Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
            if not env.is_game_over():
                q_s_prime = Q[s_prime][a_prime]
                Q[s][a] = Q[s][a] + alpha * (reward + gamma * q_s_prime - Q[s][a])
            else:
                Q[s][a] = Q[s][a] + alpha * reward
            s = s_prime
            a = a_prime
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
    Q_optimal = sarsa(env, alpha, epsilon, gamma, nb_iter)
    Pi = calcul_policy(Q_optimal)
    env.reset()
    save_results_to_pickle(Q_optimal, Pi, results_path)
    #play_a_game_by_Pi(env, Pi)


# if __name__ == '__main__':
#     game = "GridWorld"
#     parameters = {"alpha": 0.1, "epsilon": 0.1, "gamma": 0.999, "nb_iter": 1000}
#     results_path = f"../results/{game}_sarsa.pkl"
#     play_game(game, parameters, results_path)
