import yaml
import random
import pickle
def load_config(config_file, env_name):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config[env_name]

def calcul_policy(Q):
    Pi = {}
    for s in Q.keys():
        best_a = None
        best_a_score = 0.0

        for a, a_score in Q[s].items():
            if best_a is None or best_a_score <= a_score:
                best_a = a
                best_a_score = a_score

        Pi[s] = best_a
    return Pi

def play_a_game_by_Pi(env,Pi):
    while not env.is_game_over():
        env.display()
        print(env.score())
        if env.state_id() in Pi:
            a = Pi[env.state_id()]
            if env.is_forbidden(a):
                a = random.choice(env.available_actions())
                env.step(a)
            else:
                env.step(a)
        else:
            a = random.choice(env.available_actions())
            env.step(a)
    env.display()
    print(env.score())
    with open('stratégie_optimal.pkl', 'wb') as fichier:
        pickle.dump(Pi, fichier)
