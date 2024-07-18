import yaml


def load_config(config_file, env_name):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config[env_name]
