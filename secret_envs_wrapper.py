import ctypes
import platform

import numpy as np

if platform.system().lower() == "windows":
    lib_path = "./libs/secret_envs.dll"
elif platform.system().lower() == "linux":
    lib_path = "./libs/libsecret_envs.so"
elif platform.system().lower() == "darwin":
    if "intel" in platform.processor().lower():
        lib_path = "./libs/libsecret_envs_intel_macos.dylib"
    else:
        lib_path = "./libs/libsecret_envs.dylib"


class SecretEnv0Wrapper:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        # MDP functions
        self.lib.secret_env_0_num_states.argtypes = []
        self.lib.secret_env_0_num_states.restype = ctypes.c_size_t

        self.lib.secret_env_0_num_actions.argtypes = []
        self.lib.secret_env_0_num_actions.restype = ctypes.c_size_t

        self.lib.secret_env_0_num_rewards.argtypes = []
        self.lib.secret_env_0_num_rewards.restype = ctypes.c_size_t

        self.lib.secret_env_0_reward.argtypes = []
        self.lib.secret_env_0_reward.restype = ctypes.c_float

        self.lib.secret_env_0_transition_probability.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                                                 ctypes.c_size_t]
        self.lib.secret_env_0_transition_probability.restype = ctypes.c_float

        # Monte Carlo and TD Methods
        self.lib.secret_env_0_new.argtypes = []
        self.lib.secret_env_0_new.restype = ctypes.c_void_p

        self.lib.secret_env_0_new.argtypes = []
        self.lib.secret_env_0_new.restype = ctypes.c_void_p

        self.lib.secret_env_0_new.argtypes = []
        self.lib.secret_env_0_new.restype = ctypes.c_void_p

        self.lib.secret_env_0_reset.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_reset.restype = None

        self.lib.secret_env_0_display.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_display.restype = None

        self.lib.secret_env_0_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_0_is_forbidden.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_0_is_forbidden.restype = ctypes.c_bool

        self.lib.secret_env_0_is_game_over.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_is_game_over.restype = ctypes.c_bool

        self.lib.secret_env_0_available_actions.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_available_actions.restype = ctypes.POINTER(ctypes.c_size_t)

        self.lib.secret_env_0_available_actions_len.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_available_actions_len.restype = ctypes.c_size_t

        self.lib.secret_env_0_available_actions_delete.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        self.lib.secret_env_0_available_actions_delete.restype = None

        self.lib.secret_env_0_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_0_step.restype = None

        self.lib.secret_env_0_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_0_score.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_score.restype = ctypes.c_float

        self.lib.secret_env_0_delete.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_0_delete.restype = None

        self.lib.secret_env_0_from_random_state.argtypes = []
        self.lib.secret_env_0_from_random_state.restype = ctypes.c_void_p


class SecretEnv0:
    def __init__(self, wrapper=None, instance=None):
        if wrapper is None:
            wrapper = SecretEnv0Wrapper()
        self.wrapper = wrapper
        if instance is None:
            instance = self.wrapper.lib.secret_env_0_new()
        self.instance = instance

    def __del__(self):
        if self.wrapper is not None:
            self.wrapper.lib.secret_env_0_delete(self.instance)

    # MDP related Methods
    def num_states(self) -> int:
        return self.wrapper.lib.secret_env_0_num_states()

    def num_actions(self) -> int:
        return self.wrapper.lib.secret_env_0_num_actions()

    def num_rewards(self) -> int:
        return self.wrapper.lib.secret_env_0_num_rewards()

    def reward(self, i: int) -> float:
        return self.wrapper.lib.secret_env_0_reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.wrapper.lib.secret_env_0_transition_probability(s, a, s_p, r_index)

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return self.wrapper.lib.secret_env_0_state_id(self.instance)

    def reset(self):
        self.wrapper.lib.secret_env_0_reset(self.instance)

    def display(self):
        self.wrapper.lib.secret_env_0_display(self.instance)

    def is_forbidden(self, action: int) -> int:
        return self.wrapper.lib.secret_env_0_is_forbidden(self.instance, action)

    def is_game_over(self) -> bool:
        return self.wrapper.lib.secret_env_0_is_game_over(self.instance)

    def available_actions(self) -> np.ndarray:
        actions_len = self.wrapper.lib.secret_env_0_available_actions_len(self.instance)
        actions_pointer = self.wrapper.lib.secret_env_0_available_actions(self.instance)
        arr = np.ctypeslib.as_array(actions_pointer, (actions_len,))
        arr_copy = np.copy(arr)
        self.wrapper.lib.secret_env_0_available_actions_delete(actions_pointer, actions_len)
        return arr_copy

    def step(self, action: int):
        self.wrapper.lib.secret_env_0_step(self.instance, action)

    def score(self):
        return self.wrapper.lib.secret_env_0_score(self.instance)

    @staticmethod
    def from_random_state() -> 'SecretEnv0':
        wrapper = SecretEnv0Wrapper()
        instance = wrapper.lib.secret_env_0_from_random_state()
        return SecretEnv0(wrapper, instance)


class SecretEnv1Wrapper:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        # MDP functions
        self.lib.secret_env_1_num_states.argtypes = []
        self.lib.secret_env_1_num_states.restype = ctypes.c_size_t

        self.lib.secret_env_1_num_actions.argtypes = []
        self.lib.secret_env_1_num_actions.restype = ctypes.c_size_t

        self.lib.secret_env_1_num_rewards.argtypes = []
        self.lib.secret_env_1_num_rewards.restype = ctypes.c_size_t

        self.lib.secret_env_1_reward.argtypes = []
        self.lib.secret_env_1_reward.restype = ctypes.c_float

        self.lib.secret_env_1_transition_probability.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                                                 ctypes.c_size_t]
        self.lib.secret_env_1_transition_probability.restype = ctypes.c_float

        # Monte Carlo and TD Methods
        self.lib.secret_env_1_new.argtypes = []
        self.lib.secret_env_1_new.restype = ctypes.c_void_p

        self.lib.secret_env_1_new.argtypes = []
        self.lib.secret_env_1_new.restype = ctypes.c_void_p

        self.lib.secret_env_1_new.argtypes = []
        self.lib.secret_env_1_new.restype = ctypes.c_void_p

        self.lib.secret_env_1_reset.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_reset.restype = None

        self.lib.secret_env_1_display.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_display.restype = None

        self.lib.secret_env_1_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_1_is_forbidden.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_1_is_forbidden.restype = ctypes.c_bool

        self.lib.secret_env_1_is_game_over.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_is_game_over.restype = ctypes.c_bool

        self.lib.secret_env_1_available_actions.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_available_actions.restype = ctypes.POINTER(ctypes.c_size_t)

        self.lib.secret_env_1_available_actions_len.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_available_actions_len.restype = ctypes.c_size_t

        self.lib.secret_env_1_available_actions_delete.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        self.lib.secret_env_1_available_actions_delete.restype = None

        self.lib.secret_env_1_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_1_step.restype = None

        self.lib.secret_env_1_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_1_score.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_score.restype = ctypes.c_float

        self.lib.secret_env_1_delete.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_1_delete.restype = None

        self.lib.secret_env_1_from_random_state.argtypes = []
        self.lib.secret_env_1_from_random_state.restype = ctypes.c_void_p


class SecretEnv1:
    def __init__(self, wrapper=None, instance=None):
        if wrapper is None:
            wrapper = SecretEnv1Wrapper()
        self.wrapper = wrapper
        if instance is None:
            instance = self.wrapper.lib.secret_env_1_new()
        self.instance = instance

    def __del__(self):
        if self.wrapper is not None:
            self.wrapper.lib.secret_env_1_delete(self.instance)

    # MDP related Methods
    def num_states(self) -> int:
        return self.wrapper.lib.secret_env_1_num_states()

    def num_actions(self) -> int:
        return self.wrapper.lib.secret_env_1_num_actions()

    def num_rewards(self) -> int:
        return self.wrapper.lib.secret_env_1_num_rewards()

    def reward(self, i: int) -> float:
        return self.wrapper.lib.secret_env_1_reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.wrapper.lib.secret_env_1_transition_probability(s, a, s_p, r_index)

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return self.wrapper.lib.secret_env_1_state_id(self.instance)

    def reset(self):
        self.wrapper.lib.secret_env_1_reset(self.instance)

    def display(self):
        self.wrapper.lib.secret_env_1_display(self.instance)

    def is_forbidden(self, action: int) -> int:
        return self.wrapper.lib.secret_env_1_is_forbidden(self.instance, action)

    def is_game_over(self) -> bool:
        return self.wrapper.lib.secret_env_1_is_game_over(self.instance)

    def available_actions(self) -> np.ndarray:
        actions_len = self.wrapper.lib.secret_env_1_available_actions_len(self.instance)
        actions_pointer = self.wrapper.lib.secret_env_1_available_actions(self.instance)
        arr = np.ctypeslib.as_array(actions_pointer, (actions_len,))
        arr_copy = np.copy(arr)
        self.wrapper.lib.secret_env_1_available_actions_delete(actions_pointer, actions_len)
        return arr_copy

    def step(self, action: int):
        self.wrapper.lib.secret_env_1_step(self.instance, action)

    def score(self):
        return self.wrapper.lib.secret_env_1_score(self.instance)

    @staticmethod
    def from_random_state() -> 'SecretEnv1':
        wrapper = SecretEnv1Wrapper()
        instance = wrapper.lib.secret_env_1_from_random_state()
        return SecretEnv1(wrapper, instance)

class SecretEnv2Wrapper:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        # MDP functions
        self.lib.secret_env_2_num_states.argtypes = []
        self.lib.secret_env_2_num_states.restype = ctypes.c_size_t

        self.lib.secret_env_2_num_actions.argtypes = []
        self.lib.secret_env_2_num_actions.restype = ctypes.c_size_t

        self.lib.secret_env_2_num_rewards.argtypes = []
        self.lib.secret_env_2_num_rewards.restype = ctypes.c_size_t

        self.lib.secret_env_2_reward.argtypes = []
        self.lib.secret_env_2_reward.restype = ctypes.c_float

        self.lib.secret_env_2_transition_probability.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                                                 ctypes.c_size_t]
        self.lib.secret_env_2_transition_probability.restype = ctypes.c_float

        # Monte Carlo and TD Methods
        self.lib.secret_env_2_new.argtypes = []
        self.lib.secret_env_2_new.restype = ctypes.c_void_p

        self.lib.secret_env_2_new.argtypes = []
        self.lib.secret_env_2_new.restype = ctypes.c_void_p

        self.lib.secret_env_2_new.argtypes = []
        self.lib.secret_env_2_new.restype = ctypes.c_void_p

        self.lib.secret_env_2_reset.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_reset.restype = None

        self.lib.secret_env_2_display.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_display.restype = None

        self.lib.secret_env_2_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_2_is_forbidden.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_2_is_forbidden.restype = ctypes.c_bool

        self.lib.secret_env_2_is_game_over.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_is_game_over.restype = ctypes.c_bool

        self.lib.secret_env_2_available_actions.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_available_actions.restype = ctypes.POINTER(ctypes.c_size_t)

        self.lib.secret_env_2_available_actions_len.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_available_actions_len.restype = ctypes.c_size_t

        self.lib.secret_env_2_available_actions_delete.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        self.lib.secret_env_2_available_actions_delete.restype = None

        self.lib.secret_env_2_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_2_step.restype = None

        self.lib.secret_env_2_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_2_score.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_score.restype = ctypes.c_float

        self.lib.secret_env_2_delete.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_delete.restype = None

        self.lib.secret_env_2_from_random_state.argtypes = []
        self.lib.secret_env_2_from_random_state.restype = ctypes.c_void_p


class SecretEnv2:
    def __init__(self, wrapper=None, instance=None):
        if wrapper is None:
            wrapper = SecretEnv2Wrapper()
        self.wrapper = wrapper
        if instance is None:
            instance = self.wrapper.lib.secret_env_2_new()
        self.instance = instance

    def __del__(self):
        if self.wrapper is not None:
            self.wrapper.lib.secret_env_2_delete(self.instance)

    # MDP related Methods
    def num_states(self) -> int:
        return self.wrapper.lib.secret_env_2_num_states()

    def num_actions(self) -> int:
        return self.wrapper.lib.secret_env_2_num_actions()

    def num_rewards(self) -> int:
        return self.wrapper.lib.secret_env_2_num_rewards()

    def reward(self, i: int) -> float:
        return self.wrapper.lib.secret_env_2_reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.wrapper.lib.secret_env_2_transition_probability(s, a, s_p, r_index)

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return self.wrapper.lib.secret_env_2_state_id(self.instance)

    def reset(self):
        self.wrapper.lib.secret_env_2_reset(self.instance)

    def display(self):
        self.wrapper.lib.secret_env_2_display(self.instance)

    def is_forbidden(self, action: int) -> int:
        return self.wrapper.lib.secret_env_2_is_forbidden(self.instance, action)

    def is_game_over(self) -> bool:
        return self.wrapper.lib.secret_env_2_is_game_over(self.instance)

    def available_actions(self) -> np.ndarray:
        actions_len = self.wrapper.lib.secret_env_2_available_actions_len(self.instance)
        actions_pointer = self.wrapper.lib.secret_env_2_available_actions(self.instance)
        arr = np.ctypeslib.as_array(actions_pointer, (actions_len,))
        arr_copy = np.copy(arr)
        self.wrapper.lib.secret_env_2_available_actions_delete(actions_pointer, actions_len)
        return arr_copy

    def step(self, action: int):
        self.wrapper.lib.secret_env_2_step(self.instance, action)

    def score(self):
        return self.wrapper.lib.secret_env_2_score(self.instance)

    @staticmethod
    def from_random_state() -> 'SecretEnv2':
        wrapper = SecretEnv2Wrapper()
        instance = wrapper.lib.secret_env_2_from_random_state()
        return SecretEnv2(wrapper, instance)


class SecretEnv3Wrapper:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        # MDP functions
        self.lib.secret_env_3_num_states.argtypes = []
        self.lib.secret_env_3_num_states.restype = ctypes.c_size_t

        self.lib.secret_env_3_num_actions.argtypes = []
        self.lib.secret_env_3_num_actions.restype = ctypes.c_size_t

        self.lib.secret_env_3_num_rewards.argtypes = []
        self.lib.secret_env_3_num_rewards.restype = ctypes.c_size_t

        self.lib.secret_env_3_reward.argtypes = []
        self.lib.secret_env_3_reward.restype = ctypes.c_float

        self.lib.secret_env_3_transition_probability.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
                                                                 ctypes.c_size_t]
        self.lib.secret_env_3_transition_probability.restype = ctypes.c_float

        # Monte Carlo and TD Methods
        self.lib.secret_env_3_new.argtypes = []
        self.lib.secret_env_3_new.restype = ctypes.c_void_p

        self.lib.secret_env_3_new.argtypes = []
        self.lib.secret_env_3_new.restype = ctypes.c_void_p

        self.lib.secret_env_3_new.argtypes = []
        self.lib.secret_env_3_new.restype = ctypes.c_void_p

        self.lib.secret_env_3_reset.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_reset.restype = None

        self.lib.secret_env_3_display.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_display.restype = None

        self.lib.secret_env_3_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_3_is_forbidden.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_3_is_forbidden.restype = ctypes.c_bool

        self.lib.secret_env_3_is_game_over.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_is_game_over.restype = ctypes.c_bool

        self.lib.secret_env_3_available_actions.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_available_actions.restype = ctypes.POINTER(ctypes.c_size_t)

        self.lib.secret_env_3_available_actions_len.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_available_actions_len.restype = ctypes.c_size_t

        self.lib.secret_env_3_available_actions_delete.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
        self.lib.secret_env_3_available_actions_delete.restype = None

        self.lib.secret_env_3_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_3_step.restype = None

        self.lib.secret_env_3_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_3_score.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_score.restype = ctypes.c_float

        self.lib.secret_env_3_delete.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_delete.restype = None

        self.lib.secret_env_3_from_random_state.argtypes = []
        self.lib.secret_env_3_from_random_state.restype = ctypes.c_void_p


class SecretEnv3:
    def __init__(self, wrapper=None, instance=None):
        if wrapper is None:
            wrapper = SecretEnv3Wrapper()
        self.wrapper = wrapper
        if instance is None:
            instance = self.wrapper.lib.secret_env_3_new()
        self.instance = instance

    def __del__(self):
        if self.wrapper is not None:
            self.wrapper.lib.secret_env_3_delete(self.instance)

    # MDP related Methods
    def num_states(self) -> int:
        return self.wrapper.lib.secret_env_3_num_states()

    def num_actions(self) -> int:
        return self.wrapper.lib.secret_env_3_num_actions()

    def num_rewards(self) -> int:
        return self.wrapper.lib.secret_env_3_num_rewards()

    def reward(self, i: int) -> float:
        return self.wrapper.lib.secret_env_3_reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.wrapper.lib.secret_env_3_transition_probability(s, a, s_p, r_index)

    # Monte Carlo and TD Methods related functions:
    def state_id(self) -> int:
        return self.wrapper.lib.secret_env_3_state_id(self.instance)

    def reset(self):
        self.wrapper.lib.secret_env_3_reset(self.instance)

    def display(self):
        self.wrapper.lib.secret_env_3_display(self.instance)

    def is_forbidden(self, action: int) -> int:
        return self.wrapper.lib.secret_env_3_is_forbidden(self.instance, action)

    def is_game_over(self) -> bool:
        return self.wrapper.lib.secret_env_3_is_game_over(self.instance)

    def available_actions(self) -> np.ndarray:
        actions_len = self.wrapper.lib.secret_env_3_available_actions_len(self.instance)
        actions_pointer = self.wrapper.lib.secret_env_3_available_actions(self.instance)
        arr = np.ctypeslib.as_array(actions_pointer, (actions_len,))
        arr_copy = np.copy(arr)
        self.wrapper.lib.secret_env_3_available_actions_delete(actions_pointer, actions_len)
        return arr_copy

    def step(self, action: int):
        self.wrapper.lib.secret_env_3_step(self.instance, action)

    def score(self):
        return self.wrapper.lib.secret_env_3_score(self.instance)

    @staticmethod
    def from_random_state() -> 'SecretEnv3':
        wrapper = SecretEnv3Wrapper()
        instance = wrapper.lib.secret_env_3_from_random_state()
        return SecretEnv3(wrapper, instance)


if __name__ == "__main__":
    env = SecretEnv0()
    print(env.num_states())
    print(env.num_actions())
    print(env.num_rewards())
    for i in range(env.num_rewards()):
        print(env.reward(i))
    print(env.p(0, 0, 0, 0))

    print(env.available_actions())

    while not env.is_game_over():
        env.display()
        env.step(env.available_actions()[0])
    env.display()

    print(env.score())

    random_state_env = SecretEnv0.from_random_state()
    print(random_state_env.available_actions())

    env = SecretEnv1()
    print(env.num_states())
    print(env.num_actions())
    print(env.num_rewards())
    for i in range(env.num_rewards()):
        print(env.reward(i))
    print(env.p(0, 0, 0, 0))

    print(env.available_actions())

    while not env.is_game_over():
        env.display()
        env.step(env.available_actions()[1])
    env.display()

    print(env.score())

    random_state_env = SecretEnv1.from_random_state()
    print(random_state_env.available_actions())

    env = SecretEnv2()
    print(env.num_states())
    print(env.num_actions())
    print(env.num_rewards())
    for i in range(env.num_rewards()):
        print(env.reward(i))
    print(env.p(0, 0, 0, 0))

    print(env.available_actions())

    while not env.is_game_over():
        env.display()
        env.step(env.available_actions()[1])
    env.display()

    print(env.score())

    random_state_env = SecretEnv2.from_random_state()
    print(random_state_env.available_actions())

    env = SecretEnv3()
    print(env.num_states())
    print(env.num_actions())
    print(env.num_rewards())
    for i in range(env.num_rewards()):
        print(env.reward(i))
    print(env.p(0, 0, 0, 0))

    print(env.available_actions())

    while not env.is_game_over():
        env.display()
        env.step(env.available_actions()[1])
    env.display()

    print(env.score())

    random_state_env = SecretEnv3.from_random_state()
    print(random_state_env.available_actions())
