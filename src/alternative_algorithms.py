import time

import numpy as np

from gym_env.RacoonSimEnv import RacoonSimEnv


def subsetsum(env: RacoonSimEnv) -> dict:
    init_time = time.perf_counter()

    efficiency, fairness, alloc = env._subset_sum_problem_half_heuristic()

    total_time = time.perf_counter() - init_time

    return {
        "subsetsum_efficiency": efficiency,
        "subsetsum_fairness": fairness,
        "subsetsum_time": total_time,
    }


def bruteforce(env: RacoonSimEnv) -> dict:
    init_time = time.perf_counter()

    efficiency, fairness, alloc = env._brute_force_optimal_flow_allocation()

    total_time = time.perf_counter() - init_time

    return {
        "bf_efficiency": efficiency,
        "bf_fairness": fairness,
        "bf_time": total_time,
    }


def rnd_policy(env: RacoonSimEnv) -> dict:
    init_time = time.perf_counter()

    rnd_alloc = list(np.random.randint(2, size=env.n_flows))

    efficiency, fairness = env._arbitrary_policy_allocation(rnd_alloc)

    total_time = time.perf_counter() - init_time

    return {
        "rnd_efficiency": efficiency,
        "rnd_fairness": fairness,
        "rnd_time": total_time,
    }
