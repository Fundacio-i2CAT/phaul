import os
import random
import time

import numpy as np

from bh_network.het_bhnet import MMWAVE, SUB6
from gym_env.RacoonSimEnv import RacoonSimEnv


def RandomPolicyList(n_flows):
    action = []
    for i in range(n_flows):
        r = random.randint(0, 1)
        if r == 0:
            action.append(MMWAVE)
        else:
            action.append(SUB6)

    return action


def write_results(
    env: RacoonSimEnv,
    cfg: dict,
    exp_dir: str,
    s: int,
    max_efficiency: int,
    max_fairness: int,
    total_time_inference: int,
    optimal_allocation_inf: list,
):
    print(f"Sample {str(s)} finished, saving results.")

    if os.path.isdir(f"{cfg['results_dir']}/{exp_dir}/val_sample_{str(s)}") is False:
        os.mkdir(f"{cfg['results_dir']}/{exp_dir}/val_sample_{str(s)}")

    results_filename = f"{cfg['results_dir']}/{exp_dir}/val_sample_{str(s)}/results_summary_sample_{str(s)}"

    # Write results for the agent's inference

    results_file = open(results_filename, "a")

    results_file.write("\n### CONDITIONS ###")
    for f in env.het_bhnet.flow_list:
        results_file.write(f"\nFlow {f.index}: {f.data_rate} Mbps")

    results_file.write("\n### INFERENCE ###")
    results_file.write(f"\nmax_observed_load_eff: {str(max_efficiency)}")
    results_file.write(f"\nmax_observed_numflow_perc: {str(max_fairness)}")
    results_file.write(f"\nOptimal Allocation: {str(optimal_allocation_inf)}")
    results_file.write(f"\nRequired Time: {str(total_time_inference)}")
    results_file.close()

    # Write results for the Brute Force method
    if cfg["bruteforce_flag"] and cfg["network_size"] <= 20:
        start_time_bruteforce = time.perf_counter()
        (
            bf_efficiency,
            bf_fairness,
            optimal_allocation_bf,
            _,
        ) = env._brute_force_optimal_flow_allocation()
        total_time_bruteforce = time.perf_counter() - start_time_bruteforce

        results_file = open(results_filename, "a")
        results_file.write("\n### BRUTE FORCE ###")
        results_file.write(f"\nBF Load Efficiency: {str(bf_efficiency)}")
        results_file.write(f"\nBF Fairness: {str(bf_fairness)}")
        results_file.write(f"\nOptimal Allocation: {str(optimal_allocation_bf)}")
        results_file.write(f"\nRequired Time: {str(total_time_bruteforce)}")
        results_file.close()

    # Write results for the Random method
    if cfg["rnd_policy_flag"]:
        start_time_random = time.perf_counter()
        random_action = RandomPolicyList(cfg["network_size"])
        random_load_eff, random_numflow_perc = env._arbitrary_policy_allocation(
            random_action
        )
        total_time_random = time.perf_counter() - start_time_random

        results_file = open(results_filename, "a")
        results_file.write("\n### RANDOM POLICY ###")
        results_file.write(f"\nRandom Load Efficiency: {str(random_load_eff)}")
        results_file.write(f"\nRandom Fairness: {str(random_numflow_perc)}")
        results_file.write(f"\nOptimal Allocation: {str(random_action)}")
        results_file.write(f"\nRequired Time: {str(total_time_random)}")
        results_file.close()

    # Write results fort the Subset Sum method
    if cfg["subsetsum_flag"]:
        start_time_subset = time.perf_counter()
        (
            subsetsum_load_eff,
            subsetsum_numflow_perc,
            subsetsum_alloc,
            _,
        ) = env._subset_sum_problem_half_heuristic()
        total_time_subset = time.perf_counter() - start_time_subset

        results_file = open(results_filename, "a")
        results_file.write("\n### SUBSETSUM POLICY ###")
        results_file.write(f"\nmax_observed_load_eff: {str(subsetsum_load_eff)}")
        results_file.write(
            f"\nmax_observed_numflow_perc: {str(subsetsum_numflow_perc)}"
        )
        results_file.write(f"\nOptimal Allocation: {str(subsetsum_alloc)}")
        results_file.write(f"\nRequired Time: {str(total_time_subset)}")
        results_file.close()
