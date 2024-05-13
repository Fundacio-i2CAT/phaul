import json
import multiprocessing
import random
import time
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from yaml.loader import FullLoader

from alternative_algorithms import bruteforce, rnd_policy, subsetsum
from gym_env.RacoonSimEnv import RacoonSimEnv
from utils.logger import write_results


def write_results(data: dict, sample: int, save_dir: str) -> None:
    print(f"Sample {sample} finished, saving results.")

    with open(f"{save_dir}/sample_{sample}.json", "w") as f:
        json.dump(data, f, indent=1)


def create_save_dir(cfg: dict) -> str:
    exp_dir = (
        f"{cfg['exp_name']}/K_PATHS_{str(cfg['k_paths'])}"
        f"_STEPS_GAME_{str(cfg['steps_game'])}"
        f"_N_TRAIN_STEPS_{str(cfg['training_steps'])}"
        f"_NETWORK_SIZE_{str(cfg['network_size'])}"
        f"_NODE_PROB_{str(cfg['node_active_prob'])}"
        f"_LINKS_REMOVED_{str(cfg['links_to_remove'])}"
    )

    if cfg["weighted_reward"]:
        exp_dir = f"{exp_dir}_WEIGHTED_{cfg['weighted_reward']}"

    if cfg["routing_algorithm"]:
        exp_dir = f"{exp_dir}_ROUTING_{cfg['routing_algorithm'].upper()}"

    Path(cfg["results_dir"]).mkdir(exist_ok=True)
    Path(f"{cfg['results_dir']}/{cfg['exp_name']}").mkdir(exist_ok=True)
    Path(f"{cfg['results_dir']}/{exp_dir}").mkdir(exist_ok=True)

    return f"{cfg['results_dir']}/{exp_dir}"


def train(env: RacoonSimEnv, cfg: dict) -> PPO:
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=cfg["training_steps"])

    return model


def test(
    model: PPO, env: RacoonSimEnv, cfg: dict, save_dir: str, n_topology: int,
) -> None:
    # Each validation sample runs for a number of steps until the episode concludes.

    #original_G = env.het_bhnet.mm_network.G.copy()
    #env.het_bhnet.mm_network.remove_n_links(cfg["links_to_remove"])
    for s in range(cfg["validation_samples"]):
        # Set seed to ensure consistency between experiments
        random.seed(s)
        np.random.seed(s)

        """ env.reset_network_topology(
            cfg["network_size"],
            cfg["root_nodes"],
            cfg["max_child_nodes"],
            cfg["mm_edge_prob"],
            cfg["sub6_edge_prob"],
        )  # Reset topology for inference validation purposes """
        obs = env.reset()

        max_efficiency: int = 0
        max_fairness: int = 0
        max_alloc_rank: int = 0
        optimal_allocation: int = []

        init_time = time.perf_counter()

        for step in range(cfg["steps_game"]):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            # Checking the maximum observed
            alloc_rank = (1 - (100 - info["efficiency"]) / 100) + (
                1 - (100 - info["fairness"]) / 100
            )

            if alloc_rank > max_alloc_rank:
                max_alloc_rank = alloc_rank
                max_efficiency = info["efficiency"]
                max_fairness = info["fairness"]
                optimal_allocation = info["last_allocation_vector"]

            if done:
                total_time_inference = time.perf_counter() - init_time

                results = {
                    "efficiency": max_efficiency,
                    "fairness": max_fairness,
                    "time": total_time_inference,
                }

                if cfg["bruteforce_flag"] and cfg["network_size"] <= 20:
                    results.update(bruteforce(env))

                if cfg["subsetsum_flag"]:
                    results.update(subsetsum(env))

                if cfg["rnd_policy_flag"]:
                    results.update(rnd_policy(env))

                sample = s + n_topology * cfg["validation_samples"]
                write_results(results, sample, save_dir)

                #env.het_bhnet.mm_network.G = original_G


def main(cfg: DictConfig):
    print(cfg)
    exp_dir = create_save_dir(cfg)

    for topology in range(cfg["n_topologies"]):
        random.seed(topology)
        np.random.seed(topology)

        env = RacoonSimEnv(
            cfg["network_size"],
            cfg["root_nodes"],
            cfg["max_child_nodes"],
            cfg["mm_edge_prob"],
            cfg["sub6_edge_prob"],
            cfg["mm_capacity"],
            cfg["sub6_capacity"],
            cfg["min_flow_datarate"],
            cfg["max_flow_datarate"],
            cfg["node_active_prob"],
            cfg["steps_game"],
            cfg["k_paths"],
            cfg["weighted_reward"],
            cfg["routing_algorithm"],
        )

        env.het_bhnet.mm_network.remove_n_links(cfg["links_to_remove"])
        model = train(env, cfg)
        test(model, env, cfg, exp_dir, topology)


def multirun():
    with open(
        "/home/usuaris/imatge/jorge.pueyo/TFM/racoon-paper-tnsm-gym/src/cfg.yaml",
        "r",
    ) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    for links_to_remove in [0, 1, 2, 3, 4, 5]:
        for k_paths in [1, 3]:
            cfg["links_to_remove"] = links_to_remove
            cfg["k_paths"] = k_paths

            multiprocessing.Process(target=main, args=(cfg,)).start()


if __name__ == "__main__":
    multirun()
