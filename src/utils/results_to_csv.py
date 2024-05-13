import json
import re
import sys
from pathlib import Path

import pandas as pd

RESUTLS_DIR: str = "/home/usuaris/imatge/jorge.pueyo/TFM/racoon-paper-tnsm-gym/results"
EXPERIMENT_DIR: str = sys.argv[1]

PATTERN_NETWORK_SIZE = re.compile("NETWORK_SIZE_([0-9]+)")
PATTERN_N_STEPS_GAME = re.compile("STEPS_GAME_([0-9]+)")
PATTERN_N_TIMESTEPS = re.compile("N_TRAIN_STEPS_([0-9]+)")
PATTERN_WEIGHTED = re.compile("WEIGHTED_([0-9\-]+)")
PATTERN_PATHS = re.compile("K_PATHS_([0-9]+)")
PATTERN_ROUTING = re.compile("ROUTING_([A-Z]+_*[A-Z]*)")
PATTERN_SAMPLE = re.compile("sample_([0-9]+).json")
PATTERN_NODE_PROB = re.compile("NODE_PROB_([0-9\.]+)")
PATTERN_LINK_REMOVAL = re.compile("LINKS_REMOVED_([0-9])")


def fetch_data() -> pd.DataFrame:
    exp_path = Path(f"{RESUTLS_DIR}/{EXPERIMENT_DIR}")
    data: list = []

    for cfg in exp_path.iterdir():
        k_paths: int = int(PATTERN_PATHS.search(cfg.as_posix()).group(1))
        steps_game: int = int(PATTERN_N_STEPS_GAME.search(cfg.as_posix()).group(1))
        training_steps: int = int(PATTERN_N_TIMESTEPS.search(cfg.as_posix()).group(1))
        network_size: int = int(PATTERN_NETWORK_SIZE.search(cfg.as_posix()).group(1))
        routing_algorithm: str = PATTERN_ROUTING.search(cfg.as_posix()).group(1).lower()
        node_prob: float = float(PATTERN_NODE_PROB.search(cfg.as_posix()).group(1))
        links_removed: int = int(PATTERN_LINK_REMOVAL.search(cfg.as_posix()).group(1))

        cfg_data: dict = {
            "k_paths": k_paths,
            "steps_game": steps_game,
            "training_steps": training_steps,
            "network_size": network_size,
            "routing_algorithm": routing_algorithm,
            "links_removed": links_removed,
            "node_active_probability": node_prob,
        }

        for file in cfg.iterdir():
            with file.open() as f:
                n_sample = int(PATTERN_SAMPLE.search(file.as_posix()).group(1))
                sample: dict = json.load(f)

                sample["n_sample"] = n_sample
                sample.update(cfg_data)
                data.append(sample)
    return pd.DataFrame(data=data)


def main():
    df = fetch_data()
    df.to_csv(
        f"/home/usuaris/imatge/jorge.pueyo/TFM/racoon-paper-tnsm-gym/results_csv/{EXPERIMENT_DIR}.csv"
    )


if __name__ == "__main__":
    main()
