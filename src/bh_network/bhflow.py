import json
from collections import defaultdict

import numpy as np

from bh_network.dag import print_graph


class BackhaulFlow:
    def __init__(
        self,
        index: int,
        source: int,
        root_nodes: int,
        min_datarate: int,
        max_datarate: int,
        active_prob: float,
        mm_network,
        sub6_network,
        k_paths: int,
        routing_algorithm: str,
        paths_in_use: defaultdict,
    ):
        self.index = index

        self.source = source
        self.destination: int = None
        self.root_nodes = root_nodes

        self.active_prob = active_prob
        self.data_rate = (
            round(np.random.uniform(min_datarate, max_datarate))
            if np.random.random() < self.active_prob
            else 0
        )
        self.mm_network = mm_network
        self.sub6_network = sub6_network
        self.k_paths = k_paths

        self.routing_algorithm = routing_algorithm
        self.mm_paths_in_use = paths_in_use[0]
        self.sub6_paths_in_use = paths_in_use[1]

        self.mm_paths, self.mm_paths_in_use = self.find_paths(
            self.mm_network, self.mm_paths_in_use
        )
        self.sub6_paths, self.sub6_paths_in_use = self.find_paths(
            self.sub6_network, self.sub6_paths_in_use
        )

        self.path: list = None
        self.assigned_path: list = None

        self.network_type = -1
        self.traversed_edges = {}
        self.traversed_nodes = {}
        self.effective_data_rate = self.data_rate

    def get_paths_in_use(self) -> tuple:
        return self.mm_paths_in_use, self.sub6_paths_in_use

    def find_paths(self, network, paths_in_use) -> tuple:
        paths: list = [p for p in network.find_paths(self)]

        if (missing_paths := (self.k_paths - len(paths))) > 0:
            for _ in range(missing_paths):
                try:
                    paths.append(paths[0])
                except IndexError:
                    print(paths)

        if self.routing_algorithm == "last_hop":
            paths = self._filter_last_hop(paths)

        elif self.routing_algorithm == "round_robin":
            paths, paths_in_use = self._filter_round_robin(paths, paths_in_use)

        elif self.routing_algorithm == "heuristic":
            paths, paths_in_use = self._filter_heuristic(paths, paths_in_use)

        return paths, paths_in_use

    def _filter_last_hop(self, paths: list) -> list:
        used_links: list = []
        refined_paths: list = []

        for path in paths:
            if len(refined_paths) == self.k_paths:
                break

            last_link = (path[-1], path[-2])

            if last_link not in used_links:
                refined_paths.append(path)
                used_links.append(last_link)

        # If not enough paths with different last link fill with shortest
        for path in paths:
            if len(refined_paths) == self.k_paths:
                break

            if path not in refined_paths:
                refined_paths.append(path)

        return refined_paths

    def _filter_round_robin(
        self, flow_paths: list, network_paths: defaultdict
    ) -> tuple[list, defaultdict]:
        filtered_paths = []

        for _ in range(self.k_paths):
            least_used_path: tuple[list, int] = (None, float("inf"))

            for f_path in flow_paths:
                if network_paths[str(f_path)] < least_used_path[1]:
                    least_used_path = f_path, network_paths[str(f_path)]

            filtered_paths.append(least_used_path[0])
            network_paths[str(least_used_path[0])] += 1

        return filtered_paths, network_paths

    def _filter_heuristic(
        self, flow_paths: list, network_paths: defaultdict
    ) -> tuple[list, defaultdict]:
        comparison_results: list[tuple[str, int]] = []
        filtered_paths: list = []

        for p1 in flow_paths:
            comp_result = sum(
                [
                    self._compare_paths(p1, json.loads(p2)) * m
                    for p2, m in network_paths.items()
                ]
            )
            comparison_results.append((p1, comp_result))

        comparison_results.sort(key=lambda x: x[1])

        for i in range(self.k_paths):
            filtered_paths.append(comparison_results[i][0])
            network_paths[str(comparison_results[i][0])] += 1

        return filtered_paths, network_paths

    def _compare_paths(self, p1: list[int], p2: list[int]) -> int:
        p1_links: set = {
            (p1[i], p1[i + 1]) for i, _ in enumerate(p1) if (i + 1) < len(p1)
        }
        p2_links: set = {
            (p2[i], p2[i + 1]) for i, _ in enumerate(p2) if (i + 1) < len(p2)
        }

        return len(p1_links & p2_links)

    def clear_state(self) -> None:
        # Note we maintain the source and destination of the flow
        self.network_type = -1
        self.path = None
        self.traversed_edges = {}
        self.traversed_nodes = {}
        self.effective_data_rate = self.data_rate

    def reset_datarate(self, min_datarate, max_datarate) -> None:
        self.data_rate = (
            round(np.random.uniform(min_datarate, max_datarate))
            if np.random.random() < self.active_prob
            else 0
        )
