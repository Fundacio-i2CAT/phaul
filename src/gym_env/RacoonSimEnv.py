import itertools
import random
import warnings
from bdb import effective

import gym
import networkx as nx
import numpy as np
from gym import spaces

from bh_network.bhflow import BackhaulFlow
from bh_network.het_bhnet import MMWAVE, SUB6, UNDEFINED, HetBackhaulNetwork

# Treat RunimeWarning as error
warnings.filterwarnings("error")


class RacoonSimEnv(gym.Env):
    def __init__(
        self,
        network_size: int,
        root_nodes: int,
        max_child_nodes: int,
        mm_edge_prob: float,
        sub6_edge_prob: float,
        mm_capacity: tuple[int, int],
        sub6_capacity: tuple[int, int],
        min_flow_datarate: int,
        max_flow_datarate: int,
        node_active_prob: float,
        steps_game: int,
        k_paths: int,
        weighted_reward: float = None,
        routing_algorithm: str = None,
    ):
        super(RacoonSimEnv, self).__init__()

        # Variable to store last observation and actions
        self.current_step = 0
        self.last_load_efficiency = -1
        self.last_fairness = -1
        self.max_load_efficiency = -1
        self.min_load_efficiency = 200
        self.max_allocation_fairness = -1
        self.min_allocation_fairness = 200
        self.max_reallocation_efficiency = -1
        self.min_reallocation_efficiency = 200

        # Environment variables
        self.network_size = network_size
        self.root_nodes = root_nodes
        self.n_flows = network_size - root_nodes
        self.max_child_nodes = max_child_nodes
        self.mm_edge_prob = mm_edge_prob
        self.sub6_edge_prob = sub6_edge_prob
        self.mm_capacity = mm_capacity
        self.sub6_capacity = sub6_capacity
        self.min_flow_datarate = min_flow_datarate
        self.max_flow_datarate = max_flow_datarate
        self.node_active_prob = node_active_prob
        self.steps_game = steps_game
        self.weighted_reward = weighted_reward
        self.k_paths = k_paths
        self.routing_algorithm = routing_algorithm

        # Generate HetBackhaul Network
        self.het_bhnet = HetBackhaulNetwork(
            self.network_size,
            self.root_nodes,
            self.max_child_nodes,
            self.mm_edge_prob,
            self.sub6_edge_prob,
            self.mm_capacity,
            self.sub6_capacity,
            self.min_flow_datarate,
            self.max_flow_datarate,
            self.node_active_prob,
            self.k_paths,
            self.routing_algorithm,
        )

        self.previous_allocation_vector = np.empty(
            self.n_flows,
        )
        self.last_allocation_vector = np.empty(
            self.n_flows,
        )

        self.n_actions = 2 * self.n_flows * self.k_paths
        self.action_space = spaces.Discrete(n=self.n_actions)

        # Observation space: For Sub6 and mmWave networks the Ingress and Egress load
        self.last_obs = np.empty([2, self.root_nodes + self.n_flows + 1])
        self.observation_space = spaces.Box(
            low=0,
            high=100 * self.mm_capacity[1],
            shape=(2, self.root_nodes + self.n_flows + 1),
            dtype=np.float32,
        )

        self.het_bhnet.generate_flows()
        self._init_random_allocation()
        self.last_obs = self._next_observation()

    def _init_random_allocation(self):
        for i in range(len(self.last_allocation_vector)):
            flow = self.het_bhnet.flow_list[i]
            r = random.randint(0, 1)
            if r == 0:
                self.last_allocation_vector[i] = MMWAVE
                flow.assigned_path = flow.mm_paths[0]

            else:
                self.last_allocation_vector[i] = SUB6
                flow.assigned_path = flow.sub6_paths[0]

            flow.path = flow.assigned_path
            flow.destination = flow.assigned_path[-1]

    def _next_observation(self):
        # Recover state from the network
        (
            allocation,
            mm_state,
            sub6_state,
        ) = self.het_bhnet.recover_network_state()
        # Defining the observation space from the mmWave and Sub6 network states (Ingress and Egress loads)
        obs = np.array([mm_state, sub6_state])
        self.last_obs = obs
        return obs

    def _take_action(self, action):
        """
        Action space A = {0,...,2*k_paths*n_flows}
        0<i<n_flows -> Allocate flow i in the Sub6 subnetwork, use path 1
        n_flows<i<2*n_flows -> Allocate flows i in the MM subnetwork, use path 1
        Extend for k paths
        """
        path: int = action // (2 * self.n_flows)
        target_flow_index: int = action % self.n_flows

        """ target_network: SUB6 | MMWAVE = (
            SUB6 if (action % (2 * self.n_flows)) < self.n_flows else MMWAVE
        ) """
        target_network = MMWAVE
        target_flow: BackhaulFlow = self.het_bhnet.flow_list[target_flow_index]

        target_flow.assigned_path = (
            target_flow.sub6_paths[path]
            if target_network == SUB6
            else target_flow.mm_paths[path]
        )
        target_flow.destination = target_flow.assigned_path[-1]

        action_list: list = self.last_allocation_vector
        action_list[target_flow_index] = target_network

        # Deallocate flows and then implement the new allocation
        # This is required, otherwise only the flows that change would be allocated
        # and we would not benefit from additional bandwidth that is freed up

        self.het_bhnet.deallocate_flows()
        self.het_bhnet.allocate_flows(action_list)

    def _compute_allocation_fairness(self):
        """Computing Jain's fairness index based on the effective data rate allocated for each flow"""

        effective_data_rates = np.array(
            [
                f.effective_data_rate
                for f in self.het_bhnet.flow_list
                if f.effective_data_rate > 0
            ]
        )
        n = effective_data_rates.size

        num = np.sum(effective_data_rates)
        den = np.sum(effective_data_rates**2)

        try:
            fairness_index = (num * num) / (n * den)
        except:
            fairness_index = 0

        return 100 * fairness_index

    def _compute_load_efficiency(self):
        """Define load efficiency as ratio between ingress and egress load multiplied by 100"""

        cum_ingress = np.sum(self.last_obs[:, 0 : self.n_flows])
        cum_egress = np.sum(
            self.last_obs[:, self.n_flows : (self.n_flows + self.root_nodes)]
        )

        try:
            efficiency = 100 * (cum_egress / cum_ingress)
        except:
            efficiency = 1

        return efficiency

    def _compute_reallocation_efficiency(self):
        if not np.any(self.previous_allocation_vector):
            re = (
                1
                - (
                    np.sum(
                        np.abs(
                            self.last_allocation_vector
                            - self.previous_allocation_vector
                        )
                    )
                )
                / self.n_flows
            )

        else:
            re = 0

        return re

    def _compute_reward(self):
        load_efficiency = self._compute_load_efficiency()

        allocation_fairness = self._compute_allocation_fairness()

        reallocation_efficiency = self._compute_reallocation_efficiency()

        # Reward is defines as a continuous value between -1 and 1. To do so we normalize between MAX and MIN observed, e.g. reward += ((x - MIN) - (MAX - x)) / (MAX - MIN)

        # First, we update observed MIN and MAX efficiencies (load and num_flows)

        self.update_extreme_values(
            load_efficiency, allocation_fairness, reallocation_efficiency
        )

        # Second, computing reward as a continuous value between (-1 and 1) for load efficiency and num flow efficiency. Avoid the first time where MAX and MIN are initialized to load efficiency
        reward_load_eff_factor = self.normalize_reward(
            load_efficiency, self.max_load_efficiency, self.min_load_efficiency
        )
        reward_alloc_fairness_factor = self.normalize_reward(
            allocation_fairness,
            self.max_allocation_fairness,
            self.min_allocation_fairness,
        )
        reward_re_factor = self.normalize_reward(
            reallocation_efficiency,
            self.max_reallocation_efficiency,
            self.min_reallocation_efficiency,
        )

        # Third, we aggregate the load efficiency and the num flow efficiency as a single reward that can go between -2 and 2

        # FIXME: Parameter to weight rewards, [-1, 1], 1 for max load efficiency, -1 for max fairness
        if self.weighted_reward:
            reward = (1 + self.weighted_reward) * reward_load_eff_factor + (
                1 - self.weighted_reward
            ) * reward_alloc_fairness_factor
        else:
            reward = reward_load_eff_factor + reward_alloc_fairness_factor

        # FIXME: Now reward goes between -3 and 3 -> Rethink the range of the reward, maybe 0 to 1?
        # reward = reward_load_eff_factor + reward_alloc_fairness_factor + reward_re_factor

        return reward, load_efficiency, allocation_fairness

    # This function does a brute force search to find the optimal allocation
    def _brute_force_optimal_flow_allocation(self):
        optimal_load_eff = 0
        optimal_numflow_perc = 0
        optimal_alloc_rank = 0
        optimal_allocation = []

        # Define all possible flow allocations
        lst = itertools.product(list(range(2 * self.k_paths)), repeat=self.n_flows)

        for action_list in lst:
            # before each allocation we set the networks empty and start filling. Otherwise we would only apply the differences which leads to error because flows in one network may not use the freed capacity from flows that move to the other network
            self.het_bhnet.deallocate_flows()

            alloc_list = list(map(lambda x: x // self.k_paths, action_list))

            for f in self.het_bhnet.flow_list:
                path = action_list[f.index] % self.k_paths

                f.assigned_path = (
                    f.sub6_paths[path]
                    if alloc_list[f.index] == SUB6
                    else f.mm_paths[path]
                )

                f.destination = f.assigned_path[-1]

            self.het_bhnet.allocate_flows(alloc_list)
            obs = self._next_observation()
            reward, load_efficiency, fairness = self._compute_reward()

            # Ranking this allocation looking at both load_efficiency and fairness

            if self.weighted_reward is None:
                alloc_rank = (1 - (100 - load_efficiency) / 100) + (
                    1 - (100 - fairness) / 100
                )
            elif self.weighted_reward == 1:
                alloc_rank = load_efficiency
            elif self.weighted_reward == -1:
                alloc_rank = fairness
            else:
                raise Exception("Value for weighted_reward not supported")

            if alloc_rank > optimal_alloc_rank:
                optimal_alloc_rank = alloc_rank
                optimal_load_eff = load_efficiency
                optimal_numflow_perc = fairness
                optimal_allocation = alloc_list

                effective_datarates = {
                    f.index: f.effective_data_rate for f in self.het_bhnet.flow_list
                }
                # checking if this allocation is already optimal and breaking loop
                if load_efficiency == 100 and fairness == 100:
                    break

        del lst

        return (
            optimal_load_eff,
            optimal_numflow_perc,
            optimal_allocation,
        )

    # This function does a arbitrary allocation defined by the action_list
    def _arbitrary_policy_allocation(self, action_list):
        # before each allocation we set the networks empty and start filling. Otherwise we would only apply the differences which leads to error because flows in one network may not use the freed capacity from flows that move to the other network

        self.het_bhnet.deallocate_flows()

        for f in self.het_bhnet.flow_list:
            path = random.randint(0, self.k_paths - 1)

            f.assigned_path = (
                f.sub6_paths[path] if action_list[f.index] == SUB6 else f.mm_paths[path]
            )

            f.destination = f.assigned_path[-1]

        self.het_bhnet.allocate_flows(action_list)
        obs = self._next_observation()
        reward, load_efficiency, fairness = self._compute_reward()

        return load_efficiency, fairness

    def _subset_sum_problem_half_heuristic(self):
        """
        Order flows in descending size and start allocating.
        For each flow try first the MMWAVE network and if it does not fit try the SUB6 network
        """

        load_efficiency = 0
        fairness = 0

        descending_flow_list: list[BackhaulFlow] = sorted(
            self.het_bhnet.flow_list,
            key=lambda f: f.data_rate,
            reverse=True,
        )

        # before each allocation we set the networks empty and start filling. Otherwise we would only apply the differences which leads to error because flows in one network may not use the freed capacity from flows that move to the other network
        self.het_bhnet.deallocate_flows()

        # We start allocating flows one by one.
        # For each flow we try to allocate in the mmWAVE and SUB6 networks, and we allocate to the network where a larger portion of the flow fits.
        # If the allocation is the same in the two networks we allocate in the mmWAVE
        for f in descending_flow_list:
            # Stores (network, path, reward)
            tmp_rewards: list[tuple] = []

            for path in range(self.k_paths):
                # 1- Tentative allocation in MMWAVE
                selected_path = f.mm_paths[path]
                f.assigned_path, f.path = selected_path, selected_path
                f.destination = f.assigned_path[-1]

                _ = self.het_bhnet.allocate_single_flow(f, MMWAVE)
                _ = self._next_observation()
                (
                    reward_mmwave,
                    load_efficiency_mmwave,
                    fairness_mmwave,
                ) = self._compute_reward()
                self.het_bhnet.mm_network.remove_flow(f)

                tmp_rewards.append((MMWAVE, selected_path.copy(), reward_mmwave))

                # 2- Tentative allocation in SUB6
                selected_path = f.sub6_paths[path]
                f.assigned_path, f.path = selected_path, selected_path
                f.destination = f.assigned_path[-1]

                _ = self.het_bhnet.allocate_single_flow(f, SUB6)
                _ = self._next_observation()
                (
                    reward_sub6,
                    load_efficiency_sub6,
                    fairness_sub6,
                ) = self._compute_reward()
                self.het_bhnet.sub6_network.remove_flow(f)

                tmp_rewards.append((SUB6, selected_path.copy(), reward_sub6))

            # 5 - Proceed with final allocation for this flow
            max_reward = max(tmp_rewards, key=lambda x: x[2])

            f.assigned_path = max_reward[1]
            f.destination = f.assigned_path[-1]
            f.path = max_reward[1]
            _ = self.het_bhnet.allocate_single_flow(f, max_reward[0])

            (
                _,
                load_efficiency,
                fairness,
            ) = self._compute_reward()

        # Compute the per-flow allocated data rate after completing the allocation
        self.het_bhnet.compute_allocated_per_flow_rates()
        obs = self._next_observation()
        reward, load_efficiency, fairness = self._compute_reward()

        sorted_flows: list = sorted(self.het_bhnet.flow_list, key=lambda f: f.index)
        allocation: list = [f.network_type for f in sorted_flows]

        effective_datarates = {
            f.index: f.effective_data_rate for f in self.het_bhnet.flow_list
        }

        # Return load_efficiency, fairness resulting from the allocation of the last flow
        return load_efficiency, fairness, allocation

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        # Retrieve the next observation after applying the action in this step
        obs = self._next_observation()

        self.current_step += 1

        # Compute the current reward. If efficiency increases positive reward, else negative
        reward, load_efficiency, fairness = self._compute_reward()

        # Consider "done" if no change in load efficiency and number of allocated flows
        # if (load_efficiency == self.last_load_efficiency and fairness == self.last_fairness) or self.current_step > self.MaxSteps:
        # if (load_efficiency == self.last_load_efficiency and fairness == self.last_fairness):
        if (
            load_efficiency == 100 and fairness == 100
        ) or self.current_step >= self.steps_game:
            done = True
            # FIXME: Once the current game ends, save allocation. Save last one, but ideally we should save the best?
            self.previous_allocation_vector = self.last_allocation_vector
        else:
            done = False

        # Store effective datarates
        self.het_bhnet.compute_allocated_per_flow_rates()
        effective_datarates = {
            f.index: f.effective_data_rate for f in self.het_bhnet.flow_list
        }

        # Updating last load efficiency and numflow percentage
        self.last_load_efficiency = load_efficiency
        self.last_fairness = fairness

        if self.current_step > self.steps_game:
            self.current_step = 0

        assigned_paths = [flow.path for flow in self.het_bhnet.flow_list]

        # Storing dict with additional info for debugging
        info = {
            "efficiency": load_efficiency,
            "fairness": fairness,
            "last_action": action,
            "last_allocation_vector": self.last_allocation_vector,
            "effective_datarates": effective_datarates,
            "assigned_paths": assigned_paths,
        }

        return obs, reward, done, info

    # method to use to reset network topology in the environment. Note that reset() by default only resets the Flow but the network topology is fixed, since the algorithm is trained for a given topology
    def reset_network_topology(
        self,
        network_size: int,
        root_nodes: int,
        max_child_nodes: int,
        mm_edge_prob: float,
        sub6_edge_prob: float,
    ):
        self.het_bhnet.remove_flows()  # Remove flows before updating G, or it will give problems when invoking reset()
        self.het_bhnet.reset_network_topology(
            network_size, root_nodes, max_child_nodes, mm_edge_prob, sub6_edge_prob
        )

    # Function to reset flow data rates in the environment
    def reset_flows_datarate(self, min_data_rate, max_data_rate):
        self.het_bhnet.reset_flows_datarate(min_data_rate, max_data_rate)

    def reset(self):
        # Step counter
        self.current_step = 0

        # Variable to store last observation
        self.last_obs = np.empty([2, self.root_nodes + 2 * self.n_flows + 1])
        self.last_load_efficiency = -1
        self.last_fairness = -1
        self.max_load_efficiency = -1
        self.min_load_efficiency = 200
        self.max_allocation_fairness = -1
        self.min_allocation_fairness = 200
        self.last_allocation_vector = np.empty(
            self.n_flows,
        )

        # Remove flows and generate new ones
        self.het_bhnet.remove_flows()
        self.het_bhnet.generate_flows()
        self._init_random_allocation()  # random initial allocation of self.last_allocation_vector

        # This initializes self.last_obs
        return self._next_observation()

    def render(self, mode="human", close=False):
        # Render the environment to the screen
        print(f"Step: {self.current_step}")
        print(f"Load efficiency: {self.last_load_efficiency}")
        print(f"NumFlow percentage: {self.last_fairness}")

    def update_extreme_values(
        self, load_efficiency, alloc_fairness, realloc_efficiency
    ):
        if load_efficiency > self.max_load_efficiency:
            self.max_load_efficiency = load_efficiency
        if load_efficiency < self.min_load_efficiency:
            self.min_load_efficiency = load_efficiency

        if alloc_fairness > self.max_allocation_fairness:
            self.max_allocation_fairness = alloc_fairness
        if alloc_fairness < self.min_allocation_fairness:
            self.min_allocation_fairness = alloc_fairness

        if realloc_efficiency > self.max_reallocation_efficiency:
            self.max_reallocation_efficiency = realloc_efficiency
        if realloc_efficiency < self.min_reallocation_efficiency:
            self.min_reallocation_efficiency = realloc_efficiency

    def normalize_reward(self, obs, max, min):
        if obs != max or obs != min:
            return ((obs - min) - (max - obs)) / (max - min)

        else:
            return 0
