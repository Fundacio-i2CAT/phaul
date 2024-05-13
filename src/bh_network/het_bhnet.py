import random
from collections import defaultdict

from bh_network.bhflow import BackhaulFlow
from bh_network.bhnet import BackhaulNetwork

UNDEFINED = -1
SUB6 = 0
MMWAVE = 1


class HetBackhaulNetwork:
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
        k_paths: int,
        routing_algorithm=None,
    ):
        self.network_size = network_size
        self.root_nodes = root_nodes
        self.max_child_nodes = max_child_nodes
        self.mm_edge_prob = mm_edge_prob
        self.sub6_edge_prob = sub6_edge_prob
        self.mm_capacity = mm_capacity
        self.sub6_capacity = sub6_capacity
        self.min_flow_datarate = min_flow_datarate
        self.max_flow_datarate = max_flow_datarate
        self.node_active_prob = node_active_prob
        self.k_paths = k_paths
        self.routing_algorithm = routing_algorithm

        self.flow_list: list[BackhaulFlow] = []

        topology_seed = random.randint(0, 100)

        self.mm_network = BackhaulNetwork(
            self.network_size,
            self.root_nodes,
            self.max_child_nodes,
            self.mm_edge_prob,
            self.mm_capacity,
            topology_seed,
            MMWAVE,
        )

        self.sub6_network = BackhaulNetwork(
            self.network_size,
            self.root_nodes,
            self.max_child_nodes,
            self.sub6_edge_prob,
            self.mm_capacity,
            topology_seed,
            SUB6,
        )

    # This method is used if we want to reset the topology of the underlying network. We can pass as parameter a different edge probability
    def reset_network_topology(
        self,
        network_size: int,
        root_nodes: int,
        max_child_nodes: int,
        mm_edge_prob: float,
        sub6_edge_prob: float,
    ):
        self.mm_network.reset_network_topology(
            network_size, root_nodes, max_child_nodes, mm_edge_prob
        )
        self.sub6_network.reset_network_topology(
            network_size, root_nodes, max_child_nodes, sub6_edge_prob
        )

    def generate_flows(self):
        if self.routing_algorithm:
            mm_paths = defaultdict(int)
            sub6_paths = defaultdict(int)
            paths_in_use = mm_paths, sub6_paths
        else:
            paths_in_use = None

        flow_index: int = 0

        for src in range(self.root_nodes, self.network_size):
            f = BackhaulFlow(
                flow_index,
                src,
                self.root_nodes,
                self.min_flow_datarate,
                self.max_flow_datarate,
                self.node_active_prob,
                self.mm_network,
                self.sub6_network,
                self.k_paths,
                self.routing_algorithm,
                paths_in_use,
            )

            flow_index += 1
            self.flow_list.append(f)

            paths_in_use = f.get_paths_in_use()

    # This function removes the flows allocated in the networks without clearing the flow list, i.e. the Flows with src-dst and data rate are maintained
    def deallocate_flows(self):
        for f in self.flow_list:
            if f.network_type == SUB6:
                self.sub6_network.remove_flow(f)
            elif f.network_type == MMWAVE:
                self.mm_network.remove_flow(f)
            elif f.network_type == UNDEFINED:
                f.clear_state()

    # This function removes the flows allocated in the networks and clears the flow list
    def remove_flows(self):
        for f in self.flow_list:
            if f.network_type == SUB6:
                self.sub6_network.remove_flow(f)
            if f.network_type == MMWAVE:
                self.mm_network.remove_flow(f)
            if f.network_type == UNDEFINED:
                f.clear_state()

        self.flow_list = []

    # this fuunction resets the data rate for the flows in the network
    def reset_flows_datarate(self, min_data_rate, max_data_rate):
        self.min_data_rate = min_data_rate
        self.max_data_rate = max_data_rate
        for f in self.flow_list:
            f.reset_datarate(min_data_rate, max_data_rate)

    # This function allocates a single flow in the indicated network
    def allocate_single_flow(self, f: BackhaulFlow, network_type):
        # Allocate flow in mmWave network, if not already there
        if network_type == MMWAVE and f.network_type != MMWAVE:
            # Check if there will be a path for this flow in the MMWAVE network
            PathExists = self.mm_network.path_exists(f)
            if PathExists:
                # Check if the Flow is currently in the Sub6 network and remove it from there
                if f.network_type == SUB6:
                    self.sub6_network.remove_flow(f)

                # Allocate the Flow in the mmWave network
                PathExists = self.mm_network.allocate_flow(f)
            else:
                #    print('\nPath in MMWAVE network does not exist! Leaving the flow unmodified')
                # Check if the Flow is currently in the Sub6 network and remove it from there. Flow becomes in UNDEFINED state
                if f.network_type == SUB6:
                    self.sub6_network.remove_flow(f)

        # Allocate flow in Sub6 network, if not already there
        if network_type == SUB6 and f.network_type != SUB6:
            # Check if there will be a path for this flow in the MMWAVE network
            PathExists = self.sub6_network.path_exists(f)
            if PathExists:
                # Check if the Flow is currently in the MMWave network and remove it from there
                if f.network_type == MMWAVE:
                    self.mm_network.remove_flow(f)

                # Allocate the Flow in the Sub6 network
                PathExists = self.sub6_network.allocate_flow(f)
            else:
                #    print('\nPath in SUB6 network does not exist! Leaving the flow unmodified')
                # Check if the Flow is currently in the MMWave network and remove it from there. Flow becomes in UNDEFINED state
                if f.network_type == MMWAVE:
                    self.mm_network.remove_flow(f)

        return PathExists

    def _compute_per_flow_rate(self, bhnet: BackhaulNetwork):
        tmp = bhnet.G.edges()
        for u, v in bhnet.G.edges():
            # FIXME: flows_this_link has repeated flows
            flows_this_link: list[BackhaulFlow] = bhnet.G[u][v]["Flows"]
            link_capacity: int = bhnet.G[u][v]["max_load"]

            if not flows_this_link:
                continue

            load_this_link = bhnet.G[u][v]["current_load"]
            total_link_demand = 0
            for f in flows_this_link:
                total_link_demand = total_link_demand + f.data_rate

            # Case where the total required load is above capacity of the link
            if total_link_demand > link_capacity:
                # This is a bottleneck link. Compute fair share using waterfilling (https://www.comm.utoronto.ca/~jorg/teaching/ece1545/schedslides/bw-allocation.pdf)
                fair_share = 0
                n = 0
                complete = False
                while complete == False:
                    n = n + 1  # What is n used for?
                    list_under = []
                    list_over = []
                    for f in flows_this_link:
                        if f.data_rate <= fair_share:
                            list_under.append(f)
                        if f.data_rate > fair_share:
                            list_over.append(f)
                    sum_data_rate_under = 0
                    for f in list_under:
                        sum_data_rate_under = sum_data_rate_under + f.data_rate
                    f_new = (link_capacity - sum_data_rate_under) / len(list_over)
                    if f_new == fair_share:
                        complete = True
                    else:
                        fair_share = f_new

                # Compute data rate allocated to each flow in this link
                for F in flows_this_link:
                    rate_flow_this_link = min(F.data_rate, fair_share)
                    # update the minimum data rate experienced by a flow across all links it traversses
                    if (rate_flow_this_link < F.effective_data_rate) or (
                        F.effective_data_rate == -1
                    ):
                        F.effective_data_rate = rate_flow_this_link

    # This function computes the effective data rate that each flow is receiving after being allocated in the network
    def compute_allocated_per_flow_rates(self):
        self._compute_per_flow_rate(self.sub6_network)
        self._compute_per_flow_rate(self.mm_network)

    # This function receives a vector indicating the network where each flows needs to be allocated and performs the allocation if possible
    def allocate_flows(self, allocation: list):
        for i, network in enumerate(allocation):
            f = self.flow_list[i]

            if f.network_type == UNDEFINED:
                f.path = f.assigned_path

                if network == SUB6:
                    if f.path is None:
                        f.path = f.sub6_paths[0]
                    self.sub6_network.allocate_flow(f)
                else:
                    if f.path is None:
                        f.path = f.mm_paths[0]
                    self.mm_network.allocate_flow(f)

            elif network == f.network_type and f.assigned_path != f.path:
                f.path = f.assigned_path

                if network == SUB6:
                    self.sub6_network.remove_flow(f)
                    self.sub6_network.allocate_flow(f)
                else:
                    self.mm_network.remove_flow(f)
                    self.sub6_network.allocate_flow(f)

            elif network != f.network_type:
                f.path = f.assigned_path

                if network == SUB6:
                    self.mm_network.remove_flow(f)
                    self.sub6_network.allocate_flow(f)
                else:
                    self.sub6_network.remove_flow(f)
                    self.sub6_network.allocate_flow(f)

        # After all flows are allocated we compute the effective data rate for each flow.
        self.compute_allocated_per_flow_rates()

    # This function performs a dummy allocation with half the flows in one network and half the flows in the other
    def dummy_flow_allocation(self):
        # Random allocation vector
        TargetFlowNetworkTypeVector = []
        for i in range(len(self.FlowList)):
            if i % 2 == 0:
                TargetFlowNetworkTypeVector.append(MMWAVE)
        else:
            TargetFlowNetworkTypeVector.append(SUB6)

        self.allocate_flows(TargetFlowNetworkTypeVector)

    def recover_network_state(self) -> tuple[list, list, list]:
        allocation: list = [f.network_type for f in self.flow_list]

        mm_state: list = []
        sub6_state: list = []

        # Append ingress load
        for flow in self.flow_list:
            if flow.network_type == SUB6:
                sub6_state.append(flow.data_rate)
                mm_state.append(0)
            elif flow.network_type == MMWAVE:
                mm_state.append(flow.data_rate)
                sub6_state.append(0)
            elif flow.network_type == UNDEFINED:
                mm_state.append(0)
                sub6_state.append(0)

        # Append egress load
        for i in range(self.root_nodes):
            mm_state.append(self.mm_network.G.nodes[i]["aggregate_load"])
            sub6_state.append(self.sub6_network.G.nodes[i]["aggregate_load"])
            """print(
                f"Aggregate node load: {self.mm_network.G.nodes[i]['aggregate_load']}",
                flush=True,
            ) """

        # Add number of allocated flows in each network
        mm_state.append(self.mm_network.n_allocated_flows)
        sub6_state.append(self.sub6_network.n_allocated_flows)

        return allocation, mm_state, sub6_state
