import networkx as nx
import numpy as np
import random

from bh_network.bhflow import BackhaulFlow
from bh_network.dag import create_adg

UNDEFINED = -1
SUB6 = 0
MMWAVE = 1


class BackhaulNetwork:
    def __init__(
        self,
        network_size: int,
        root_nodes: int,
        max_child_nodes: int,
        edge_prob: float,
        edge_capacity: tuple[int, int],
        topology_seed: int,
        network_type,
    ) -> None:
        self.network_size = network_size

        self.root_nodes = root_nodes
        self.max_child_nodes = max_child_nodes
        self.edge_prob = edge_prob
        self.edge_capacity = edge_capacity
        self.topology_seed = topology_seed
        self.network_type = network_type
        self.n_allocated_flows = 0

        self.init_graph()

    def init_graph(self) -> None:
        self.G, _, _ = create_adg(
            self.network_size,
            self.root_nodes,
            self.max_child_nodes,
            self.edge_prob,
            seed=self.topology_seed,
        )

        for i in range(self.network_size):
            if i < self.root_nodes:
                self.G.nodes[i]["access"] = False
                self.G.nodes[i]["exit"] = True
                self.G.nodes[i]["aggregate_load"] = 0
            else:
                self.G.nodes[i]["access"] = True
                self.G.nodes[i]["exit"] = False
                self.G.nodes[i]["aggregate_load"] = 0

        # Setting the edge attributes
        for u, v in self.G.edges():
            self.G[u][v]["max_load"] = round(np.random.uniform(*self.edge_capacity))
            self.G[u][v]["current_load"] = 0
            self.G[u][v][
                "Flows"
            ] = []  # contains a list with the indexes of the flows traversing this link

    def remove_n_links(self, n: int) -> None:
        for _ in range(n):
            src = list(map(lambda x: x[0], list(self.G.edges())))
            valid_nodes = [s for s in src if src.count(s) > 1]

            link_to_remove = random.choice(list(self.G.edges(nbunch=valid_nodes)))
            self.G.remove_edge(*link_to_remove)

    def reset_network_topology(
        self, network_size, root_nodes, max_child_nodes, edge_prob
    ) -> None:
        self.network_size = network_size
        self.root_nodes = root_nodes
        self.max_child_nodes = max_child_nodes
        self.edge_prob = edge_prob
        self.topology_seed += 1

        self.init_graph()

    def path_exists(self, flow: BackhaulFlow) -> bool:
        return nx.has_path(self.G, flow.source, flow.destination)

    def find_paths(self, flow: BackhaulFlow) -> list:
        return nx.all_simple_paths(
            self.G, flow.source, [dst for dst in range(self.root_nodes)]
        )

    def allocate_flow(self, flow: BackhaulFlow) -> None:
        if flow.path is None:
            return None
        path = flow.path

        flow.network_type = self.network_type
        self.n_allocated_flows = self.n_allocated_flows + 1

        # Updating the current_load attribute
        # Notice that we start allocating load from the flow to the edges in src->dst order. Then, to subsequent edges we can only add as much load as it fits in the previous ones, e.g. if in the first edge the added load is 0, it must be 0 in the subsequent links too
        min_flow_load_in_previous_links = 100000  # initialize it to a large number

        if flow.source == flow.destination:
            self.G.nodes[flow.source]["aggregate_load"] += flow.data_rate

        for i in range(len(path) - 1):
            n_src = path[i]
            n_dst = path[i + 1]

            # Adding this flow to the set of flows traversing this link
            present_flows = [f.source for f in self.G[n_src][n_dst]["Flows"]]
            if flow.source not in present_flows:
                self.G[n_src][n_dst]["Flows"].append(flow)

            # Updating current_load
            current_load = self.G[n_src][n_dst]["current_load"]
            max_load_flow = min(
                current_load + flow.data_rate, self.G[n_src][n_dst]["max_load"]
            )
            added_load = min(
                max_load_flow - current_load, min_flow_load_in_previous_links
            )  # if in a previous link there was a bottleneck we are constrained

            # Update the actual load in the link
            self.G[n_src][n_dst]["current_load"] += added_load
            if added_load < min_flow_load_in_previous_links:
                # Update the bottleneck load for subsequent links
                min_flow_load_in_previous_links = added_load
            # Storing the traversed edges in the flow object

            key = str(n_src) + "-" + str(n_dst)
            flow.traversed_edges[key] = added_load
            # Updating the node attributes
            if n_src == flow.source:
                self.G.nodes[n_src]["aggregate_load"] += flow.data_rate
                # Offered Load
                flow.traversed_nodes[n_src] = flow.data_rate
            if n_dst == flow.destination:
                self.G.nodes[n_dst]["aggregate_load"] += added_load
                flow.traversed_nodes[n_dst] = added_load

    def remove_flow(self, flow: BackhaulFlow):
        if flow.path is None:
            return None

        # Special case where flow is assigned new path but we want to remove it from the previous one
        if flow.path != flow.assigned_path:
            destination = flow.path[-1]
        else:
            destination = flow.destination

        # Removing the load this flow has introduced in the different edges it traversed
        for i in range(len(flow.path) - 1):
            n_src = flow.path[i]
            n_dst = flow.path[i + 1]
            key = str(n_src) + "-" + str(n_dst)

            added_load = flow.traversed_edges[key]

            if n_dst == destination:
                last_node_load = added_load
            try:
                self.G[n_src][n_dst]["current_load"] -= added_load
                flows_present = self.G[n_src][n_dst]["Flows"]
            except KeyError:
                pass

            try:
                for f in self.G[n_src][n_dst]["Flows"]:
                    if f.source == flow.source:
                        self.G[n_src][n_dst]["Flows"].remove(f)
            except KeyError:
                pass

        # Removing the load the flow introduced in the access and exit nodes
        self.G.nodes[flow.source]["aggregate_load"] -= flow.data_rate
        self.G.nodes[destination]["aggregate_load"] -= last_node_load

        flow.clear_state()
        self.n_allocated_flows -= 1
