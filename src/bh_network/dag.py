import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def create_adg(
    total_nodes: int, root_nodes: int, max_child_nodes: int, edge_prob: float, seed=0
) -> tuple[nx.DiGraph, list, list]:
    rng = np.random.RandomState(seed)

    G = nx.DiGraph()
    layers: list[list] = []
    hierarchy: list[list] = []

    # Initialize first layer containing 'root_nodes' nodes.
    layers.append(list(range(root_nodes)))
    G.add_nodes_from(layers[0])
    node_count = root_nodes

    # Add 'total_nodes' amount of nodes following a multi-layer parent/child strategy.
    while node_count < total_nodes:
        current_layer = len(layers)
        layers.append([])

        for parent_node in layers[current_layer - 1]:
            hierarchy.append([])
            try:
                child_nodes = rng.randint(
                    1, min(max_child_nodes, total_nodes - node_count)
                )
            except ValueError:
                child_nodes = 1

            for child in range(node_count, node_count + child_nodes):
                if node_count == total_nodes:
                    break

                layers[current_layer].append(node_count)
                G.add_node(child)
                hierarchy[parent_node].append(child)
                G.add_edge(child, parent_node)
                node_count += 1

    # Once all parent/child nodes are connected, add random connection between parent's neighbours and childs.
    for n_layer, layer in enumerate(layers):
        if n_layer == len(layers) - 1:
            break

        try:
            for parent_pos, parent_node in enumerate(layer):
                if parent_pos == 0:
                    for child_node in hierarchy[parent_node + 1]:
                        if np.random.random() < edge_prob:
                            G.add_edge(child_node, parent_node)

                elif parent_pos == len(layer):
                    for child_node in hierarchy[parent_node - 1]:
                        if np.random.random() < edge_prob:
                            G.add_edge(child_node, parent_node)

                else:
                    for child_node in hierarchy[parent_node + 1]:
                        if np.random.random() < edge_prob:
                            G.add_edge(child_node, parent_node)

                    for child_node in hierarchy[parent_node - 1]:
                        if np.random.random() < edge_prob:
                            G.add_edge(child_node, parent_node)
        except IndexError:
            continue

    return G, layers, hierarchy


def print_graph(graph: nx.DiGraph, save_path: str) -> None:
    pos = nx.shell_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=1000,
        font_size=10,
        node_color="skyblue",
        arrows=False,
    )
    plt.savefig(f"{save_path}.png")


def main():
    graph, layers, hierarchy = create_adg(20, 3, 3, 0.2, seed=25)
    pos = nx.shell_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=1000,
        font_size=10,
        node_color="skyblue",
        arrows=False,
    )
    plt.savefig("graph.png")
    print(layers)
    print(hierarchy)
    print(graph.edges())


if __name__ == "__main__":
    main()
