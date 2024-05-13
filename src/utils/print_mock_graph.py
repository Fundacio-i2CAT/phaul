from collections import defaultdict
from random import randint
from re import sub

import BackhaulNetwork.BackhaulNetwork as bn
import matplotlib.pyplot as plt
import networkx as nx


def mock_network():
    het_network = bn.HetBackhaulNetwork(
        10, 0.3, 3, 2, 1000, 300, 50, 300, 15, fixed_topology=True
    )
    het_network.generate_flows()

    mock_allocation = [randint(0, 1) for i in range(len(het_network.FlowList))]
    het_network.allocate_flows(mock_allocation)

    mm_n = het_network.mmWaveNetwork.G
    sub6_n = het_network.Sub6Network.G

    return het_network, mm_n, sub6_n, mock_allocation


def get_traversed_edges(flow):
    path = flow.path
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return edges


def print_backhaul(het, network, node_color, alloc, save_path):
    pos = nx.kamada_kawai_layout(network)

    # Print nodes
    acces_nodes = [
        i for i in range(len(network.nodes)) if network.nodes[i]["access"] is True
    ]
    root_nodes = [
        i for i in range(len(network.nodes)) if network.nodes[i]["exit"] is True
    ]
    remaining_nodes = [
        i
        for i in range(len(network.nodes))
        if network.nodes[i]["access"] is False and network.nodes[i]["exit"] is False
    ]

    nx.draw_networkx_nodes(
        network, nodelist=acces_nodes, node_color="b", pos=pos, label="Access"
    )
    nx.draw_networkx_nodes(
        network, nodelist=root_nodes, node_color="k", pos=pos, label="Root"
    )
    nx.draw_networkx_nodes(
        network, nodelist=remaining_nodes, node_color=node_color, pos=pos
    )
    nx.draw_networkx_labels(network, pos, font_color="w")

    # Print edges
    nx.draw_networkx_edges(network, pos)

    flows = [f for f in het.FlowList if f.index in alloc]
    edge_labels = {(u, v): 0 for u, v in network.edges()}
    for flow in flows:
        edges = get_traversed_edges(flow)
        for edge in edges:
            try:
                edge_labels[edge] += flow.DataRate
            except KeyError:
                edge = (edge[1], edge[0])
                edge_labels[edge] += flow.DataRate

        label = (
            "F"
            + str(flow.index)
            + " ("
            + str(flow.src)
            + ","
            + str(flow.dst)
            + ") "
            + str(flow.DataRate)
        )
        nx.draw_networkx_edges(
            network,
            pos,
            edgelist=edges,
            edge_color=(
                randint(0, 255) / 255,
                randint(0, 255) / 255,
                randint(0, 255) / 255,
            ),
            label=label,
        )

    # Print edge labels
    # edge_labels = {(u, v): load for u,v,load in network.edges().data('current_load')}
    nx.draw_networkx_edge_labels(network, pos, edge_labels=edge_labels)

    plt.legend()

    # nx.draw_networkx_edges(network, pos, edgelist=alloc, edge_color="g")
    nx.draw_networkx_edge_labels(network, pos, edge_labels=edge_labels)

    plt.savefig(save_path, dpi=500)
    plt.close()


def print_join_network(het, alloc, effective_datarates, save_path):
    joined_net = nx.disjoint_union(het.mmWaveNetwork.G, het.Sub6Network.G)

    # Create real access nodes
    true_acces_nodes = [len(joined_net.nodes) + i for i in range(het.NumAccessNodes)]
    for i in range(het.NumAccessNodes):
        joined_net.add_node(len(joined_net.nodes), access=True)
        joined_net.add_edge(len(joined_net.nodes) - 1, i)
        joined_net.add_edge(len(joined_net.nodes) - 1, i + het.NetworkSize)

    # Crete real root nodes
    true_root_nodes = [len(joined_net.nodes) + i for i in range(het.NumRootNodes)]
    for i in range(het.NumRootNodes):
        joined_net.add_node(len(joined_net.nodes), root=True)
        joined_net.add_edge(
            len(joined_net.nodes) - 1, het.NetworkSize - het.NumRootNodes + i
        )
        joined_net.add_edge(
            len(joined_net.nodes) - 1, 2 * het.NetworkSize - het.NumRootNodes + i
        )

    mm_nodes = list(joined_net.nodes)[: het.NetworkSize]
    sub6_nodes = list(joined_net.nodes)[het.NetworkSize : 2 * het.NetworkSize]

    # pos = nx.kamada_kawai_layout(joined_net)
    # pos = nx.planar_layout(joined_net)
    pos = nx.spring_layout(joined_net)
    plt.figure(figsize=(15, 15))

    nx.draw_networkx_nodes(
        joined_net, nodelist=true_acces_nodes, node_color="b", pos=pos, label="Access"
    )
    nx.draw_networkx_nodes(
        joined_net, nodelist=true_root_nodes, node_color="k", pos=pos, label="Root"
    )

    nx.draw_networkx_nodes(
        joined_net, nodelist=mm_nodes, node_color="tab:blue", pos=pos, label="Mm"
    )
    nx.draw_networkx_nodes(
        joined_net, nodelist=sub6_nodes, node_color="r", pos=pos, label="Sub6"
    )

    nx.draw_networkx_labels(joined_net, pos, font_color="w")

    # Print edges
    nx.draw_networkx_edges(joined_net, pos)

    mm_alloc = [i for i in range(len(alloc)) if alloc[i] == 1]
    sub6_alloc = [i for i in range(len(alloc)) if alloc[i] == 0]

    flows = [f for f in het.FlowList if f.index in alloc]
    mm_flows = [f for f in het.FlowList if f.index in mm_alloc]
    sub6_flows = [f for f in het.FlowList if f.index in sub6_alloc]

    edge_labels = {(u, v): 0 for u, v in joined_net.edges()}

    # Edges for MM network
    for flow in mm_flows:
        edges = get_traversed_edges(flow)
        for edge in edges:
            try:
                edge_labels[edge] += flow.DataRate
            except KeyError:
                pass
                # edge = (edge[1], edge[0])
                # edge_labels[edge] += flow.DataRate

            # if edge_labels[edge] > het.mmWaveMaxEdgeLoadMbps:
            # edge_labels[edge] = het.mmWaveMaxEdgeLoadMbps

        label = (
            "F"
            + str(flow.index)
            + " ("
            + str(flow.src)
            + ","
            + str(flow.dst)
            + ") "
            + str(flow.DataRate)
            + "-"
            + str(effective_datarates[flow.index])
        )
        nx.draw_networkx_edges(
            joined_net,
            pos,
            edgelist=edges,
            edge_color=(
                randint(0, 255) / 255,
                randint(0, 255) / 255,
                randint(0, 255) / 255,
            ),
            label=label,
        )

        edge_labels[(flow.src, flow.src + 2 * het.NetworkSize)] += flow.DataRate
        edge_labels[
            (
                flow.dst,
                flow.dst + het.NetworkSize + het.NumAccessNodes + het.NumRootNodes,
            )
        ] += effective_datarates[flow.index]

    # Edges for Sub6 network
    for flow in sub6_flows:
        edges = get_traversed_edges(flow)
        for edge in edges:
            edge = (edge[0] + het.NetworkSize, edge[1] + het.NetworkSize)
            try:
                edge_labels[edge] += flow.DataRate
            except KeyError:
                pass
                # edge = (edge[1], edge[0])
                # edge_labels[edge] += flow.DataRate

            # if edge_labels[edge] > het.Sub6MaxEdgeLoadMbps:
            # edge_labels[edge] = het.Sub6MaxEdgeLoadMbps

        label = (
            "F"
            + str(flow.index)
            + " ("
            + str(flow.src + het.NetworkSize)
            + ","
            + str(flow.dst + het.NetworkSize)
            + ") "
            + str(flow.DataRate)
            + "-"
            + str(effective_datarates[flow.index])
        )
        nx.draw_networkx_edges(
            joined_net,
            pos,
            edgelist=edges,
            edge_color=(
                randint(0, 255) / 255,
                randint(0, 255) / 255,
                randint(0, 255) / 255,
            ),
            label=label,
        )

        edge_labels[
            (flow.src + het.NetworkSize, flow.src + 2 * het.NetworkSize)
        ] += flow.DataRate
        edge_labels[
            (
                flow.dst + het.NetworkSize,
                flow.dst + het.NetworkSize + het.NumAccessNodes + het.NumRootNodes,
            )
        ] += effective_datarates[flow.index]

    # Print edge labels

    nx.draw_networkx_edge_labels(joined_net, pos, edge_labels=edge_labels)

    plt.legend()

    # nx.draw_networkx_edges(network, pos, edgelist=alloc, edge_color="g")
    # nx.draw_networkx_edge_labels(network, pos, edge_labels=edge_labels)

    plt.savefig(save_path, dpi=250)
    plt.close()


if __name__ == "__main__":
    het_network, mm_n, sub6_n, mock_allocation = mock_network()

    # sub6_alloc = [i for i in range(len(mock_allocation)) if mock_allocation[i] == 0]
    # mm_alloc = [i for i in range(len(mock_allocation)) if mock_allocation[i] == 1]

    # print_backhaul(het_network, mm_n, "tab:blue", mm_alloc, "topology_visualization/mm.png")
    # print_backhaul(het_network, sub6_n, "r", sub6_alloc, "topology_visualization/sub6.png")

    effective_datarates: dict = {
        i: het_network.FlowList[i].DataRate - 20
        for i in range(len(het_network.FlowList))
    }

    print_join_network(
        het_network,
        mock_allocation,
        effective_datarates,
        "topology_visualization/joined_eff.png",
    )
