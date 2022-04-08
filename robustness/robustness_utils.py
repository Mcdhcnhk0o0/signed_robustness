import numpy as np
import random as rd
import networkx as nx
from common.file_operations import Dataset


def build_graph_from_networkx(dataset: Dataset):

    graph = nx.Graph()
    for v1 in dataset.data:
        for v2 in dataset.data[v1]:
            graph.add_edge(v1, v2)

    return graph


def build_graph_from_numpy(dataset: Dataset):

    n = dataset.vnum
    arr = np.zeros(shape=(n, n))
    data = dataset.data

    for v1 in data:
        for v2 in data[v1]:
            arr[v1, v2] = data[v1][v2]

    return arr


class NodeSort:

    def __init__(self, node_list: list):
        self.node_list = node_list

    def random_sort(self):
        rd.shuffle(self.node_list)
        return self.node_list

    def node_frustration_sort(self, partition: dict, neighbor_structure: dict):
        """

        """
        frustrations = {}

        for cid, community in partition.items():
            # calculate the frustration for each cluster
            for single_node in community:
                if single_node not in neighbor_structure:
                    frustrations[single_node] = -1
                    continue
                node_nbr = neighbor_structure[single_node]
                frustrations[single_node] = len(node_nbr['+'] - community) + len(community & node_nbr['-'])

        self.node_list.sort(key=frustrations.get, reverse=True)
        return self.node_list

    def betweenness_centrality_sort(self, dataset):
        """

        """
        graph = build_graph_from_networkx(dataset)
        bcs = nx.centrality.betweenness_centrality(graph)
        self.node_list.sort(key=bcs.get, reverse=True)
        return self.node_list

    def degree_centrality_sort(self, neighbor_structure):
        """

        """
        degrees = {}
        for node in self.node_list:
            if node not in neighbor_structure:
                degrees[node] = -1
                continue
            degrees[node] = len(neighbor_structure[node]["+"]) + len(neighbor_structure[node]["-"])
        self.node_list.sort(key=degrees.get, reverse=True)
        return self.node_list

