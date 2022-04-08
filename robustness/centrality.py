import os.path
import random as rd
import networkx as nx
import robustness.robustness_utils as utils

from common.file_operations import FileOperations


class Centrality:

    RANDOM = 1
    NODE_FRUSTRATION = 2
    BETWEENNESS = 3
    DEGREE = 4
    R_DEGREE = 5

    def __init__(self, centrality):
        self._centrality = centrality
        assert centrality in {Centrality.RANDOM, Centrality.NODE_FRUSTRATION, Centrality.BETWEENNESS, Centrality.DEGREE,
                              Centrality.R_DEGREE}

    def get_centrality_of_nodes(self, param) -> dict:

        if self._centrality == Centrality.RANDOM:
            return self._get_centrality_by_random(param.vnum)

        elif self._centrality == Centrality.NODE_FRUSTRATION:
            partition, nbr = param['partition'], param['neighbor_structure']
            return self._get_centrality_by_frustration(partition, nbr)

        elif self._centrality == Centrality.BETWEENNESS:
            return self._get_centrality_by_betweenness(param)

        elif self._centrality == Centrality.DEGREE:
            _, nbr = param['partition'], param['neighbor_structure']
            return self._get_centrality_by_degree(nbr)
        elif self._centrality == Centrality.R_DEGREE:
            _, nbr = param['partition'], param['neighbor_structure']
            return self._get_centrality_by_r_degree(nbr)

    def get_node_with_best_centrality(self, param, candidate):

        cents = self.get_centrality_of_nodes(param)
        cents = {i: cents[i] for i in candidate}

        if self._centrality == Centrality.RANDOM:
            return max(cents, key=cents.get)

        elif self._centrality == Centrality.NODE_FRUSTRATION:
            return min(cents, key=cents.get)

        elif self._centrality == Centrality.BETWEENNESS:
            return max(cents, key=cents.get)

        elif self._centrality == Centrality.DEGREE:
            return max(cents, key=cents.get)

        elif self._centrality == Centrality.R_DEGREE:
            return max(cents, key=cents.get)

    @staticmethod
    def _get_centrality_by_random(n):
        return {i: rd.random() for i in range(n)}

    # @staticmethod
    # def _get_centrality_by_repeatable_random(n):
    #     default_random_seq = r"random_seq.rb"
    #     if os.path.exists(default_random_seq):
    #         return FileOperations.load_array_from_file(default_random_seq)
    #     else:
    #

    @staticmethod
    def _get_centrality_by_frustration(partition, neighbor_structure):

        frustrations = {}

        for cid, community in partition.items():
            for single_node in community:
                if single_node not in neighbor_structure:
                    frustrations[single_node] = -1
                    continue
                node_nbr = neighbor_structure[single_node]
                frustrations[single_node] = len(node_nbr['+'] - community) + len(community & node_nbr['-'])

        return frustrations

    @staticmethod
    def _get_centrality_by_betweenness(dataset):

        graph = utils.build_graph_from_networkx(dataset)
        bcs = nx.centrality.betweenness_centrality(graph)
        return bcs

    @staticmethod
    def _get_centrality_by_degree(neighbor_structure):
        from collections import defaultdict
        degrees = defaultdict(int)
        for i in neighbor_structure.keys():
            degrees[i] = len(neighbor_structure[i]['+'])
        return degrees

    @staticmethod
    def _get_centrality_by_r_degree(neighbor_structure):
        from collections import defaultdict
        degrees = defaultdict(int)
        for i in neighbor_structure.keys():
            degrees[i] = len(neighbor_structure[i]['+']) - len(neighbor_structure[i]['-'])
        return degrees
