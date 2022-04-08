
import random as rd
from balance.objective_function import ObjectiveFunction
from balance.neighborhood import Neighborhood


class LocalSearch:
    """
    The class of local search procedure.
    Two methods of local search are defined here.
    """

    def __init__(self, obj_function: ObjectiveFunction, neighborhood: Neighborhood, node_available=None):
        """
        class initialization

        :param obj_function: an instance of objective function
        :param neighborhood: neighborhood structure
        """

        self.objective_function = obj_function
        self.neighborhood = neighborhood
        if node_available:
            self.node_list = self.__node_sort(node_available=node_available)
            self.abandoned = set(range(obj_function.vnum)) - set(self.node_list)

    def local_move(self):
        """
        each node is moved from its current cluster to neighbor clusters
        Note: solution and partition are changed in the iterations

        :return: None
        """

        improvement = True
        ct = 0
        obj = self.objective_function
        nbr = self.neighborhood

        while improvement:

            improvement = False
            ct += 1

            if ct >= 100:
                break
            for node in self.node_list:

                min_delta = 0
                candidate = -1

                for nbr_cluster in nbr.get_adjacent_cluster(node, obj.solution):
                    delta = obj.delta_caused_by_move(node, nbr_cluster, nbr.neighborhood_structure[node])
                    if delta < min_delta:
                        min_delta = delta
                        candidate = nbr_cluster

                if candidate != -1:
                    obj.move(node, candidate, min_delta)
                    improvement = True

    def community_merge(self):
        """
        each cluster is attempted to be merged with its neighborhood clusters
        Note: solution and partition are changed in the iterations

        :return: None
        """

        obj = self.objective_function
        nbr = self.neighborhood
        cluster_list = set(obj.partition.keys()) - self.abandoned
        tabu_list = set()
        # rd.shuffle(cluster_list)
        ct = 0

        for c1 in cluster_list:

            ct += 1
            # each cluster is merged only once
            if c1 in tabu_list:
                continue

            min_delta = 0
            candidate = -1
            for c2 in nbr.get_adjacent_cluster_of_cluster(c1, obj.solution, obj.partition):
                delta = obj.delta_caused_by_merge(c1, c2, nbr.neighborhood_structure)
                if delta < min_delta:
                    min_delta = delta
                    candidate = c2

            if candidate != -1:
                obj.merge(c1, candidate, min_delta)
                tabu_list.add(c1)
                tabu_list.add(candidate)

    def community_decompose(self, cluster):
        """
        some nodes are removed out of the cluster

        :return: None
        """
        pass

    def __node_sort(self, node_available=None, rank_criteria=''):
        """
        sort the nodes in the dataset according to

        :param rank_criteria: a custom criteria, default: random
        :return: a node list in order
        """

        if not node_available:
            node_available = list(range(self.objective_function.vnum))

        if rank_criteria == 'degree':

            nbr = self.neighborhood.neighborhood_structure
            nbr_len = [(i, len(nbr[i]['+']) + len(nbr[i]['-'])) for i in node_available]
            nbr_len.sort(key=lambda x: x[1], reverse=True)
            node_available = [x[0] for x in nbr_len]

        else:
            rd.shuffle(node_available)

        return node_available
