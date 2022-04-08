# encoding: utf-8

from common.file_operations import Dataset


class Neighborhood:
    """
    The class of the neighborhood structure.
    The positive and negative neighborhoods of each node are stored here.
    Also, the adjacent cluster can be got when a partition is specified using this class.
    """

    def __init__(self, dataset: Dataset):
        """
        class initialization

        :param dataset: a reference to a Dataset instance
        """
        self._dataset = dataset
        self.neighborhood_structure = self.__collect_neighbor_info()

    def get_adjacent_cluster(self, node, solution) -> set:
        """
        dynamically get the adjacent clusters of a node under a specific partition

        :param node: number of node
        :param solution: a solution vector
        :return: the adjacent clusters
        """
        nbr = self.neighborhood_structure[node]['+'] | self.neighborhood_structure[node]['-']
        nbr_community = set([solution[i] for i in nbr])
        # when the element doesn't exist, discard would not raise KeyError
        nbr_community.discard(solution[node])

        return nbr_community

    def get_adjacent_cluster_of_cluster(self, cid, solution, partition):
        """
        dynamically get the adjacent clusters of a cluster under a specific partition

        :param cid: number of cluster
        :param solution: a solution vector
        :param partition: a partition
        :return: the adjacent clusters
        """

        adjacent_node = set()
        # TODO the algorithm about adjacent community needs to be optimized
        # the algorithm to find neighbor clusters is inefficient

        for node in partition[cid]:
            adjacent_node = adjacent_node | self.neighborhood_structure[node]['+'] | self.neighborhood_structure[node]['-']

        adjacent_community = set([solution[i] for i in adjacent_node])
        adjacent_community.discard(cid)

        return adjacent_community

    def __collect_neighbor_info(self) -> dict:
        """
        construct the neighborhood structure

        :return: a dictionary, {node_id: all neighborhoods (see self.__get_neighbors(node))}
        """
        neighbors = dict()
        for i in range(self._dataset.vnum):
            neighbor = self.__get_neighbors(i)
            if not neighbor['+'] and not neighbor['-']:
                continue
            else:
                neighbors[i] = neighbor

        return neighbors

    def __get_neighbors(self, node) -> dict:
        """
        find all the positive and negative neighborhoods of a node

        :param node: the number of node
        :return: a dictionary, {"+": set() of positive neighborhoods, "-": set() of negative neighborhoods}
        """

        # prejudgment
        nbr = self._dataset.data[node].keys()
        nbr_values = self._dataset.data[node].values()
        neg_num = (len(nbr_values) - sum(nbr_values)) / 2
        pos_num = len(nbr_values) - neg_num

        # find neighborhoods
        neg_nbr, pos_nbr = set(), set()
        if pos_num > neg_num:
            # negative first
            for nid, attr in self._dataset.data[node].items():
                if neg_num == 0:
                    break
                if attr == -1:
                    neg_nbr.add(nid)
                    neg_num -= 1
            pos_nbr = set(nbr) - set(neg_nbr)
            nbr_structure = {
                '+': pos_nbr,
                '-': neg_nbr
            }

        else:
            # positive
            for nid, attr in self._dataset.data[node].items():
                if pos_num == 0:
                    break
                if attr == 1:
                    pos_nbr.add(nid)
                    pos_num -= 1
            neg_nbr = set(nbr) - set(pos_nbr)
            nbr_structure = {
                '+': pos_nbr,
                '-': neg_nbr
            }

        return nbr_structure
