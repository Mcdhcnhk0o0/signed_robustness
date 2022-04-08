from common.file_operations import Dataset
from balance.local_search import LocalSearch

from balance import balance_utils as utils


class Initialization:
    """
    The class of initialization.
    Some initialization methods are defined here.
    """

    def __init__(self, dataset: Dataset, neighborhood):
        self._dataset = dataset
        self.neighborhood = neighborhood

    def default_initialization(self):
        """
        all the nodes are in different clusters

        :return: solution: list or dict, partition: dict(cluster_id: set())
        """

        return utils.default_initialization(self._dataset.vnum)

    def greedy_initialization(self, obj_function):
        """
        construct a partition using a greedy function

        :param obj_function: the greedy function
        :return: solution: list or dict, partition: dict(cluster_id: set())
        """

        _, node_available = utils.check_node_list(obj_function.vnum, self.neighborhood.neighborhood_structure)
        method = LocalSearch(obj_function=obj_function, neighborhood=self.neighborhood, node_available=node_available)
        method.local_move()
        method.objective_function.update_objective_function()
        return method.objective_function.solution, method.objective_function.partition

    def multi_start_greedy_initialization(self):
        """
        greedy initialization is cheap, multi start initialization for better performance

        :return: list or dict, partition: dict(cluster_id: set())
        """

        best_solution, best_partition = None, None
        best_value = 2 ** 64
        # This method was removed because of some bugs.
        return best_solution, best_partition
