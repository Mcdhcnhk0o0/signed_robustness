from common.file_operations import Dataset

import abc
from balance import balance_utils as utils


class ObjectiveFunction:
    """
    The class of the objective function.
    Calculation methods of the objective function are defined here with a solution vector and a partition.
    The specific objective function should inherit this class and implement the abstract methods.
    """

    def __init__(self, dataset: Dataset, init_solution=None):
        """
        class initialization

        :param dataset: a reference to a Dataset instance
        :param init_solution: optional, the initial solution vector
        """

        self._dataset = dataset
        self.obj_value = dataset.enum

        if init_solution is None:
            self.solution, self.partition = utils.default_initialization(self._dataset.vnum)
        else:
            self.solution = init_solution
            self.partition = utils.solution2partition(init_solution)
        self.update_objective_function()

    def set_solution(self, solution):
        """
        set the solution vector and partition

        :param solution: the solution vector
        :return: None
        """
        self.solution = solution
        self.partition = utils.solution2partition(solution)

    def update_objective_function(self):
        """
        active update method of the objective function value

        :return: current objective function value
        """
        self.obj_value = self.objective_function()
        return self.obj_value

    @abc.abstractmethod
    def objective_function(self):
        return self.obj_value

    def objective_function_v2(self, neighborhood, partition=None):
        return self.obj_value

    @abc.abstractmethod
    def move(self, node, destination, delta):
        pass

    @abc.abstractmethod
    def delta_caused_by_move(self, node, destination, neighborhood):
        pass

    @abc.abstractmethod
    def merge(self, c1, c2, delta):
        pass

    @abc.abstractmethod
    def delta_caused_by_merge(self, c1, c2, neighborhood):
        pass

    def decompose(self, node, delta):
        pass

    def delta_caused_by_decompose(self, node, neighborhood):
        pass

    @property
    def vnum(self):
        # quick access of the number of nodes in the dataset
        return self._dataset.vnum
