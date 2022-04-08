# encoding: utf-8
from balance import *
from common.file_operations import Dataset
from common.file_operations import FileOperations

import time
import math
from balance import balance_utils as utils
import random as rd


class IteratedGreedy:
    """
    IG：
        1. s_z <- generate an initial solution;
        2. s_* <- apply local search to s_z;
        3. while termination condition is not satisfied do:
            s_p <- apply destruction to s_*
            s_' <- apply construction to s_p
            s_' <- apply local search to s_'
            if acceptance criterion is satisfied then
                s_* <- s_'
            end
        4. return s_*
    """

    def __init__(self, dataset: Dataset, beta=0.3):
        """
        class initialization

        :param dataset: a given dataset
        """
        self._dataset = dataset
        self.neighborhood = Neighborhood(dataset=dataset)
        self.objective_function = Frustration(dataset)
        self.node_available = self.__pretreatment()
        self.local_search = LocalSearch(self.objective_function, self.neighborhood, self.node_available)
        self.beta = beta
        self.T = 0
        self.ct = 1

    def initialization(self, output=True):
        init = Initialization(self._dataset, self.neighborhood)
        solution, partition = init.greedy_initialization(self.objective_function)
        self.objective_function.set_solution(solution)
        self.objective_function.update_objective_function()
        self.T = self.objective_function.obj_value
        if output:
            print("Initialization complete!")
            print("Initial value:", self.objective_function.obj_value)
            print('Num of initial clusters', len(self.objective_function.partition))

    def destruction_and_reconstruction(self):
        destruction_nodes = self.__destruction()
        self.__reconstruction(destruction_nodes)

    def acceptance_criterion(self, status, alpha=0.99, method='metropolis'):
        obj = self.objective_function
        last_solution = status['solution']
        last_value = status['value']
        if method == 'better':
            if last_value < obj.obj_value:
                obj.set_solution(last_solution)
                obj.obj_value = last_value
        else:

            if last_value < obj.obj_value and rd.random() > math.exp((last_value - obj.obj_value) / self.T):
                obj.set_solution(last_solution)
                obj.obj_value = last_value
        self.T *= alpha

    def run(self, max_iter=2000, output=True, multi_start=False):
        print("IG is running……")
        max_iter = max(max_iter, 10)
        start_time = time.time()
        if not multi_start:
            self.initialization(output=output)
        abandoned = self._dataset.vnum - len(self.node_available)
        ls = self.local_search
        ls.local_move()
        ls.community_merge()
        best_values = []

        while self.ct <= max_iter:
            status = self.record_status()
            self.destruction_and_reconstruction()
            ls.local_move()
            ls.community_merge()

            self.acceptance_criterion(status, method='better')

            if output:
                current_time = time.time()
                print("execution time: ", current_time - start_time, "s")
                print('%d/%d: best value --> %d with %d clusters' %
                      (self.ct, max_iter, self.objective_function.obj_value, len(self.objective_function.partition) - abandoned))
            best_values.append(self.objective_function.obj_value)
            self.ct += 1
            if self.ct == max_iter and best_values[-10] != best_values[-1]:
                max_iter += 10

        end_time = time.time()
        print('IG Complete!')
        print('=' * 40)
        print('Best Value:', self.objective_function.obj_value)
        print('time cost:', end_time - start_time, "s")
        return best_values

    def record_status(self):
        """
        for acceptance criterion

        :return: current partition status
        """
        status = {
            'solution': self.objective_function.solution.copy(),
            'value': self.objective_function.obj_value
        }
        return status

    def __destruction(self, roulette=False) -> list:
        """
        destruction phase

        :return: a list of the removed nodes
        """

        def get_frustration_of_clusters():
            frustrations = {}
            partition = self.objective_function.partition
            ns = self.neighborhood.neighborhood_structure

            for cid, community in partition.items():
                # calculate the frustration for each cluster
                for single_node in community:
                    if single_node not in ns:
                        continue
                    node_nbr = ns[single_node]
                    frustrations[single_node] = len(node_nbr['+'] - community) + len(community & node_nbr['-'])

            return frustrations

        obj = self.objective_function
        num_of_removed = int(obj.vnum * self.beta)
        if 0.9 ** self.ct > 0.1 and roulette:
            candidate = get_frustration_of_clusters()
            num_of_roulette = int(num_of_removed * 0.9 ** self.ct)
            random_selected_node = rd.sample(self.node_available, num_of_removed - num_of_roulette)
            removed_node = utils.roulette(candidate, num_of_roulette)
            removed_node.extend(random_selected_node)
            removed_node = list(set(removed_node))
        else:
            removed_node = rd.sample(self.node_available, int(obj.vnum * self.beta))
        for node in removed_node:
            delta = obj.delta_caused_by_decompose(node, self.neighborhood.neighborhood_structure[node])
            obj.decompose(node, delta=delta)

        return removed_node

    def __reconstruction(self, isolated_node):
        """
        construction phase

        :param isolated_node: a removed node list
        :return: None, reconstruct the partition
        """

        obj = self.objective_function
        nbr = self.neighborhood

        for node in isolated_node:
            min_delta = 0
            candidate = -1

            for nbr_community in nbr.get_adjacent_cluster(node, obj.solution):

                delta = obj.delta_caused_by_move(node, nbr_community, nbr.neighborhood_structure[node])
                if delta < min_delta:
                    min_delta = delta
                    candidate = nbr_community
            if candidate != -1:
                obj.move(node, candidate, min_delta)

    def __pretreatment(self):
        alone = set()
        isolated = set()

        for i in range(self._dataset.vnum):
            if i not in self.neighborhood.neighborhood_structure.keys():
                alone.add(i)
                continue
            if not self.neighborhood.neighborhood_structure[i]['+']:
                isolated.add(i)
        for node in isolated:
            for nbr in self.neighborhood.neighborhood_structure[node]["-"]:
                self.neighborhood.neighborhood_structure[nbr]["-"].remove(node)
            del self.neighborhood.neighborhood_structure[node]

        node_available = set(range(self._dataset.vnum)) - alone - isolated
        return list(node_available)

    # def __reconstruction_old(self, destruction_nodes, p_type):
    #
    #     obj = self.objective_function
    #
    #     if p_type == 'all':
    #         candidate_community = set(obj.partition.keys())
    #         for node in destruction_nodes:
    #             try:
    #                 obj.solution[node] = rd.choice(list(candidate_community - {obj.solution[node]}))
    #             except IndexError:
    #                 continue
    #
    #     elif p_type == 'neighbor':
    #         for node in destruction_nodes:
    #             try:
    #                 obj.solution[node] = rd.choice(list(obj.get_adjacent_community(node, self.neighborhood)))
    #             except IndexError:
    #                 continue
    #
    #     obj.partition = utils.solution2partition(obj.solution)
    #     obj.update_objective_function()


def get_end_position(values):
    target = values[-1]
    for i in range(len(values)):
        if values[i] == target:
            return i + 1
    return len(values)


if __name__ == '__main__':

    # file_name = r'../datasets/slashdot-undirected-size600-part0.g'
    file_name = r'../datasets/H/H98.g'
    ds = FileOperations.load_data(file_name)

    ig = IteratedGreedy(ds)
    ig.run(max_iter=200)
    print(ig.objective_function.partition)
