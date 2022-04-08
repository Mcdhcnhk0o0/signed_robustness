import os
import abc
import matplotlib.pyplot as plt
import algorithm.iterated_greedy_algorithm as ig

from loguru import logger
from robustness.centrality import Centrality
from common.file_operations import DynamicDataset, FileOperations

logger.add(r"../results/robustness.log")


class NetworkAttack:

    ATTACK_CENTRALITY = {1: 'random',
                         2: 'node_frustration',
                         3: 'betweenness',
                         4: 'degree',
                         5: 'r_degree'}

    def __init__(self, dataset: DynamicDataset, centrality):
        self.dataset = dataset
        self.centrality = centrality
        self.process = []
        self.node_attack_sequence = []
        self.node_attack_sequence_cache = None
        self.node_available = set(dataset.data.keys())
        self.pick_helper = Centrality(centrality)

        # if self.centrality == Centrality.RANDOM:
        #     if os.path.exists(r"random_seq_for_97.rb"):
        #         self.node_attack_sequence_cache = list(FileOperations.load_array_from_file(r"random_seq_for_97.rb"))
        #     else:
        #         raise RuntimeError("Unrepeatable random!")

    def attack_node(self, node):
        if node not in self.node_available:
            return
        self.dataset.reverse_node(node)
        self.node_available.remove(node)

    def cluster_attack(self):
        pass

    @abc.abstractmethod
    def pick_next(self, **kwargs):
        pass

    @abc.abstractmethod
    def execute(self):
        pass

    @staticmethod
    def algorithm_to_get_frustration(dataset, max_iter=150):
        alg = ig.IteratedGreedy(dataset=dataset)
        alg.run(max_iter=max_iter, output=False)
        return alg

    def show_process(self):
        x = range(len(self.process))
        plt.plot(x, self.process)
        plt.show()

