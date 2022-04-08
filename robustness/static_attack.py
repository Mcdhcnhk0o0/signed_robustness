from abc import ABC
from common.file_operations import DynamicDataset, FileOperations
from network_attack import NetworkAttack
from centrality import Centrality

import os
import numpy as np
import random as rd


class StaticAttack(NetworkAttack, ABC):
    DEFAULT_RANDOM_SEQ = r"random_seq_for_0.2.rb"

    def __init__(self, dataset: DynamicDataset, centrality):
        super().__init__(dataset, centrality)
        self.attack_sequence = self.get_attack_sequence()
        self.t = 0

    def get_attack_sequence(self):

        if os.path.exists(StaticAttack.DEFAULT_RANDOM_SEQ):
            return FileOperations.load_array_from_file(StaticAttack.DEFAULT_RANDOM_SEQ)

        node_list = []
        if self.centrality == Centrality.RANDOM:
            node_list = list(set(self.node_available))
            rd.shuffle(node_list)
        elif self.centrality == Centrality.DEGREE:
            pass

        node_list = np.array(node_list)
        FileOperations.write_array_to_file(node_list, file_name=StaticAttack.DEFAULT_RANDOM_SEQ)
        return node_list

    def pick_next(self):
        return self.attack_sequence[self.t]

    def execute(self, k=1):
        """
        the order of attack is computed in advance
        """

        num_of_attack = int(k * self.dataset.vnum)
        alg = self.algorithm_to_get_frustration(self.dataset)
        robustness_value = 0
        m = self.dataset.enum

        for i in range(num_of_attack):

            if not self.node_available:
                break

            print("Attack is processing: {0} / {1} ...".format(i, num_of_attack))
            current_node = self.pick_next()
            self.attack_node(current_node)
            alg = self.algorithm_to_get_frustration(self.dataset, max_iter=200)
            current_robustness = alg.objective_function.obj_value
            robustness_value += (m - current_robustness)

            self.process.append(current_robustness)
            self.t += 1

            print("Node", current_node, "is attacked!")
            print("Current frustration index:", alg.objective_function.obj_value)
            print("Max cluster size:", max([len(c) for c in alg.objective_function.partition.values()]))

        robustness_value = robustness_value / num_of_attack / m

        return robustness_value


if __name__ == "__main__":
    file_name = r'../datasets/synthetic_for_robustness/syn_network_5_100_12_0.2_0.0_0'
    ds = FileOperations.load_data(file_name).to_dynamic()

    attack = StaticAttack(dataset=ds, centrality=Centrality.RANDOM)
    rb = attack.execute(k=1)
    FileOperations.write_array_to_file(np.array(attack.process),
                                       file_name="process_101_" + attack.ATTACK_CENTRALITY[
                                           attack.centrality] + ".txt")
    FileOperations.write_array_to_file(np.array(attack.node_attack_sequence),
                                       file_name="attack_sequence_101_" + attack.ATTACK_CENTRALITY[
                                           attack.centrality] + ".txt")
    attack.show_process()
    print(rb)
