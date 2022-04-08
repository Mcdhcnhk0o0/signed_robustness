from abc import ABC

from common.file_operations import FileOperations
from robustness.network_attack import NetworkAttack
from robustness.centrality import Centrality

import numpy as np
import common.file_operations as fo


class DynamicAttack(NetworkAttack, ABC):

    def pick_next(self, alg, candidate=None):

        if self.node_attack_sequence_cache is not None:
            target = self.node_attack_sequence_cache.pop(0)
            return target

        if not candidate:
            candidate = self.node_available

        if self.centrality == Centrality.NODE_FRUSTRATION or self.centrality == Centrality.DEGREE\
                or self.centrality == Centrality.R_DEGREE:
            param = {'partition': alg.objective_function.partition,
                     'neighbor_structure': alg.neighborhood.neighborhood_structure}
        else:
            param = self.dataset
        return self.pick_helper.get_node_with_best_centrality(param, candidate)

    def execute(self, k=1.0):

        num_of_attack = int(k * self.dataset.vnum)
        alg = self.algorithm_to_get_frustration(self.dataset)
        robustness_value = 0
        m = self.dataset.enum

        for i in range(num_of_attack):

            if not self.node_available:
                break

            print("Attack is processing: {0} / {1} ...".format(i, num_of_attack))
            current_node = self.pick_next(alg=alg)
            self.attack_node(current_node)
            alg = self.algorithm_to_get_frustration(self.dataset, max_iter=200)
            current_robustness = alg.objective_function.obj_value
            robustness_value += (m - current_robustness)

            self.process.append(current_robustness)
            self.node_attack_sequence.append(current_node)

            print("Node", current_node, "is attacked!")
            print("Current frustration index:", alg.objective_function.obj_value)
            print("Max cluster size:", max([len(c) for c in alg.objective_function.partition.values()]))

        robustness_value = robustness_value / num_of_attack / m

        return robustness_value


if __name__ == "__main__":

    file_name = r'../datasets/synthetic_for_robustness/syn_network_5_100_12_0.6_0.0_0'
    ds = FileOperations.load_data(file_name).to_dynamic()

    attack = DynamicAttack(dataset=ds, centrality=Centrality.RANDOM)
    rb = attack.execute(k=1)
    fo.FileOperations.write_array_to_file(np.array(attack.process),
                                          file_name="process_0.6_" + attack.ATTACK_CENTRALITY[
                                              attack.centrality] + ".txt")
    fo.FileOperations.write_array_to_file(np.array(attack.node_attack_sequence),
                                          file_name="attack_sequence_0.6_" + attack.ATTACK_CENTRALITY[
                                              attack.centrality] + ".txt")
    attack.show_process()
    print(rb)
