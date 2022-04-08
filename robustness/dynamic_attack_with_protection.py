from dynamic_attack import DynamicAttack

from common.file_operations import FileOperations
from robustness.centrality import Centrality

import numpy as np
import common.file_operations as fo


class DynamicAttackWithProtection(DynamicAttack):

    def get_protected_nodes(self, p=0.2) -> set:

        protected_nodes = set()
        num_of_protected_nodes = int(self.dataset.vnum * p)
        candidate = self.node_available.copy()
        alg = self.algorithm_to_get_frustration(self.dataset)

        for _ in range(num_of_protected_nodes):
            node = self.pick_next(alg, candidate)
            candidate.remove(node)
            protected_nodes.add(node)

        return protected_nodes

    def get_protected_nodes_by_frustration(self, p=0.2):

        alg = self.algorithm_to_get_frustration(self.dataset)
        initial_frustration = alg.objective_function.obj_value
        all_nodes = self.node_available.copy()
        delta_f = dict()

        for v in all_nodes:
            self.attack_node(v)
            alg = self.algorithm_to_get_frustration(self.dataset)
            current_frustration = alg.objective_function.obj_value
            delta_f[v] = current_frustration - initial_frustration
            # 攻击后再撤销攻击
            self.dataset.reverse_node(v)
            self.node_available.add(v)

        sorted_f = sorted(delta_f.items(), key=lambda x: x[1], reverse=True)
        num_of_protected_nodes = int(self.dataset.vnum * p)

        return {u[0] for u in sorted_f[:num_of_protected_nodes]}

    def get_protected_nodes_randomly(self, p=0.2):

        all_nodes = list(self.node_available.copy())
        num_of_protected_nodes = int(self.dataset.vnum * p)
        import random as rd
        rd.shuffle(all_nodes)
        return set(all_nodes[:num_of_protected_nodes])

    def execute(self, k=1.0):

        num_of_attack = int(k * self.dataset.vnum)
        alg = self.algorithm_to_get_frustration(self.dataset)
        robustness_value = 0
        m = self.dataset.enum

        pns = self.get_protected_nodes(p=0.1)
        # pns = self.get_protected_nodes_by_frustration(p=0.2)
        # pns = self.get_protected_nodes_randomly()
        for i in range(num_of_attack):

            if not self.node_available:
                break

            print("Attack is processing: {0} / {1} ...".format(i, num_of_attack))
            current_node = self.pick_next(alg=alg)

            if current_node in pns:
                print("Current node", current_node, "is protected!")
                self.process.append(alg.objective_function.obj_value)
                self.node_attack_sequence.append(current_node)
                self.node_available.remove(current_node)
                continue

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

    file_name = r'../datasets/H/H97.g'
    ds = FileOperations.load_data(file_name).to_dynamic()

    attack = DynamicAttackWithProtection(dataset=ds, centrality=Centrality.R_DEGREE)
    rb = attack.execute(k=1)
    file_path = r"../results/greedy_robust"
    fo.FileOperations.write_array_to_file(np.array(attack.process),
                                          file_name=file_path + "/(c1)process_97_0.2_" + attack.ATTACK_CENTRALITY[
                                              attack.centrality] + ".txt")
    fo.FileOperations.write_array_to_file(np.array(attack.node_attack_sequence),
                                          file_name=file_path + "/(c1)attack_sequence_97_0.2_" + attack.ATTACK_CENTRALITY[
                                              attack.centrality] + ".txt")
    attack.show_process()
    print(rb)
