from common.file_operations import FileOperations
from robustness.dynamic_attack import DynamicAttack

import robustness.centrality as cent


if __name__ == "__main__":

    file_name = r'datasets/synthetic_for_robustness/syn_network_5_100_12_0.6_0.0_0'
    ds = FileOperations.load_data(file_name).to_dynamic()

    attack = DynamicAttack(dataset=ds, centrality=cent.Centrality.RANDOM)
    rb = attack.execute(k=1)
    attack.show_process()
    print(rb)
