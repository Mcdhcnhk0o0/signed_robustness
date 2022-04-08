import common.file_operations
import matplotlib.pyplot as plt

from collections import Counter


def get_degree_distribution(dataset: common.file_operations.Dataset, color='r'):

    degrees = [len(dataset.data[node]) for node in dataset.data.keys()]
    counter = Counter(degrees)
    for k, v in counter.items():
        plt.bar(k, v, color=color)
    # plt.show()


ds1 = common.file_operations.FileOperations.load_data(r"../datasets/H/H97.g")
ds2 = common.file_operations.FileOperations.load_data(r"../datasets/synthetic_for_robustness/syn_network_5_100_12_0.2_0.0_0")
get_degree_distribution(ds1, color="red")
plt.xlabel("degree")
plt.ylabel("number of nodes")
plt.show()
get_degree_distribution(ds2, color="blue")
plt.xlabel("degree")
plt.ylabel("number of nodes")
plt.show()