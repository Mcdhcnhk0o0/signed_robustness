import re
import os
import datetime
import collections
import numpy as np


class Dataset:
    """
    data structure for a given dataset

    vnum：num of vertices, int
    enum：num of edges, int
    dataset: a graph stored by adjacency table using hash, dict(dict())
    """

    def __init__(self):
        self.vnum = 0
        self.enum = 0
        self.data = dict()

    def to_dynamic(self):
        dynamic_dataset = DynamicDataset()
        dynamic_dataset.data = self.data
        dynamic_dataset.vnum = self.vnum
        dynamic_dataset.enum = self.enum
        return dynamic_dataset


class DynamicDataset(Dataset):
    """
    data structure for a given dataset about robustness
    """

    def reverse_node(self, v):
        for nbr in self.data[v]:
            self.reverse_edge(v, nbr)

    def reverse_edge(self, v1, v2):
        self.data[v1][v2] *= -1
        self.data[v2][v1] *= -1

    def remove_node(self, v):
        if v not in self.data:
            return
        for nbr in self.data[v]:
            del self.data[nbr][v]

        del self.data[v]

    @staticmethod
    def dataset2dynamic(dataset: Dataset):
        dynamic_dataset = DynamicDataset()
        dynamic_dataset.data = dataset.data
        dynamic_dataset.vnum = dataset.vnum
        dynamic_dataset.enum = dataset.enum
        return dynamic_dataset


class FileOperations:

    @staticmethod
    def load_data(path: str, network_type='signed', header=True) -> Dataset:
        """
        read data from a local file

        :param path: file path
        :param network_type: enum {"unsigned", "signed"}
        :return: an instance of class Dataset
        """

        dataset = Dataset()
        data = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        print('Loading data from ' + path)

        with open(path) as f:
            # the first line
            if header:
                first_line = f.readline()
                vnum, enum = first_line.split()
                dataset.vnum, dataset.enum = int(vnum), int(enum)
            # the remaining
            if network_type == 'signed':
                # each line in the file is expected in the form of "n1 n2 attr"
                for each in f:
                    n1, n2, attr = each.split()
                    if attr != '1' and attr != '-1' and attr != '1.0' and attr != '-1.0':
                        continue
                    n1, n2 = int(n1), int(n2)
                    data[n1][n2] = data[n2][n1] = 1 if attr in {"1", "1.0"} else -1
            elif network_type == 'unsigned':
                # expected form: "n1 n2"
                for each in f:
                    n1, n2 = each.split(",")
                    n1, n2 = int(n1), int(n2)
                    data[n1][n2] = data[n2][n1] = 1
            else:
                raise TypeError('no such type of network')

        if min(data.keys()) == 1 and max(data.keys()) == dataset.vnum:
            print("The dataset may start with 1. Corresponding method is called.")
            return FileOperations.load_data_with_start_one(path, network_type=network_type)

        dataset.data = data
        print('Loading complete!')

        return dataset

    @staticmethod
    def load_data_with_start_one(path: str, network_type='signed') -> Dataset:
        """
        In some files or programming languages, the index of an array or the number of a node starts from 1.

        """
        dataset = Dataset()
        data = collections.defaultdict(dict)
        with open(path) as f:
            header = f.readline()
            vnum, enum = header.split()
            dataset.vnum, dataset.enum = int(vnum), int(enum)
            if network_type == 'signed':
                for each in f:
                    n1, n2, attr = each.split()
                    n1, n2 = int(n1) - 1, int(n2) - 1
                    data[n1][n2] = data[n2][n1] = int(attr)
            elif network_type == 'unsigned':
                for each in f:
                    n1, n2 = each.split()
                    n1, n2 = int(n1) - 1, int(n2) - 1
                    data[n1][n2] = data[n2][n1] = 1
            else:
                raise TypeError('no such type of network')
        dataset.data = data
        return dataset

    @staticmethod
    def load_data_native(path: str) -> list:
        dataset = []
        with open(path) as f:
            f.readline()
            for each in f:
                n1, n2, attr = each.split()
                if attr != '1' and attr != '-1':
                    continue
                n1, n2 = int(n1), int(n2)
                dataset.append([n1, n2, int(attr)])

        return dataset

    @staticmethod
    def write_array_to_file(array: np.ndarray, file_name):
        """
        serialize the array to a local file
        """
        file_name = FileOperations.check_existence(file_name)
        if len(array.shape) == 1:  # 1d
            with open(file_name, 'w') as f:
                f.write(",".join([str(e) for e in array]))
        elif len(array.shape) == 2:  # 2d
            with open(file_name, 'w') as f:
                for i in range(array.shape[0]):
                    f.write(",".join([str(e) for e in array[i, :]]))
                    f.write("\n")

    @staticmethod
    def load_array_from_file(file_name) -> np.ndarray:
        """
        deserialize
        """
        content = []
        with open(file_name) as f:
            for line in f:
                # 支持文件使用 , 或者 空格对元素进行分割
                content.append([float(e) for e in re.split(pattern=", | ,| |,", string=line)])
        if not content:
            raise FileNotFoundError("Cannot resolve the file.")
        if len(content) == 1:
            return np.array(content[0])
        else:
            return np.array(content)

    @staticmethod
    def dataset2g(dataset, file_name='generated_dataset'):
        """
        write a generated dataset to a local file

        :param file_name: give the file a name
        :param dataset: an instance of class Dataset
        :return: file_name
        """

        file_name = FileOperations.check_existence(file_name)

        with open(file_name, 'w') as f:
            f.write(str(dataset.vnum) + '\t' + str(dataset.enum) + '\n')
            for node in dataset.data:
                for nbr, attr in dataset.data[node].items():
                    f.write(str(node) + '\t' + str(nbr) + '\t' + str(attr) + '\n')

        print('-> The dataset is write as ' + file_name)
        return file_name

    @staticmethod
    def dataset2csv(dataset, file_name="default_file_name", col=3):

        file_name = FileOperations.check_existence(file_name)

        with open(file_name, 'w') as f:
            if col == 2:
                for node in dataset.data:
                    for nbr, attr in dataset.data[node].items():
                        f.write(str(node) + ',' + str(nbr) + '\n')
            elif col == 3:
                for node in dataset.data:
                    for nbr, attr in dataset.data[node].items():
                        f.write(str(node) + ',' + str(nbr) + ',' + str(attr) + '\n')

        print('-> The dataset is write as ' + file_name)
        return file_name

    @staticmethod
    def dataset2gephi(file_name: str, alg, dataset_path: str):
        """
        output a specified csv file for gephi
        """

        file_name = FileOperations.check_existence(file_name)

        solution = alg.objective_function.solution
        node_available = alg.node_available
        edge_file_header = "Source,Target,Type,Id,Label,timeset,Weight"
        node_file_header = "Id,Label,timeset"

        index = 0
        with open(dataset_path) as f:
            f.readline()
            with open(file_name + "_edge_list.csv", 'w') as g:
                g.write(edge_file_header)
                g.write("\n")
                for line in f:
                    element = line.split()
                    new_line = [element[0], element[1], "Undirected", str(index), element[2], "", "1"]
                    g.write(",".join(new_line))
                    g.write("\n")
                    index += 1

        alone = 999999
        with open(file_name + "_node_list.csv", 'w') as f:
            f.write(node_file_header)
            f.write("\n")
            for i in range(len(solution)):
                cid = solution[i]
                if i not in node_available:
                    cid = alone
                f.write(",".join([str(i), str(cid), ""]))
                f.write("\n")

        print("The edge list and node list are written!")

    @staticmethod
    def check_existence(file_name: str):
        file_name_sec = file_name.split(".")
        if os.path.exists(file_name):
            time_stamp = "_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")
            if len(file_name_sec) > 1:
                file_name = ".".join(file_name_sec[:-1]) + time_stamp + "." + file_name_sec[-1]
            else:
                file_name = file_name + time_stamp
        return file_name
