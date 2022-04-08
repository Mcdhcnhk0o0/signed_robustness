import numpy as np


class PageRank:

    def __init__(self, dataset):
        import robustness.robustness_utils as utils
        self.arr = utils.build_graph_from_numpy(dataset=dataset)
        self.__ignore_signs()

    def __ignore_signs(self):
        self.arr = np.abs(self.arr)

    def __trans_pre(self):
        """
        构造转移矩阵
        """
        b = np.transpose(self.arr)  # 把矩阵转置
        c = np.zeros(self.arr.shape, dtype=float)
        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                c[i][j] = self.arr[i][j] / b[j].sum()  # 把所有的元素重新分配
        return c

    @staticmethod
    def __init_pre(c):
        """
        pr值的初始化
        """
        pr = np.zeros((c.shape[0], 1), dtype=float)
        for i in range(c.shape[0]):
            pr[i] = float(1) / c.shape[0]
        return pr

    @staticmethod
    def _page_rank(p, m, v):
        """
        PageRank核心算法，p是网页跳转概率，m是转移矩阵，v是pr值
        """
        while not (v == p * np.dot(m, v) + (1 - p) * v).all():
            v = p * np.dot(m, v) + (1 - p) * v
            print((v == p * np.dot(m, v) + (1 - p) * v).all())
        return v

    def execute(self, p=0.85):
        trans_matrix = self.__trans_pre()
        pr = self.__init_pre(trans_matrix)
        return self._page_rank(p, trans_matrix, pr)

