from balance.objective_function import ObjectiveFunction


class Frustration(ObjectiveFunction):
    """
    The class of the frustration, implements ObjectiveFunction.
    """

    def objective_function(self):
        """
        calculate the line index of structural balance using a solution vector, O(m)

        :return: frustration index
        """

        data = self._dataset.data
        sl = self.solution
        frustration = 0

        for node in range(self._dataset.vnum):
            cid = sl[node]
            for nbr, attr in data[node].items():
                # negative edges within clusters
                if sl[nbr] == cid and attr == -1:
                    frustration += 1
                # positive edges between clusters
                elif sl[nbr] != cid and attr == 1:
                    frustration += 1

        assert frustration % 2 == 0
        return frustration // 2

    def objective_function_v2(self, neighborhood, partition=None):
        """
        calculate the line index of structural balance using a partition, more effective

        :param partition: a given partition, default: current partition
        :param neighborhood: neighborhood structure need to be given in advance
        :return: frustration index
        """

        frustrations = {}
        if partition is None:
            partition = self.partition

        for cid, community in partition.items():

            pos_out, neg_in = 0, 0

            # calculate the frustration for each cluster
            for node in community:
                if node not in neighborhood:
                    continue
                node_nbr = neighborhood[node]
                pos_out += len(node_nbr['+'] - community)
                neg_in += len(community & node_nbr['-'])

            frustrations[cid] = pos_out + neg_in

        return sum(frustrations.values()) // 2

    def objective_function_native(self, triad):

        sl = self.solution
        frustration = 0

        for n1, n2, attr in triad:
            c1, c2 = sl[n1], sl[n2]
            if c1 == c2 and attr == -1:
                frustration += 1
            elif c1 != c2 and attr == 1:
                frustration += 1

        return frustration // 2

    def delta_caused_by_move(self, node, destination, node_neighborhood):
        """
        calculate the change of frustration index when a node is moved from its current cluster to destination cluster

        :param node: number of node
        :param node_neighborhood: neighborhood of node
        :param destination: target cluster
        :return: change of frustration index, negative if better
        """

        delta = 0
        current_cluster = self.solution[node]

        if current_cluster == destination:
            return 0

        for v in node_neighborhood['+']:
            cid = self.solution[v]
            # positive edges are expected within clusters
            if cid != current_cluster and cid == destination:
                # the frustration index decreases if positive edges are moved into clusters
                delta -= 1
            elif cid == current_cluster and cid != destination:
                delta += 1

        for v in node_neighborhood['-']:
            cid = self.solution[v]
            # negative edges are expected between clusters
            if cid != current_cluster and cid == destination:
                # the frustration index increases if negative edges are moved into clusters
                delta += 1
            elif cid == current_cluster and cid != destination:
                delta -= 1

        return delta

    def move(self, node, destination, delta):
        """
        move the node into the destination cluster
        Note: the partition and solution are changed when the function is called

        :param node: number of node
        :param destination: target cluster
        :param delta: change of the frustration index, obtained by delta_caused_by_move()
        :return: None
        """

        pre_cid = self.solution[node]

        self.solution[node] = destination
        self.partition[pre_cid].remove(node)
        if not self.partition[pre_cid]:
            del self.partition[pre_cid]
        self.partition[destination].add(node)

        self.obj_value += delta

    def delta_caused_by_merge(self, c1, c2, neighborhood):
        """
        calculate the change of frustration index when cluster c1 and cluster c2 are merged

        :param c1: number of cluster
        :param c2: number of cluster
        :param neighborhood: the whole neighborhood structure
        :return: change of frustration index, negative if better
        """

        c1_community, c2_community = self.partition[c1], self.partition[c2]
        data = self._dataset.data
        delta = 0

        for node in c1_community:
            intersection = (neighborhood[node]['+'] | neighborhood[node]['-']) & c2_community

            for another_node in intersection:
                if data[node][another_node] == 1:
                    delta -= 1
                elif data[node][another_node] == -1:
                    delta += 1

        return delta

    def merge(self, c1, c2, delta):
        """
        merge cluster c2 into c1
        Note: the cluster c2 is removed after the operation

        :param c1: number of cluster
        :param c2: number of cluster
        :param delta: change of the frustration index, obtained by delta_caused_by_merge()
        :return: None
        """

        for node in self.partition[c2]:
            self.solution[node] = c1

        self.partition[c1] = self.partition[c1] | self.partition[c2]
        del self.partition[c2]

        self.obj_value += delta

    def delta_caused_by_decompose(self, node, node_neighborhood):
        """
        calculate the change of frustration index when a node is moved out

        :param node: number of node
        :param node_neighborhood: neighborhood of node
        :return: change of frustration index, negative if better
        """

        cid = self.solution[node]
        if len(self.partition[cid]) == 1:
            return 0

        delta_pos = len([v for v in node_neighborhood['+'] if self.solution[v] == cid])
        delta_neg = len([v for v in node_neighborhood['-'] if self.solution[v] == cid])
        delta = 0
        for v in node_neighborhood['+']:
            if self.solution[v] == cid:
                delta += 1
        for v in node_neighborhood['-']:
            if self.solution[v] == cid:
                delta -= 1

        assert delta == delta_pos - delta_neg
        return delta

    def decompose(self, node, delta):
        """
        move the node out of its current cluster

        :param node: number of node
        :param delta: change of the frustration index, obtained by delta_caused_by_decompose()
        :return: None
        """

        if delta == 0:
            return

        pre_cid = self.solution[node]
        cid_available = node
        while cid_available in self.partition.keys():
            cid_available += 1

        self.partition[cid_available] = {node}
        self.partition[pre_cid].remove(node)
        self.solution[node] = cid_available

        self.obj_value += delta


if __name__ == "__main__":

    file_path = r'F:\github\structural_balance_ig\datasets\slashdot-undirected-size2000-part0.g'
