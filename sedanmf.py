import numpy as np
import networkx as nx
from typing import Dict
from sklearn.decomposition import NMF
from karateclub.estimator import Estimator


class SEDANMF(Estimator):
    """
    An implementation of similarity enhancement of deep autoencoder-like nonnegative matrix
    factorization base on "DANMF" (https://github.com/benedekrozemberczki/karateclub).
    We had rewritten the update rules of DANMF and provided a method for calculating the similarity matrix.
    """
    """
    Args:
        layers (list): Autoencoder layer sizes in a list of integers. Default [32, 8].
        pre_iterations (int): Number of pre-training epochs. Default 100.
        iterations (int): Number of training epochs. Default 100.
        seed (int): Random seed for weight initializations. Default 42.
        lamb (float): Regularization parameter. Default 0.01.
        seed (int): Random seed value. Default is 42.
    """

    def __init__(
            self,
            pre_iterations: int = 100,
            iterations: int = 100,
            seed: int = 42,
            lamb: float = 0.005,
    ):
        self._Pre_iterations = pre_iterations
        self.iterations = iterations
        self.seed = seed
        self.lamb = lamb
        self.seed = seed

    def _setup_target_matrices(self, graph):
        """
        Override: add self._A_dense
        Setup target matrix for pre-training process.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph being clustered.
        """
        self._graph = graph
        self._A = nx.adjacency_matrix(
            self._graph, nodelist=range(self._graph.number_of_nodes())
        )
        self._A_dense = self._A.toarray()
        self._L = nx.laplacian_matrix(
            self._graph, nodelist=range(self._graph.number_of_nodes())
        )
        self._D = self._L + self._A

    def _setup_z(self, i):
        """
        Setup target matrix for pre-training process.

        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i == 0:
            self._Z = self._A
        else:
            self._Z = self._V_s[i - 1]

    def _sklearn_pretrain(self, i):
        """
        Pre-training a single layer of the model with sklearn.

        Arg types:
            * **i** *(int)* - The layer index.
        """
        nmf_model = NMF(
            n_components=self.layers[i],
            init="random",
            random_state=self.seed,
            max_iter=self._Pre_iterations,
        )

        U = nmf_model.fit_transform(self._Z)
        V = nmf_model.components_
        return U, V

    def _pre_training(self):
        """
        Pre-training each NMF layer.
        """
        self._U_s = []
        self._V_s = []
        for i in range(self._p):
            self._setup_z(i)
            U, V = self._sklearn_pretrain(i)
            self._U_s.append(U)
            self._V_s.append(V)

    def _setup_Q(self):
        """
        Setting up Q matrices.
        """
        self._Q_s = [None for _ in range(self._p + 1)]
        self._Q_s[self._p] = np.eye(self.layers[self._p - 1])
        for i in range(self._p - 1, -1, -1):
            self._Q_s[i] = np.dot(self._U_s[i], self._Q_s[i + 1])

    def _update_U(self, i):
        """
        Override
        Updating left hand factors.

        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i == 0:
            R = np.multiply(self._S, np.multiply(self._S, self._U_s[0].dot(self._Q_s[1]).dot(self._Vp))).dot(
                self._VpT).dot(
                self._Q_s[1].T)
            R = R + self._A_sq.dot(self._U_s[0].dot(self._Q_s[1].dot(self._Q_s[1].T)))
            Ru = self._A_dense.dot(self._V_s[self._p - 1].T.dot(self._Q_s[1].T)) + np.multiply(self._S,
                                                                                               np.multiply(
                                                                                                   self._A_dense,
                                                                                                   self._S)).dot(
                self._V_s[self._p - 1].T.dot(self._Q_s[1].T))
            self._U_s[0] = (self._U_s[0] * Ru) / np.maximum(R, 10 ** -10)
        else:
            R = self._P.T.dot(
                np.multiply(self._S, np.multiply(self._S, self._P.dot(self._U_s[i]).dot(self._Q_s[i + 1]).dot(
                    self._Vp)))).dot(self._VpT).dot(self._Q_s[i + 1].T)
            R = R + self._A_sq.dot(self._P).T.dot(self._P).dot(self._U_s[i]).dot(self._Q_s[i + 1]).dot(
                self._Q_s[i + 1].T)
            Ru = self._A_dense.dot(self._P).T.dot(self._V_s[self._p - 1].T).dot(self._Q_s[i + 1].T) + np.multiply(
                self._S,
                np.multiply(
                    self._A_dense,
                    self._S)).dot(
                self._P).T.dot(self._V_s[self._p - 1].T).dot(self._Q_s[i + 1].T)
            self._U_s[i] = (self._U_s[i] * Ru) / np.maximum(R, 10 ** -10)

    def _update_P(self, i):
        """
        Setting up P matrices.

        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i == 0:
            self._P = self._U_s[0]
        else:
            self._P = self._P.dot(self._U_s[i])

    def _update_V(self, i):
        """
        Override
        Updating right hand factors.

        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i < self._p - 1:
            Vu = np.multiply(self._S, np.multiply(self._A_dense, self._S)).dot(self._P).T + np.multiply(self._S,
                self._A_dense.dot(self._P).T) + 2 * self._A_dense.dot(self._P).T
            Vd = np.multiply(self._S, self._P.T.dot(self._P.dot(self._V_s[i]))) + 2 * self._V_s[i]
            self._V_s[i] = self._V_s[i] * Vu / np.maximum(Vd, 10 ** -10)
        else:
            Vu = np.multiply(self._S, np.multiply(self._A_dense, self._S)).dot(self._P).T + np.multiply(self._S,
                self._A_dense.dot(
                    self._P).T) + 2 * self._A_dense.dot(
                self._P).T + (
                         self.lamb * self._A_dense.dot(self._V_s[i].T)).T
            Vd = np.multiply(self._S, self._P.T.dot(self._P.dot(self._V_s[i])))
            Vd = Vd + 2 * self._V_s[i] + (self.lamb * self._D.dot(self._V_s[i].T)).T
            self._V_s[i] = self._V_s[i] * Vu / np.maximum(Vd, 10 ** -10)

    def _setup_VpVpT(self):
        self._VpVpT = self._V_s[self._p - 1].dot(self._V_s[self._p - 1].T)

    def _setup_Asq(self):
        self._A_sq = self._A.dot(self._A.T)

    def get_embedding(self) -> np.array:
        r"""Getting the bottleneck layer embedding.

        Return types:
            * **embedding** *(Numpy array)* - The bottleneck layer embedding of nodes.
        """
        embedding = [self._P, self._V_s[-1].T]
        embedding = np.concatenate(embedding, axis=1)
        return embedding

    def get_memberships(self) -> Dict[int, int]:
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dict)*: Node cluster memberships.
        """
        index = np.argmax(self._P, axis=1)
        memberships = {int(i): int(index[i]) for i in range(len(index))}
        return memberships

    def fit(self, graph: nx.classes.graph.Graph):
        """
        Override
        Fitting SE-DANMF .

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._set_seed()
        graph = self._check_graph(graph)
        self._setup_target_matrices(graph)
        # self._build_S_None()
        self._build_S()
        self._pre_training()
        self._setup_Asq()
        for iteration in range(self.iterations):
            self._setup_Q()
            self._setup_VpVpT()
            self._Vp = self._V_s[self._p - 1]
            self._VpT = self._V_s[self._p - 1].T
            for i in range(self._p):
                self._update_U(i)
                self._update_P(i)
                self._update_V(i)

    def _build_S_None(self):
        """
        Test the effect of adding S or no
        """
        n = self._graph.number_of_nodes()
        self.layers = [n, n]
        self._p = len(self.layers)

        self._S = [[1 for _ in range(n)] for _ in range(n)]

    def _build_S(self):
        """
        Building the similarity matrix
        """
        n = self._graph.number_of_nodes()
        self.layers = [n, n]
        self._p = len(self.layers)

        w = np.zeros((n, n), dtype=float)
        columns = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                columns[i, j] = 0
                w[i, j] = 0
            w[i, i] = 1
            columns[i, i] = 1
        for edge in self._graph.edges():
            w[edge[0], edge[1]] = 1
            w[edge[1], edge[0]] = 1

        self._S = []
        for i in range(n):
            temp = []
            for j in range(n):
                xi = w[i]
                xj = columns[j]
                sum_xi = np.sqrt(np.sum(xi ** 2))
                sum_xj = np.sqrt(np.sum(xj ** 2))
                sum_x = sum_xj * sum_xi
                temp.append(np.sum((w[i, j]) * np.dot(xi, xj) / sum_x))
            self._S.append(temp)
        self._S = np.array(self._S)
