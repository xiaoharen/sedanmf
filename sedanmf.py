import numpy as np
import networkx as nx
from typing import Dict

from karateclub import Estimator
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity


class SEDANMF(Estimator):
    """
    An implementation of similarity enhancement of deep autoencoder-like nonnegative matrix
    factorization base on "DANMF" (https://github.com/benedekrozemberczki/karateclub).
    We had rewritten the update rules of DANMF and provided a method for calculating the similarity matrix.
    """

    """
    Args:
        pre_iterations (int): Number of pre-training epochs. Default 10.
        iterations (int): Number of training epochs. Default 1000.
        seed (int): Random seed value. Default is 42.
        lamb (float): Regularization parameter. Default 0.01.
        span (int): The span of factorization. Default 2.
    """

    def __init__(
        self,
        pre_iterations: int = 10,
        iterations: int = 1000,
        seed: int = 42,
        lamb: float = 0.01,
        span: int = 2
    ):
        self.pre_iterations = pre_iterations
        self.iterations = iterations
        self.seed = seed
        self.lamb = lamb
        self.span = span

    def _set_layers(self):
        """
        Autoencoder layer sizes in a list of integers
        """
        self.layers = []
        temp = self._n
        while temp >= 2:
            self.layers.append(temp)
            temp //= self.span
        self._p = len(self.layers)

    def _setup_target_matrices(self, graph):
        """
        Setup target matrix for pre-training process.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph being clustered.
        """
        self._graph = graph
        self._n = self._graph.number_of_nodes()
        self._A = nx.adjacency_matrix(
            self._graph, nodelist=range(self._graph.number_of_nodes())
        )
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
            max_iter=self.pre_iterations,
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
        Updating left hand factors.

        Override by Jaylen Foo

        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i == 0:
            R = self._U_s[0].dot(self._Q_s[1].dot(self._VpVpT).dot(self._Q_s[1].T))
            if self._S_sqr.shape == R.shape:
                R = np.multiply(self._S_sqr, R)
            R = R + self._A_sq.dot(self._U_s[0].dot(self._Q_s[1].dot(self._Q_s[1].T)))
            Ru = self._A.dot(self._V_s[self._p - 1].T.dot(self._Q_s[1].T))
            if self._S_sqr.shape == Ru.shape:
                Ru = np.multiply(self._S_sqr, Ru)
            Ru = Ru + self._A.dot(self._V_s[self._p - 1].T.dot(self._Q_s[1].T))
            self._U_s[0] = (self._U_s[0] * Ru) / np.maximum(R, 10**-10)
        else:
            R = (self._P.T.dot(self._P).dot(self._U_s[i]).dot(self._Q_s[i + 1]).dot(self._VpVpT).dot(self._Q_s[i + 1].T))
            if self._S_sqr.shape == R.shape:
                R = np.multiply(self._S_sqr, R)
            R = R + self._A_sq.dot(self._P).T.dot(self._P).dot(self._U_s[i]).dot(self._Q_s[i + 1]).dot(self._Q_s[i + 1].T)
            Ru = self._A.dot(self._P).T.dot(self._V_s[self._p - 1].T).dot(self._Q_s[i + 1].T)
            if self._S_sqr.shape == Ru.shape:
                Ru = np.multiply(self._S_sqr, Ru)
            Ru = Ru + self._A.dot(self._P).T.dot(self._V_s[self._p - 1].T).dot(self._Q_s[i + 1].T)
            self._U_s[i] = (self._U_s[i] * Ru) / np.maximum(R, 10**-10)

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
        Updating right hand factors.

        Override by Jaylen Foo

        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i < self._p - 1:
            Vu = self._A.dot(self._P).T
            if self._S_sqr.shape == Vu.shape:
                Vu = np.multiply(self._S_sqr, Vu) + np.multiply(self._S, Vu)
            else:
                Vu = 4 * Vu
            Vd = 2 * self._P.T.dot(self._P).dot(self._V_s[i])
            if self._S_sqr.shape == Vd.shape:
                Vd = np.multiply(self._S, Vd)
            Vd = Vd + 2 * self._V_s[i]
            self._V_s[i] = self._V_s[i] * Vu / np.maximum(Vd, 10 ** -10)
        else:
            Vu = self._A.dot(self._P).T
            if self._S_sqr.shape == Vu.shape:
                Vu = np.multiply(self._S_sqr, Vu) + np.multiply(self._S, Vu)
            else:
                Vu = 4 * Vu
            Vu = Vu + 2 * (self.lamb * self._A.dot(self._V_s[i].T)).T
            Vd = 2 * self._P.T.dot(self._P).dot(self._V_s[i])
            if self._S_sqr.shape == Vd.shape:
                Vd = np.multiply(self._S, Vd)
            Vd = Vd + 2 * self._V_s[i] + (self.lamb * self._D.dot(self._V_s[i].T)).T
            self._V_s[i] = self._V_s[i] * Vu / np.maximum(Vd, 10**-10)

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
        Fitting a DANMF clustering model.

        Override by Jaylen Foo

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """

        self._set_seed()
        graph = self._check_graph(graph)
        self._setup_target_matrices(graph)
        self._set_layers()
        self._build_S()
        # self._build_S_None()
        self._pre_training()
        self._setup_Asq()
        self.loss = []
        for iteration in range(self.iterations):
            self._setup_Q()
            self._setup_VpVpT()
            for i in range(self._p):
                self._update_U(i)
                self._update_P(i)
                self._update_V(i)
            self._calculate_cost(iteration)

    def _build_S_None(self):
        """
        Test the effect of adding S or no
        """
        self.layers = [self._n, self._n]
        self._p = len(self.layers)

        self._S = [[1 for _ in range(self._n)] for _ in range(self._n)]
        self._S_sqr = np.multiply(self._S, self._S)

    def _build_S(self):
        """
        Building the similarity matrix
        """
        G = nx.to_numpy_array(self._graph)
        self._S = cosine_similarity(G)
        self._S_sqr = np.multiply(self._S, self._S)

    def _calculate_cost(self, i):
        """
        Calculate loss.
        :param i: Global iteration.
        """
        reconstruction_loss_1 = np.linalg.norm(np.multiply(self._S, self._A - self._P.dot(self._V_s[-1])), ord="fro") ** 2
        reconstruction_loss_2 = np.linalg.norm(self._V_s[-1] - self._A.dot(self._P).T, ord="fro") ** 2
        regularization_loss = np.trace(self._V_s[-1].dot(self._L.dot(self._V_s[-1].T)))
        self.loss.append(reconstruction_loss_1 + reconstruction_loss_2 + regularization_loss)
