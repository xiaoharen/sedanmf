import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import random
import itertools

from sedanmf import SEDANMF


def generate_negative_samples(G, positive_edges, num_negative_samples):
    """
    Generate negative samples (non-existent edges)
    """
    all_possible_edges = list(itertools.combinations(G.nodes(), 2))
    negative_edges = set(all_possible_edges) - set(positive_edges)
    negative_samples = random.sample(list(negative_edges), num_negative_samples)
    return negative_samples


def link_prediction_scores(G, node_embeddings):
    """
    Calculate the link prediction score between all node pairs
    """
    node_pairs = list(itertools.combinations(G.nodes(), 2))
    scores = []
    for u, v in node_pairs:
        if u < v:  # Only consider one half of the edges in an undirected graph to avoid repetition
            score = np.dot(node_embeddings[u], node_embeddings[v])
            scores.append(score)
    return scores


def evaluate_link_prediction(y_true, y_scores):
    """
    Calculate AUC and AP (Average Precision) for link prediction
    """
    roc_auc = roc_auc_score(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    return roc_auc, average_precision


def get_input():
    file_path = 'test_data/sw.txt'

    G = nx.Graph()

    with open(file_path, 'r') as file:
        for line in file:
            nodes = line.strip().split()
            u, v = int(nodes[0]), int(nodes[1])
            G.add_edge(u, v)

    # G = nx.karate_club_graph()
    return G

def bipartite_link_prediction(G):
    edges = list(G.edges())
    train_edges, test_edges = train_test_split(edges, test_size=0.20, random_state=0)

    G_train = G.copy()
    G_train.remove_edges_from(test_edges)

    model = SEDANMF()

    model.fit(G_train)

    node_embeddings = model.get_embedding()

    train_scores = link_prediction_scores(G_train, node_embeddings)

    test_edge_labels = [1] * len(test_edges)
    num_negative_samples = len(test_edges)
    negative_samples = generate_negative_samples(G, edges, num_negative_samples)
    test_non_edge_labels = [0] * len(negative_samples)

    test_scores = []
    for edge in itertools.chain(test_edges, negative_samples):
        u, v = edge
        score = np.dot(node_embeddings[u], node_embeddings[v])
        test_scores.append(score)

    y_true = test_edge_labels + test_non_edge_labels
    y_scores = test_scores

    roc_auc, average_precision = evaluate_link_prediction(y_true, y_scores)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average-Precision: {average_precision:.4f}")


if __name__ == "__main__":
    G = get_input()
    bipartite_link_prediction(G)