import torch, random
import torch.nn.functional as F
import numpy as np
import networkx as nx
import igraph as ig
from igraph import Graph
import torch_geometric as tg
import matplotlib.pyplot as plt
from pprint import pprint

def generate_random_ER_graph(N=1000, n=50, p=0.10):
    all_er_graphs = []
    for i in range(N):
        p = torch.rand(1).item()
        graph = Graph.Erdos_Renyi(n, p)
        all_er_graphs.append([graph, n, p])

    return all_er_graphs

def count_triangles(adj):
    new_graph = nx.from_numpy_matrix(adj)
    counts = nx.triangles(new_graph)
    n_tri = sum(counts.values()) // 3

    return n_tri

def encode_graphs(graphs, N=1000, n=50):
    all_features = torch.Tensor(N, n, 1)
    all_labels = torch.Tensor(N, 1)

    all_data = []

    for i in range(N):
        graph, n, p = graphs[i]
        node_count = graph.vcount()
        adj = np.array(list(graph.get_adjacency()))
        edge_idx, edge_wt = tg.utils.dense_to_sparse(torch.from_numpy(adj))

        features = torch.zeros(size=(node_count, 1))

        """
        Get number of triangles from current graph
        """
        label = count_triangles(adj)

        all_features[i] = features
        all_labels[i] = label

        temp = tg.data.Data(
            x=all_features,
            y=all_labels,
            edge_index=edge_idx,
        )

        all_data.append(temp)

    return all_data

graphs = generate_random_ER_graph()
data = encode_graphs(graphs)
pprint (data[:5])