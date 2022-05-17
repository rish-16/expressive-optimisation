import torch, random
import torch.nn.functional as F
import numpy as np
import igraph as ig
from igraph import Graph
import torch_geometric as tg
import matplotlib.pyplot as plt

def generate_random_ER_graph(N=1000, n=50, p=0.10):
    all_er_graphs = []
    for i in range(N):
        p = torch.rand(1).item()
        graph = Graph.Erdos_Renyi(n, p)
        all_er_graphs.append([graph, n, p])

    return all_er_graphs     

def encode_graphs(graphs, N=10000, n=50):
    all_features = torch.Tensor(N, n, 1)
    all_labels = torch.Tensor(N, 1)

    all_data = []

    for i in range(N):
        graph = graphs[i]
        node_count = graph.vcount()
        adj = np.array(graph.get_adjacency())
        edge_idx, edge_wt = tg.utils.dense_to_sparse(adj)

        features = torch.zeros(size=(node_count, 1))

        """
        Get number of triangles from current graph
        """
        label = ... # TODO

        all_features[i] = features
        all_labels[i] = label

        temp = tg.data.Data(
            x=all_features,
            y=all_labels,
            edge_index=edge_idx,
        )

        all_data.append(temp)

    return all_data