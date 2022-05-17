import torch, random
import torch.nn.functional as F
import numpy as np
import igraph as ig
from igraph import Graph

"""
generate random graphs
generate random features {1, 0}
generate associated labels
"""

def generate_random_ER_graphs(N=10000, p=0.20):
    all_er_graphs = []
    for i in range(N):
        n = random.randint(0, 1000)
        graph = Graph.Erdos_Renyi(n, p)
        all_er_graphs.append([graph, n, p])

    return all_er_graphs        

def encode_graphs(all_graphs, target=2):
    dataset = []
    for record in all_graphs:
        graph, n, p = record
        features = torch.Tensor((n, 1))
        labels = torch.Tensor((n, 1))
        for i in range(n):
            label = random.randint(0, 11)
            if label == target:
                features[i] = torch.tensor(1)
            else:
                features[i] = torch.tensor(0)
        