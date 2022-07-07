import torch, random
import torch.nn.functional as F
import numpy as np
import igraph as ig
from igraph import Graph
import torch_geometric as tg
import matplotlib.pyplot as plt

def plot_graph(graph, length):
    # labels cannot be None
    fig, axs = plt.subplots(figsize=(10, 8))
    ig.plot(
        graph, 
        target=axs,
        vertex_size=25,
        vertex_color="yellow", # blue, red, green, yellow
        vertex_label=list(range(length)),
        layout="circle"
    )
    plt.title("Chain of length " + str(length))
    plt.show()

def create_chains(length):
    graph = ig.Graph()
    graph.add_vertex("0")
    for i in range(1, L):
        graph.add_vertex(str(i))
        graph.add_edge(str(i-1), str(i))
    return graph

def encode_chains(graph, N=50000, plot_sample=False):

    adj = torch.from_numpy(np.array(list(graph.get_adjacency())))
    edge_index, edge_wt = tg.utils.dense_to_sparse(adj)
    node_count = graph.vcount()
    edge_count = graph.ecount()

    all_features = torch.Tensor(N, 1)
    labels = torch.Tensor(N, 1)

    for r in range(N):
        colour = 0
        label = 0

        if r % 5000 == 0 and plot_sample:
            temp = digits.tolist()
            plot_graph(graph, temp, str(target_count))

        all_features[r] = colour
        labels[r] = label

    return tg.data.Data(all_features, edge_index=edge_index, y=labels)

L = 10
graph = create_chains(L)
data = encode_chains(graph)
print (data)
