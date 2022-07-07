import torch, random
import torch.nn.functional as F
import numpy as np
import igraph as ig
from igraph import Graph
import torch_geometric as tg
import matplotlib.pyplot as plt

"""
generate random graphs
generate random features {1, 0}
generate associated labels
"""

def global_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# global_seed(0)

def plot_graph(graph, labels, title=0):
    # labels cannot be None
    fig, axs = plt.subplots(figsize=(10, 8))
    ig.plot(
        graph, 
        target=axs,
        vertex_size=20,
        vertex_color="yellow", # blue, red, green, yellow
        vertex_label=labels
    )
    plt.title("Graph with target count " + title)
    plt.show()

def generate_random_ER_graph(n=50, p=0.10):
    return Graph.Erdos_Renyi(n, p)

def encode_graphs(graph, N=50000, target=2, plot_sample=False):

    adj = torch.from_numpy(np.array(list(graph.get_adjacency())))
    edge_index, edge_wt = tg.utils.dense_to_sparse(adj)
    node_count = graph.vcount()
    edge_count = graph.ecount()

    all_features = torch.Tensor(N, node_count, 1)
    labels = torch.Tensor(N, 1)

    for r in range(N):
        digits = torch.randint(0, 5, size=(node_count, 1))
        target_count = 0
        for idx in range(node_count):
            i = digits[idx].item()
            if i == target:
                digits[idx] = torch.tensor(1)
                target_count += 1
            else:
                digits[idx] = torch.tensor(0)

        if r % 5000 == 0 and plot_sample:
            temp = digits.tolist()
            plot_graph(graph, temp, str(target_count))

        all_features[r] = digits
        labels[r] = target_count

    return tg.data.Data(all_features, edge_index=edge_index, y=labels)

g = generate_random_ER_graph()
data = encode_graphs(g, plot_sample=True)
print (data.edge_index.shape)