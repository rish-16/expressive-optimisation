import numpy as np
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt

graph = ig.Graph()
graph.add_vertex("0")
graph.add_vertex("1")
graph.add_vertex("2")
graph.add_vertex("3")
graph.add_vertex("4")

graph.add_edge("0", "1")
graph.add_edge("1", "2")
graph.add_edge("0", "2")
graph.add_edge("0", "3")
graph.add_edge("3", "2")
graph.add_edge("0", "4")
graph.add_edge("1", "4")

adj = np.array(list(graph.get_adjacency()))
new_graph = nx.from_numpy_matrix(adj)
counts = nx.triangles(new_graph)
n_tri = sum(counts.values()) // 3

print (n_tri)

def plot_graph(graph):
    # labels cannot be None
    fig, axs = plt.subplots(figsize=(10, 8))
    ig.plot(
        graph, 
        target=axs,
        vertex_size=20,
        vertex_color="yellow", # blue, red, green, yellow
        layout="circle"
    )
    plt.show()

# plot_graph(graph)