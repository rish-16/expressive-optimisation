import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn

servings = {
    "GCN": None, # graph convolutional network
    "GIN": None, # graph isomorphism network
    "ESAN": None, # equivariant subgraph aggregation network
    "2-IGN": None, # 2-invariant graph network
    "2-IGN+": None, # provably powerful graph networks
    "P-GNN": None, # position-aware graph neural networks
    "k-GNN": None, # k-WL graph neural network
    "MPNN": None # message passing neural network
}
