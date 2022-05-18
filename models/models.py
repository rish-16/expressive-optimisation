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

class GCN(nn.Module):
    def __init__(self, indim, hidden, outdim):
        self.conv1 = tgnn.GraphConv(indim, hidden)
        self.conv2 = tgnn.GraphConv(hidden, hidden)
        self.conv3 = tgnn.GraphConv(hidden, outdim)

    def forward(self, x, edge_idx):
        x = torch.relu(self.conv1(x, edge_idx))
        x = torch.relu(self.conv2(x, edge_idx))
        out = torch.softmax(self.conv3(x, edge_idx), 1)

        return out

class GIN(nn.Module):
    def __init__(self, indim, hidden, outdim):
        self.l1 = nn.Linear()
        self.conv1 = tgnn.GINConv(nn, )
        self.conv2 = tgnn.GraphConv(hidden, hidden)
        self.conv3 = tgnn.GraphConv(hidden, outdim)

    def forward(self, x, edge_idx):
        x = torch.relu(self.conv1(x, edge_idx))
        x = torch.relu(self.conv2(x, edge_idx))
        out = torch.softmax(self.conv3(x, edge_idx), 1)

        return out        