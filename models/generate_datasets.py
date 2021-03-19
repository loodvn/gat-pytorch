import networkx as nx
import matplotlib.pyplot as plt
import itertools
import torch
import torch_geometric.data
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import KFold

def split_function(number):

    list = []

    for i in range(3, number - 2):
        list.append(tuple((i, number - i)))
    return list


def generate_cycles(nodes):  # this is for creating a list of graphs

    single_cycles = []
    split_cycles = []
    pairs = split_function(nodes)

    for (i, j) in pairs:
        split_cycles.append((nx.disjoint_union(nx.cycle_graph(i),
                                               nx.cycle_graph(j)), 0))

        single_cycles.append((nx.cycle_graph(nodes), 1))

    return single_cycles, split_cycles


nodes = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

cycle_graphs = []


def generate_cycle_pairs(nodes):

    cycles = []

    full_cycles = []

    pairs = split_function(nodes)

    for (i, j) in pairs:
        cycles.append(tuple(((nx.disjoint_union(nx.cycle_graph(i), nx.cycle_graph(j)), 0),
                             ((nx.cycle_graph(nodes), 1)))))

    return cycles


for node in nodes:
    single_cycles, split_cycles = generate_cycles(node)
    cycle_graphs += single_cycles
    cycle_graphs += split_cycles

torch_dataset = []

def dataset_generation():

    for (graph, target) in cycle_graphs:
        number_of_nodes = graph.number_of_nodes()
        torch_graph = torch_geometric.utils.convert.from_networkx(graph)
        torch_graph.x = torch.zeros((number_of_nodes, 50))
        torch_graph.y = torch.tensor(target, dtype=torch.float)

        torch_dataset.append(torch_graph)
    return torch_dataset

output = dataset_generation()
print(output)