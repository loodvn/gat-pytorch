import pickle

# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx

# Main computation libraries
import scipy.sparse as sp
import numpy as np
import igraph as ig


# Deep learning related imports
import torch

import os
import enum
import time

from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from scipy.stats import entropy

import typing
from typing import List

# from utils import GATCora, load_cora_data, train_cora_ow, prep_model_and_data_for_analysis

# node_label_colour_map = {
#     "Cora": {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"},
#     'Citeseer': {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink"}
#     ''
# }
node_label_to_colour_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}


FIGURE_DIR_PATH = os.curdir + f'/figures/neighbourhood_plots/'


#
# Pick the node id you want to visualize the attention for!
#

def draw_neighbour_attention_distribution(graph_labels: torch.Tensor, 
                                        edge_index: torch.Tensor,
                                        attention_weights: List[torch.Tensor], 
                                        dataset_name: str, 
                                        layer_num: int, 
                                        head_num: int, 
                                        show: bool,
                                        node_id=None):

    # These are randomly drawn nodes with degree 10. This is just for comparision purposes.
    node_list = {
        "Cora": [48, 74, 133, 231, 482, 490, 695, 702, 711, 735, 833, 867],    # [2,  1986, 1701, 306, 1358],
        "Citeseer": [567, 620, 709, 865, 1033, 1275, 1759, 1918, 1971, 1981, 2063, 2097], 
        "Pubmed": [407, 555, 831, 872, 884, 912, 926, 966, 1008, 1033, 1098, 1169],
        "PPI": [240, 268, 298, 306, 313, 328, 331, 350, 358, 388]
    }

    if node_id is None:
        # Interesting nodes for each dataset. 
        node_id_list = node_list.get(dataset_name)
    else: 
        node_id_list = [node_id]
    
    # Seperate out the edge index into source and target nodes.
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Useful code atm - # TODO: Remove.
    unique, counts = np.unique(target_nodes.numpy(), return_counts=True)
    node_count = dict(zip(unique, counts))
    ten_valued = [k for (k,v) in node_count.items() if v == 10]    

    # Loop over all the nodes:
    for node_id in node_id_list:

        # Get the neighbour nodes indices for this target_node.
        neighbour_node_indices = torch.eq(target_nodes, node_id)

        # Get the IDs of all the neighbours.
        neighbour_nodes_ids = source_nodes[neighbour_node_indices].cpu().numpy()
        size_of_neighborhood = len(neighbour_nodes_ids)

        # Now we can obtain the attention weights we want by using the layer we are at, alongside the filter for the neighbour indicies and the head number.
        attention_weights_for_neighbours_at_head_at_layer = attention_weights[layer_num][neighbour_node_indices, head_num].cpu().numpy()
        # Normalise these weights as to make the plot consistent in terms of relative attention weight between different edges.
        attention_weights_for_neighbours_at_head_at_layer /= np.max(attention_weights_for_neighbours_at_head_at_layer) 
        # Apply some scaling to give a good visual rep.
        attention_weights_for_neighbours_at_head_at_layer *= (60 / size_of_neighborhood)

        # iGraph requires that the nodes are given contingious ids, from 0 -> n. So in order to do this map the node values into this range. 
        ig_graph = ig.Graph()
        ig_graph.add_vertices(size_of_neighborhood)
        dataset_id_to_igraph_id = dict(zip(neighbour_nodes_ids, range(len(neighbour_nodes_ids))))
        ig_graph.add_edges([(dataset_id_to_igraph_id[node_id], dataset_id_to_igraph_id[neighbor]) for neighbor in neighbour_nodes_ids])

        # Recover the node labels (y values) from the dataset for the neighbours. Use these to create a list of the same length as the number of neighbours + 1 which 
        # are all the colours for the nodes we are going to graph
        neighbour_node_labels = graph_labels[neighbour_nodes_ids].cpu().numpy()
        
        # Plot the graph using the coloring based on which dataset is used. Edge width is the attention weights, and the reingold_tilford circular layout gives a 'star like'
        # layout which makes for each visualisation.
        if show:
            # This is if we want to display the graph and then save it
            if dataset_name != 'PPI':
                cora_node_color_mapping = [node_label_to_colour_map[node_label] for node_label in neighbour_node_labels]
                displayed_graph = ig.plot(ig_graph, edge_width=attention_weights_for_neighbours_at_head_at_layer, layout=ig_graph.layout_reingold_tilford_circular(), vertex_color = cora_node_color_mapping)
            else:
                displayed_graph = ig.plot(ig_graph, edge_width=attention_weights_for_neighbours_at_head_at_layer, layout=ig_graph.layout_reingold_tilford_circular())

            displayed_graph.save(os.path.join(FIGURE_DIR_PATH, dataset_name, 'layer_{}_head_{}_node_{}.png'.format(layer_num, head_num, node_id)))
        else:
            # Save only.
            if dataset_name != 'PPI':
                cora_node_color_mapping = [node_label_to_colour_map[node_label] for node_label in neighbour_node_labels]
                ig.plot(ig_graph, os.path.join(FIGURE_DIR_PATH, dataset_name, 'layer_{}_head_{}_node_{}.png'.format(layer_num, head_num, node_id)), edge_width=attention_weights_for_neighbours_at_head_at_layer, layout=ig_graph.layout_reingold_tilford_circular(), vertex_color = cora_node_color_mapping)
            else:
                ig.plot(ig_graph, os.path.join(FIGURE_DIR_PATH, dataset_name, 'layer_{}_head_{}_node_{}.png'.format(layer_num, head_num, node_id)),edge_width=attention_weights_for_neighbours_at_head_at_layer, layout=ig_graph.layout_reingold_tilford_circular())

