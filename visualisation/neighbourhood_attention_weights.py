''' Script for plotting the attention weights within a neighbourhood of a node. In this script with use python-iGraph as our graphing backend.
The inspiration for this script is taken from https://github.com/gordicaleksa/pytorch-GAT/blob/main/The%20Annotated%20GAT%20(PPI).ipynb.
The author (Aleksa Gordic) produces similar plots, using iGraph, but only for Cora and PPI, and only for several selective nodes. We extend
this work by considering a random collection of degree 10 nodes, and investigate how the attention weights differ across these '''

import os
from typing import List
import igraph as ig
import numpy as np
import torch

# Colour map which is used for single label classification datasets to allow the viewer to see the different classes of the nodes.
node_label_to_colour_map = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "yellow", 5: "pink", 6: "gray"}
FIGURE_DIR_PATH = os.curdir + f'/figures/neighbourhood_plots/'


def draw_neighbour_attention_distribution(graph_labels: torch.Tensor,
                                          edge_index: torch.Tensor,
                                          attention_weights: List[torch.Tensor],
                                          dataset_name: str,
                                          layer_num: int,
                                          head_num: int,
                                          show: bool,
                                          save: bool,
                                          node_id=None):
    # These are randomly drawn nodes with degree 10. This is just for comparison purposes.
    node_list = {
        "Cora": [48, 74, 133, 231, 482, 490, 695, 702, 711, 735, 833, 867],
        "Citeseer": [567, 620, 709, 865, 1033, 1275, 1759, 1918, 1971, 1981, 2063, 2097],
        "Pubmed": [407, 555, 831, 872, 884, 912, 926, 966, 1008, 1033, 1098, 1169],
        "PPI": [240, 268, 298, 306, 313, 328, 331, 350, 358, 388],
        "PATTERN": range(10),
    }

    if node_id is None:
        node_id_list = node_list.get(dataset_name)
    else:
        node_id_list = [node_id]

    # Separate out the edge index into source and target nodes.
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

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
            planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
            if dataset_name in planetoid_datasets + ['PATTERN']:
                cora_node_color_mapping = [node_label_to_colour_map[node_label] for node_label in neighbour_node_labels]
                displayed_graph = ig.plot(ig_graph, edge_width=attention_weights_for_neighbours_at_head_at_layer,
                                          layout=ig_graph.layout_reingold_tilford_circular(),
                                          vertex_color=cora_node_color_mapping)
            else:
                displayed_graph = ig.plot(ig_graph, edge_width=attention_weights_for_neighbours_at_head_at_layer,
                                          layout=ig_graph.layout_reingold_tilford_circular())

            displayed_graph.save(os.path.join(FIGURE_DIR_PATH, dataset_name,
                                              'layer_{}_head_{}_node_{}.png'.format(layer_num, head_num, node_id)))
        if save or not show:
            # Save only.
            # Create intermediate directories if they do not exist. Could extract this out to a utils file.
            if not os.path.isdir(FIGURE_DIR_PATH):
                os.mkdir(FIGURE_DIR_PATH)
            save_dir = os.path.join(FIGURE_DIR_PATH, dataset_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

            planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
            if dataset_name in planetoid_datasets + ['PATTERN']:
                cora_node_color_mapping = [node_label_to_colour_map[node_label] for node_label in neighbour_node_labels]
                ig.plot(ig_graph, os.path.join(FIGURE_DIR_PATH, dataset_name,
                                               'layer_{}_head_{}_node_{}.png'.format(layer_num, head_num, node_id)),
                        edge_width=attention_weights_for_neighbours_at_head_at_layer,
                        layout=ig_graph.layout_reingold_tilford_circular(), vertex_color=cora_node_color_mapping)
            elif dataset_name == 'PPI':
                ig.plot(ig_graph, os.path.join(FIGURE_DIR_PATH, dataset_name,
                                               'layer_{}_head_{}_node_{}.png'.format(layer_num, head_num, node_id)),
                        edge_width=attention_weights_for_neighbours_at_head_at_layer,
                        layout=ig_graph.layout_reingold_tilford_circular())
            else:
                raise ValueError(f"Dataset name not valid. Expected one of {planetoid_datasets}/PPI/PATTERN")
