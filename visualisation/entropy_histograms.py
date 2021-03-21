import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import entropy
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import typing
from typing import List
import torch

FIGURE_DIR_PATH = os.curdir + f'/figures/entropy_histograms/'

def create_single_entropy_histogram(entropy_values: List[float], 
                                    graph_title: str, 
                                    color: str, 
                                    label: str, 
                                    number_of_bins: int, 
                                    transductive: bool):
    
    # Extract the maximum entropy value for setting the range.
    max_value = np.max(entropy_values)
    bar_width = max_value / number_of_bins
    histogram_values, histogram_bins = np.histogram(entropy_values, bins=number_of_bins, range=(0.0, max_value))

    if color == 'darkred':
        plt.bar(histogram_bins[:number_of_bins], histogram_values, width=bar_width, color=color, alpha=0.6, edgecolor="none", label=label)
        # darkblue
    else:
        if transductive:
            # To account for the bars being the same shape.
            plt.bar(histogram_bins[:number_of_bins], histogram_values, width=bar_width * 0.75, color=color, alpha=0.6, edgecolor="none", label=label)
        else:
            plt.bar(histogram_bins[:number_of_bins], histogram_values, width=bar_width, color=color, alpha=0.6, edgecolor="none", label=label)
        plt.legend(loc="upper right")
        # dark orange

    plt.xlabel('Node Entropy Value')
    plt.ylabel('Number of Nodes')
    plt.title(graph_title)



def create_attention_weight_dual_entropy_histogram(attention_entropy_values: List[float], 
                                                uniform_entropy_values: List[float],
                                                dataset_name: str, 
                                                layer_num: int, 
                                                head_num: int, 
                                                show: bool,
                                                save: bool,
                                                transductive: bool):

    number_of_bins = 25
    create_single_entropy_histogram(uniform_entropy_values, graph_title="Attention Weight Entropy Plot: Layer {}. Head {}.".format(layer_num, head_num), color='darkred', label='Uniform Weights', number_of_bins=number_of_bins, transductive=transductive)
    create_single_entropy_histogram(attention_entropy_values, graph_title="Attention Weight Entropy Plot: Layer {}. Head {}.".format(layer_num, head_num), color='green', label='Attention Weights', number_of_bins=number_of_bins, transductive=transductive)

    histo_fig = plt.gcf() 
    if save: histo_fig.savefig(os.path.join(FIGURE_DIR_PATH, dataset_name, f'layer_{layer_num}_head_{head_num}.jpg'))
    if show: 
        plt.show()
    plt.close()


def draw_entropy_histrogram(edge_index: torch.Tensor,
                            attention_weights: List[torch.Tensor],
                            num_nodes: int, 
                            dataset_name: str):

    # In our implementation we use the 'target node' that which is in the position[1] of the edge_index to perform the softmax normalisation on, so in order to be consistent we use the same here
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    # Need the edge index for the filtering on the entropy calculations.
    for layer in range(0,len(attention_weights)):
        
        attention_weights_for_layer = attention_weights[layer]
        num_of_heads = attention_weights_for_layer.shape[1]

        for head in range(num_of_heads):
            # Filter the attention weights to those for the head in question.
            attention_weights_for_head = attention_weights_for_layer[:, head]

            # So then we can set up a list to record the entropys
            uniform_dist_entropy_list = []
            neighbourhood_entropy_list = []

            for node_id in range(0, num_nodes):
                # Find all the places in which that node occurs as the target node, and use this as a filter for the attention weights
                weights_for_node_output = attention_weights_for_head[target_nodes == node_id]
                # These values sum to 1, so we can treat as a probablity dist and can therefore calculate the entropy on the values.
                neighbourhood_entropy_list.append(entropy(weights_for_node_output, base=2))
                # For a comparision we add the uniform distribution over the neigbours by assigning each weight to be 1 / (# of neighbours)
                uniform_dist_entropy_list.append(entropy(np.ones(len(weights_for_node_output))/len(weights_for_node_output), base=2))

            # Call the attention mechanism. 
            create_attention_weight_dual_entropy_histogram(neighbourhood_entropy_list, uniform_dist_entropy_list, dataset_name=dataset_name, layer_num=layer, head_num=head, show=False, save=True, transductive=(dataset_name != 'PPI'))




