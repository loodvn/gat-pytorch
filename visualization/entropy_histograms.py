import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from torch_geometric.datasets import CitationFull
from scipy.stats import entropy
from torch_geometric.utils import add_self_loops
import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
import typing
from typing import List
from utils import GATCora, load_cora_data, train_cora_ow, prep_model_and_data_for_analysis

FIGURE_DIR_PATH = f'../figures/attention_histograms/'

def create_single_entropy_histogram(entropy_values: List[float], 
                                    graph_title: str, 
                                    color: str, 
                                    label: str, 
                                    number_of_bins: int):
    
    # Extract the maximum entropy value for setting the range.
    max_value = np.max(entropy_values)
    bar_width = max_value / number_of_bins
    histogram_values, histogram_bins = np.histogram(entropy_values, bins=number_of_bins, range=(0.0, max_value))

    if color == 'blue':
        plt.bar(histogram_bins[:number_of_bins], histogram_values, width=bar_width, color=color, alpha=1, edgecolor='darkblue', label=label)
    else:
        plt.bar(histogram_bins[:number_of_bins], histogram_values, width=bar_width * 0.75, color=color, alpha=1, edgecolor='darkorange', label=label)

    plt.xlabel('Node Entropy Value')
    plt.ylabel('Number of Nodes')
    plt.title(graph_title)



def create_attention_weight_dual_entropy_histogram(attention_entropy_values: List[float], 
                                                uniform_entropy_values: List[float],
                                                dataset_name: str, 
                                                layer_num: int, 
                                                head_num: int, 
                                                show: bool,
                                                save: bool):

    number_of_bins = 25
    create_single_entropy_histogram(uniform_entropy_values, graph_title="Attention Weight Entropy Plot: Layer {}. Head {}.".format(layer_num, head_num), color='blue', label='uniform weights', number_of_bins=number_of_bins)
    create_single_entropy_histogram(attention_entropy_values, graph_title="Attention Weight Entropy Plot: Layer {}. Head {}.".format(layer_num, head_num), color='orange', label='attention weights', number_of_bins=number_of_bins)

    histo_fig = plt.gcf() 
    if save: histo_fig.savefig(os.path.join(FIGURE_DIR_PATH, dataset_name, f'layer_{layer_num}_head_{head_num}.jpg'))
    if show: 
        plt.show()
        plt.close()


def run_entropy_histrogram(graph: torch_geometric.data.data.Data, 
                           attention_weights: List[torch.Tensor]):

    # In our implementation we use the 'target node' that which is in the position[1] of the edge_index to perform the softmax normalisation on, so in order to be consistent we use the same here
    source_nodes = graph.edge_index[0]
    target_nodes = graph.edge_index[1]

    # Need the edge index for the filtering on the entropy calculations.
    for layer in range(0,len(attention_weights)):
        
        # TODO - Make sure we have the data stacked.
        attention_weights_for_layer = attention_weights[layer]
        num_of_heads = attention_weights_for_layer.shape[1]
        num_nodes = graph.x.shape[0]

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
            create_attention_weight_dual_entropy_histogram(neighbourhood_entropy_list, uniform_dist_entropy_list, dataset_name='cora', layer_num=layer, head_num=head, show=True, save=True)




if __name__ == "__main__":
    cora_graph = load_cora_data()
    attention_weights = train_cora_ow()
    run_entropy_histrogram(cora_graph, attention_weights)

