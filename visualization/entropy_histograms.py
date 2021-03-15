import pickle

# Visualization related imports
import matplotlib.pyplot as plt
import networkx as nx

# Main computation libraries
import scipy.sparse as sp
import numpy as np

# Deep learning related imports
import torch

import os
import enum

import time

from torch_geometric.datasets import CitationFull

from scipy.stats import entropy



# All Cora data is stored as pickle
def pickle_read(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def pickle_save(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')
CORA_PATH = os.path.join(DATA_DIR_PATH, 'cora')

cora_dataset = CitationFull(root='/tmp/Cora', name='Cora')
print(len(cora_dataset))
print(cora_dataset.num_classes)
print(cora_dataset.num_node_features)
print(cora_dataset.num_edge_features)
# Should have 1433 features per node.
print(cora_dataset.num_features)

cora_graph = cora_dataset[0]
print(cora_graph)
print('Test mask: {}'.format(cora_graph.test_mask))
print('Train_mask: {}'.format(cora_graph.train_mask))
print('Val mask: {}'.format(cora_graph.val_mask))

# Cora contains a single, undirected citation graph.
# Contains 2708 nodes, each of which have a class lavel of 0 - 6.
print(cora_graph.x[0])
print('Y values: {}'.format(cora_graph.y))






# # Entropy calculations.

# # gat here is the 4th return from gat forward pass.

# # get the model
# model_state = torch.load(model_path)
# # place back into the GAT architecure we have
# # ToDo.
# # set into eval mode
# gat.load_state_dict(model_state["state_dict"], strict=True)
# gat.eval()

# # Run the predictionas and collect the data.
# with torch.no_grad():
# # Step 3: Run predictions and collect the high dimensional data
# # Unsure about how much of this we need.
# # This is where we actually run the eval, because then we can 
# all_nodes_unnormalized_scores, _ = gat((node_features, edge_index))  # shape = (N, num of classes)

# all_nodes_unnormalized_scores = all_nodes_unnormalized_scores.cpu().numpy()

# num_heads_per_layer = [layer.num_of_heads for layer in gat.gat_net]
# num_layers = len(num_heads_per_layer)

# # self.gat_net = nn.Sequential(*gat_layers,)

# all_attention_weights = gat.gat_net[layer_id].attention_weights.squeeze(dim=-1).cpu().numpy()
# # This has the form: [N, NH] where N is the number of nodes, and H is the number of heads.
# # This should return all attention weights at one layer for all the heads so we need to break up the into the different heads.
# for head_id in range(num_heads_per_layer[layer_id]):
#     # So then we can set up a list to record the entropys
#     uniform_dist_entropy_list = []
#     neighbourhood_entropy_list = []
#     for target_node_id in range(num_of_nodes): # So this would just be the total number of nodes.
#         neigborhood_attention = all_attention_weights[target_node_ids == target_node_id].flatten()
#         ideal_uniform_attention = np.ones(len(neigborhood_attention))/len(neigborhood_attention)


FIGURE_DIR_PATH = f'../figures/attention_histograms/'

def create_single_entropy_histogram(entropy_values, graph_title, color, label, number_of_bins):
    
    # Extract the maximum entropy value for setting the range.
    max_value = np.max(entropy_values)
    bar_width = max_value / number_of_bins
    histogram_values, histogram_bins = np.histogram(entropy_values, bins=number_of_bins, range=(0.0, max_value))
    print(len(histogram_values))
    print(len(histogram_bins))

    plt.bar(histogram_bins[:number_of_bins], histogram_values, width=bar_width, color=color, alpha=0.5, edgecolor=color, label=label)
    plt.xlabel('Node Entropy Value')
    plt.ylabel('Number of Nodes')
    plt.title(graph_title)



def draw_attention_weight_entropy_histogram_pair(attention_entropy_values, uniform_entropy_values, show, dataset, layer_num, head_num):

    number_of_bins = 25
    create_single_entropy_histogram(uniform_entropy_values, graph_title="", color='blue', label='uniform weights', number_of_bins=25)
    create_single_entropy_histogram(attention_entropy_values, graph_title="Attention Weight Entropy Plot: Layer X. Head Y", color='orange', label='attention weights', number_of_bins=25)

    histo_fig = plt.gcf() 
    histo_fig.savefig(os.path.join(FIGURE_DIR_PATH, dataset, f'layer_{layer_num}_head_{head_num}.jpg'))
    if show: plt.show()
    plt.close()




# Get current figure we have been plotting on.

uniform_entropy_values = np.random.rand(100, 1) * 10
attention_entropy_values = uniform_entropy_values + np.random.rand(100, 1) - 0.5

# print("Entropy: {}".format(entropy(uniform_entropy_values, base=2)))


# Save the figure
dataset = 'cora'
layer_id, head_id = 0, 0

draw_attention_weight_entropy_histogram_pair(attention_entropy_values, uniform_entropy_values, True, dataset, layer_id, head_id)
