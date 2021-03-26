''' Visualization script for looking at the normalised weights learnt by the attention mechanism. We take the attention weights, scale them by their degree 
and then plot these onto a histogram and compare with a uniform dist. '''

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch



FIGURE_DIR_PATH = os.curdir + f'/figures/weight_histograms'

def create_attention_weight_dual_histogram(attention_weight_values: List[float],
                                            uniform_weight_values: List[float],
                                            dataset_name: str,
                                            layer_num: int,
                                            head_num: int,
                                            epoch: int,
                                            show: bool,
                                            save: bool,
                                            transductive: bool):

    # Plot the histogram on the same axis using a logarithmic scale on the y axis.
    low_range, high_range = (0, 5) if dataset_name == "PPI" else (0.5, 1.5)
    plt.hist(x=attention_weight_values, bins=20, range=[low_range, high_range], color='green', alpha=0.7, label='Attention Weights')
    if dataset_name == "PPI": plt.yscale('log')
    plt.hist(x=uniform_weight_values, bins=20, range=[low_range, high_range], color='darkred', alpha=0.7, label='Uniform Weights')
    if dataset_name == "PPI": plt.yscale('log')
    plt.legend()
    plt.title("Attention Weight Plot: Epochs: {}. Layer {}. Head {}.".format(epoch, layer_num, head_num))
    plt.xlabel("Attention Weight")
    plt.ylabel("Log Frequency") if dataset_name == "PPI" else plt.ylabel("Frequency")


    histo_fig = plt.gcf()
    if save:
        # Create intermediate directories if they do not exist. Could extract this out to a utils file.
        if not os.path.isdir(FIGURE_DIR_PATH):
            os.mkdir(FIGURE_DIR_PATH)
        save_dir = os.path.join(FIGURE_DIR_PATH, dataset_name)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        histo_fig.savefig(os.path.join(FIGURE_DIR_PATH, dataset_name, f'epoch_'+str(epoch), f'layer_{layer_num}_head_{head_num}.jpg'))
    if show:
        plt.show()
    plt.close()


def draw_weights_histogram(edge_index: torch.Tensor,
                           attention_weights: List[torch.Tensor],
                           epoch_number: int,
                           num_nodes: int,
                           dataset_name: str,
                           save: bool):
    # In our implementation we use the 'target node' that which is in the position[1] of the edge_index to perform the softmax normalisation on, so in order to be consistent we use the same here
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    for layer in range(0, len(attention_weights)): # len(attention_weights)):

        attention_weights_for_layer = attention_weights[layer]
        num_of_heads = attention_weights_for_layer.shape[1]

        # Just using head 0, can extend for the others.        
        for head in range(0,1): # range(num_of_heads):
            # Filter the attention weights to those for the head in question.
            attention_weights_for_head = attention_weights_for_layer[:, head]

            # So then we can set up a list plot the histograms with.
            neighbourhood_weights = []

            for node_id in range(0, num_nodes):
                # Find all the places in which that node occurs as the target node, and use this as a filter for the attention weights

                # OW / LVN - Change to see if we multiple by the degree whether this makes the vis any cleaner. 
                # IDEA: Before we were accounting for the different degrees by using a histogram, which highlights any aggregated differences in the neighbourhood distributions.
                # In this, we instead scale the weights by degree, meaning that initially we expected to see all 1's, because we have softmaxed over the degree, only to multiple by it.
                # But, as the training continutes we should see this changes to a histogram more skewed towards 0.
                weights_for_node_output = attention_weights_for_head[target_nodes == node_id] * attention_weights_for_head[target_nodes == node_id].shape[0]
                weights_for_node_output = weights_for_node_output.detach().cpu().numpy()

                # Collect all of these.
                for weight in weights_for_node_output:
                    if weight < 5:
                        neighbourhood_weights.append(weight)
            
            # Reflect the uniform dist.
            uniform_dist_weights = [1 for i in neighbourhood_weights]

            print("Len of uniform: {}. Len of attention: {}".format(len(uniform_dist_weights), len(neighbourhood_weights)))

            # Call the histogram plotting tool. 
            create_attention_weight_dual_histogram(neighbourhood_weights, uniform_dist_weights,
                                                   dataset_name=dataset_name, epoch=epoch_number, layer_num=layer, head_num=head,
                                                   show=True, save=save, transductive=(dataset_name != 'PPI'))
