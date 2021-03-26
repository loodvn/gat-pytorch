''' 
Runner script for visualisation.
This can perform graphing of the neighbours with edge width in proportion to the attention coeffs, entropy histograms for the dist of attention weights, and then
also plotting the normalised weights as a histogram.
'''

import argparse
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, PPI
import data_utils
from run_config import data_config
from visualisation.entropy_histograms import draw_entropy_histogram
from visualisation.neighbourhood_attention_weights import draw_neighbour_attention_distribution
from visualisation.weight_histograms import draw_weights_histogram


def get_test_data(dataset_name):
    # Quick and easy function to get the data we require for a test run to get the attention weights.
    if dataset_name == 'Cora' or dataset_name == 'Citeseer' or dataset_name == 'Pubmed':
        return DataLoader(Planetoid(root='/tmp/' + dataset_name, name=dataset_name))
    else:
        return DataLoader(PPI(root='/tmp/PPI', split='test'))


def main(dataset, vis_type, checkpoint_path=None):
    # Used to run for each dataset the epoch 100 checkpoint, which should give a good level of accuracy / F1 score.
    default_file_name_ending = "-100epochs.ckpt"
    if dataset not in data_config.keys():
        print(f"Dataset not valid. Must be one of {data_config.keys()}. {dataset} given.")
    else:
        config = data_config[dataset]
        config['test_type'] = 'Inductive' if dataset == 'PPI' else "Transductive"

        # Load the model, get the test data and prepare a test batch and then make a call to the forwards function.
        gat_model = data_utils.load(config, default_file_name_ending, checkpoint_path=checkpoint_path)

        test_data = get_test_data(config['dataset'])
        batch = next(iter(test_data))
        outputs, edge_index, attention_list = gat_model.forward_and_return_attention(batch,
                                                                                     return_attention_weights=True)
        # Detach tensors so that we can use them
        attention_list = [att.detach().cpu() for att in attention_list]

        if vis_type == "Entropy":
            draw_entropy_histogram(edge_index=edge_index, attention_weights=attention_list, num_nodes=batch.x.size()[0],
                                   dataset_name=config['dataset'])
        elif vis_type == "Neighbourhood":
            draw_neighbour_attention_distribution(graph_labels=batch.y, edge_index=edge_index,
                                                  attention_weights=attention_list, dataset_name=config['dataset'],
                                                  layer_num=0, head_num=0, show=True)
        elif vis_type == "Weight":
            if config['dataset'] == "PPI":
                epochs_recorded = [1, 5, 10, 20, 50, 100]
                for epoch_number in epochs_recorded:
                    # We need to load up the different modules in succesion for the different. Once this is loaded we complete a forward pass on a batch of then
                    # test data and plot the weight histogram for these.
                    file_ending = "-" + str(epoch_number) + "epochs.ckpt"
                    gat_model = data_utils.load(config, file_ending)
                    outputs, edge_index, attention_list = gat_model.forward_and_return_attention(batch,
                                                                                                 return_attention_weights=True)
                    draw_weights_histogram(edge_index=edge_index, attention_weights=attention_list,
                                           num_nodes=batch.x.size()[0], epoch_number=epoch_number,
                                           dataset_name=config['dataset'])
            else:
                file_ending = "-100epochs.ckpt"
                gat_model = data_utils.load(config, file_ending)
                outputs, edge_index, attention_list = gat_model.forward_and_return_attention(batch,
                                                                                             return_attention_weights=True)
                draw_weights_histogram(edge_index=edge_index, attention_weights=attention_list,
                                       num_nodes=batch.x.size()[0], epoch_number=100,
                                       dataset_name=config['dataset'])
        else:
            raise Exception(
                "Unknown visualisation type. Please use one of 'Entropy', 'Weight (only for PPI)' or 'Neighbourhood'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--vis_type', default='Entropy')  # Entropy or Neighbour or Weight.
    parser.add_argument('--checkpoint_path')

    args = parser.parse_args()
    dataset = args.dataset

    main(dataset, args.vis_type, args.checkpoint_path)
