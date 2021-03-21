import argparse

from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid, PPI

import data_utils
from run_config import data_config
from visualisation.entropy_histograms import draw_entropy_histogram
from visualisation.neighbourhood_attention_weights import draw_neighbour_attention_distribution


def run_neighbourhood_attention_distribution(config, file_name_ending):
    gat_model = data_utils.load(config, file_name_ending)
    test_data = get_test_data(config['dataset'])
    batch = next(iter(test_data))
    outputs, edge_index, attention_list = gat_model.forward_and_return_attention(batch,
                                                                                 return_attention_coeffs=True)

    # print("Size of outputs from GatModel. \nOutput: {}. Edge Index: {}. Attention List: {}.".format(outputs.size(), edge_index.size(), attention_list))
    # print("Attention List tensor sizes: ")
    # for att in attention_list:
    #     print("Attention size: {}".format(att.shape))

    draw_neighbour_attention_distribution(graph_labels=batch.y, edge_index=edge_index,
                                          attention_weights=attention_list, dataset_name=config['dataset'],
                                          layer_num=0, head_num=0, show=False)


def run_entropy_histrogram(config, file_name_ending):
    gat_model = data_utils.load(config, file_name_ending)
    test_data = get_test_data(config['dataset'])
    batch = next(iter(test_data))
    outputs, edge_index, attention_list = gat_model.forward_and_return_attention(batch,
                                                                                 return_attention_coeffs=True)

    # print("Size of outputs from GatModel. \nOutput: {}. Edge Index: {}. Attention List: {}.".format(outputs.size(), edge_index.size(), attention_list))
    # print("Attention List tensor sizes: ")
    # for att in attention_list:
    #     print("Attention size: {}".format(att.shape))
    draw_entropy_histogram(edge_index=edge_index, attention_weights=attention_list, num_nodes=batch.x.size()[0],
                           dataset_name=config['dataset'])


def get_test_data(dataset_name):
    # Quick and easy function to get the data we require for a test run to get the attention weights.
    if dataset_name == 'Cora' or dataset_name == 'Citeseer' or dataset_name == 'Pubmed':
        return DataLoader(Planetoid(root='/tmp/' + dataset_name, name=dataset_name))
    else:
        return DataLoader(PPI(root='/tmp/PPI', split='test'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--vis_type', default='Entropy')  # Entropy or Neigh
    
    args = parser.parse_args()
    dataset = args.dataset

    file_name_ending = "-100epochs.ckpt"

    if dataset not in data_config.keys():
        print(f"Dataset not valid. Must be one of {data_config.keys()}. {dataset} given.")
    else:
        config = data_config[dataset]
        di = {k: v for k, v in args.__dict__.items() if v is not None}

        config.update(di)
        vis_type = config.pop('vis_type')
        if vis_type == "Entropy":
            run_entropy_histrogram(config, file_name_ending)
        else:
            run_neighbourhood_attention_distribution(config, file_name_ending)
