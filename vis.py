import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models.gat_layer import GATLayer
from models.trans_gat import transGAT
from models.indu_gat import induGAT
from run_config import data_config
import data_utils as d_utils
from torch_geometric.datasets import Planetoid, PPI
from visualisation.neighbourhood_attention_weights import draw_neighbour_attention_distribution
from visualisation.entropy_histograms import draw_entropy_histrogram



def load_model(config, file_name_ending):
    """
    Args:
        config (dict): containing useful info such as the test type, which is what style of model this is.

    Returns:
        pl.LightningModule : model which has had it's weights loaded bcak from a checkpoint.
    """

    if config['test_type'] == 'Inductive':
        gat_model = induGAT(config)
    else:
        gat_model = transGAT(config)
    
    # TODO: Change to what we do decide to name this.
    gat_model = d_utils.load(config, file_name_ending)

    return gat_model




def run_neighbourhood_attention_distribution(config, file_name_ending):
    
    try:
        gat_model = load_model(config, file_name_ending)
        test_data = get_test_data(config['dataset'])
        batch = next(iter(test_data))
        outputs, edge_index, attention_list = gat_model.forward_and_return_attention(batch, return_attention_coeffs=True)
        
        # print("Size of outputs from GatModel. \nOutput: {}. Edge Index: {}. Attention List: {}.".format(outputs.size(), edge_index.size(), attention_list))
        # print("Attention List tensor sizes: ")
        # for att in attention_list:
        #     print("Attention size: {}".format(att.shape))
        
        draw_neighbour_attention_distribution(graph_labels=batch.y, edge_index=edge_index, attention_weights=attention_list, dataset_name=config['dataset'], layer_num=0, head_num=0, show=False)

    except FileNotFoundError:
        print("There is no saved checkpoint for this dataset!")         




def run_entropy_histrogram(config, file_name_ending):

    try:
        gat_model = load_model(config, file_name_ending)
        test_data = get_test_data(config['dataset'])
        batch = next(iter(test_data))
        outputs, edge_index, attention_list = gat_model.forward_and_return_attention(batch, return_attention_coeffs=True)

        # print("Size of outputs from GatModel. \nOutput: {}. Edge Index: {}. Attention List: {}.".format(outputs.size(), edge_index.size(), attention_list))
        # print("Attention List tensor sizes: ")
        # for att in attention_list:
        #     print("Attention size: {}".format(att.shape))
        draw_entropy_histrogram(edge_index=edge_index, attention_weights=attention_list, num_nodes=batch.x.size()[0], dataset_name=config['dataset'])
    except FileNotFoundError:
        print("There is no saved checkpoint for this dataset!")    



def get_test_data(dataset_name):
    # Quick and easy function to get the data we require for a test run to get the attention weights.
    if dataset_name == 'Cora' or dataset_name == 'Citeseer' or dataset_name == 'Pubmed':
        return DataLoader(Planetoid(root='/tmp/' + dataset_name, name=dataset_name))
    else:
        return DataLoader(PPI(root='/tmp/PPI', split='test'))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--vis_type', default='Entropy')
    
    args = parser.parse_args()
    dataset = args.dataset

    file_name_ending = "-100epochs.ckpt"

    try:
        config = data_config[dataset]
        for arg in vars(args):
            val = getattr(args, arg)
            if val != None:
                # could this cause issue with booleans??
                if d_utils.is_number(val):
                    config[arg] = float(val)
                else:
                    config[arg] = val


        if config['vis_type'] == "Entropy":
            run_entropy_histrogram(config, file_name_ending)
        else: 
            run_neighbourhood_attention_distribution(config, file_name_ending)

    except KeyError:
        print("That dataset is invalid")