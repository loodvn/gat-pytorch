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


def run(config):
    checkpoint_callback = d_utils.checkpoint(filename=config['dataset']+'-best')

    early_stop_callback = d_utils.early_stop()

    trainer = pl.Trainer(max_epochs=int(config['num_epochs']), callbacks=[checkpoint_callback, early_stop_callback])#, show_progress_bar=True)

    if config['exec_type'] == 'train':
        if config['test_type'] == 'Inductive':
            gat = induGAT(config)
        else:
            gat = transGAT(config)

        trainer.fit(gat)
        trainer.test(gat)
        checkpoint_callback.best_model_path
        trainer.test(gat)
    else:
        try:
            gat = d_utils.load(config)
            trainer.test(gat)
        except FileNotFoundError:
            print("There is no saved checkpoint for this dataset!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--num_epochs') # should be max_epochs
    parser.add_argument('--l2_reg')
    parser.add_argument('--learning_rate')
    # this could throw an error if flag is set
    parser.add_argument('--patience')
    parser.add_argument('--exec_type', default='train') 
    parser.add_argument('--histograms', default=False)

    args = parser.parse_args()
    dataset = args.dataset

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
        run(config)
    except KeyError:
        print("That dataset is invalid")