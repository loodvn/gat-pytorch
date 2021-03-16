import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from models.gat_layer import GATLayer
from models.trans_gat import transGAT
from models.indu_gat import induGAT


def train(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads):
    if dataset == 'PPI':
        gat = induGAT()
    else:
        gat = transGAT(dataset=dataset, node_features=node_features, num_classes=num_classes, out_heads=out_heads, lr=learning_rate, l2_reg=l2_reg)

    trainer = pl.Trainer(max_epochs=max_epochs)
    
    trainer.fit(gat)

    trainer.test()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--max_epochs', default=100)
    parser.add_argument('--l2', default=0.0005)
    parser.add_argument('--lr', default=0.005)

    parser.add_argument('--histograms', default=False)
    parser.add_argument('--save', default=True)

    args = parser.parse_args()
    
    dataset = args.dataset
    max_epochs = args.max_epochs
    # task_type = 'transductive'
    learning_rate = args.lr
    l2_reg = args.l2

    out_heads = 1

    if dataset == 'Cora':
        node_features = 1433
        num_classes = 7
    elif dataset == 'Citeseer':
        node_features = 3703
        num_classes = 6  
    elif dataset == 'Pubmed':
        node_features = 500
        num_classes = 3
        learning_rate = 0.01
        l2_reg = 0.001
        out_heads = 8
    elif dataset == 'PPI':
        node_features = 50
        num_classes = 121
        # task_type = 'inductive'
    else:
        sys.exit('Dataset is invalid')
    
    train(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads)
    
