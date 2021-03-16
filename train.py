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


def train(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads, save):
    if dataset == 'PPI':
        gat = induGAT()
    else:
        gat = transGAT(dataset=dataset, node_features=node_features, num_classes=num_classes, out_heads=out_heads, lr=learning_rate, l2_reg=l2_reg)

    trainer = pl.Trainer(max_epochs=max_epochs)
    
    trainer.fit(gat)

    trainer.test()

    if save:
        trainer.save_checkpoint(dataset + ".ckpt")
    
def load(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads):
    print('I am loading :)')
    # Will need to change this as loaded dataset could be an induGAT
    # loaded_model = transGAT(dataset=dataset, node_features=node_features, num_classes=num_classes, lr=learning_rate, l2_reg=l2_reg, out_heads=out_heads).load_from_checkpoint(checkpoint_path=dataset+".ckpt")
    # loaded_model = transGAT.load_from_checkpoint(checkpoint_path=dataset+".ckpt", dataset=dataset, node_features=node_features, num_classes=num_classes)
    loaded_model = transGAT(dataset=dataset, node_features=node_features, num_classes=num_classes)
    trainer = pl.Trainer(resume_from_checkpoint=dataset+".ckpt")
    # trainer = pl.Trainer()
    # trainer.fit(loaded_model)
    trainer.test(loaded_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read in dataset and any other flags from command line')
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--max_epochs', default=100)
    parser.add_argument('--l2', default=0.0005)
    parser.add_argument('--lr', default=0.005)
    parser.add_argument('--exec_type', default='train') # could also be load

    parser.add_argument('--histograms', default=False)
    parser.add_argument('--save', default=False)

    args = parser.parse_args()
    
    dataset = args.dataset
    max_epochs = args.max_epochs
    learning_rate = args.lr
    l2_reg = args.l2

    out_heads = 1

    print(args.save)

    
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
    else:
        sys.exit('Dataset is invalid')

    if args.exec_type == 'train':
        train(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads, args.save)

    elif args.exec_type == 'load':
        load(dataset, node_features, num_classes, max_epochs, learning_rate, l2_reg, out_heads)
    
