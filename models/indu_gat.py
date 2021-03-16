import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from models.gat_layer import GATLayer

class induGAT(pl.LightningModule):
    def __init__(self, dataset, node_features, num_classes, in_heads=4, out_heads=6, head_features=8, l2_reg=0.0005, dropout=0.6):
        super(transGAT, self).__init__()
        self.dataset = dataset

        # From GAT paper, Section 3.3
        self.head_features = head_features
        self.in_heads = in_heads
        self.out_heads = out_heads

        self.l2_reg = l2_reg
        self.dropout = dropout

        # These are cora specific and shouldnt be explicitly declared
        self.node_features = node_features
        self.num_classes = num_classes

        # Is out for layer 1 correct?
        self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.in_heads, add_self_loops=True, dropout=self.dropout) # add self loops?
        self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, heads=out_heads, concat=False, dropout=self.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # dropout to avoid overfitting as dataset is small
        x = F.dropout(x, p=self.dropout)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1) # think paper only mentions normal softmax? - this is fine, ollie says for speed increase   
