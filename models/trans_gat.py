import argparse
import sys

import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

from models.gat_layer import GATLayer

# TODO improve logging, e.g. tensorboard
# TODO validation
# TODO loading correctly
# pl.seed_everything(42)


class transGAT(pl.LightningModule):
    def __init__(self, dataset, node_features, num_classes, in_heads=8, out_heads=1, head_features=8, l2_reg=0.0005, lr = 0.005, dropout=0.6):
        super(transGAT, self).__init__()
        self.dataset_name = dataset

        self.head_features = head_features
        self.in_heads = in_heads
        self.out_heads = out_heads

        self.lr = lr
        self.l2_reg = l2_reg
        self.dropout = 0 #dropout

        self.node_features = node_features
        self.num_classes = num_classes

        # self.patience = 100
        # self.lossMin = 1e10
        # print(self.lossMin)
        # print(self.dataset)
        # print(self.head_features)
        # print(self.in_heads)
        # print(self.out_heads)
        # print(self.lr)
        # print(self.l2_reg)
        # print(self.dropout)
        # print(self.node_features)
        # print(self.num_classes)

        # self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.in_heads, dropout=self.dropout, bias=False, add_self_loops=False)
        self.gat1 = GATLayer(in_features=self.node_features, out_features=self.head_features, num_heads=self.in_heads, concat=True, dropout=self.dropout, bias=False, add_self_loops=False, const_attention=False)
        # self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, heads=out_heads, concat=False, dropout=self.dropout, bias=False, add_self_loops=False)
        self.gat2 = GATLayer(in_features=self.head_features * self.in_heads, out_features=self.num_classes, num_heads=out_heads, concat=False, dropout=self.dropout, bias=False, add_self_loops=False, const_attention=False)
        # print(self.gat1)
        # print(self.gat2)

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Dropout is applied within the layer
        x = self.gat1(x, edge_index)
        x = nn.ELU()(x)
        x = self.gat2(x, edge_index)
        # Returning raw logits
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)  # There's only one step in epoch so we log on epoch
        # TODO log histogram of attention weights?
        return loss

    def on_train_end(self):
        print("tmp attention coeffs: ", self.gat1.normalised_attention_coeffs)
        print("tmp attention coeffs mean, std, max: ", self.gat1.normalised_attention_coeffs.mean(), self.gat1.normalised_attention_coeffs.std(), self.gat1.normalised_attention_coeffs.max())

    def validation_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        val_loss = self.criterion(out[batch.val_mask], batch.y[batch.val_mask])

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = float (pred[batch.val_mask].eq(batch.y[batch.val_mask]).sum().item())
        val_acc = (correct / batch.val_mask.sum().item())

        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Copied from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs
        out = self(batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        
        test_correct = pred[batch.test_mask] == batch.y[batch.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(batch.test_mask.sum())  # Derive ratio of correct predictions.

        self.log('test_acc', test_acc, on_epoch=True, prog_bar=True, logger=True)
        print("This is the test accuracy")
        print(test_acc)
        return test_acc

    def prepare_data(self):
        self.dataset = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)

    # Transductive: Load whole graph, mask out when calculating loss
    def train_dataloader(self):
        return DataLoader(self.dataset)
        
    def val_dataloader(self):
        return DataLoader(self.dataset)

    def test_dataloader(self):
        return DataLoader(self.dataset)
