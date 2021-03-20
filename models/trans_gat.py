import argparse
import sys

import pytorch_lightning as pl
import torch
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from .gat_layer import GATLayer
import torch.nn.functional as F
from models.GATModel import GATModel


# pl.seed_everything(42)


class transGAT(GATModel):
    def __init__(self, config):
        super().__init__(config)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def training_step(self, batch, batch_idx): 
        out = self(batch)
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)  # There's only one step in epoch so we log on epoch
        # TODO log histogram of attention weights?
        return loss

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
        # out, edge_index, attention_list = self.forward_and_return_attention(batch, return_attention_coeffs=True)
        out = self(batch)
        # self.attention_weights_list = attention_list
        # OW: TODO - Use these attention weights.
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        
        test_correct = pred[batch.test_mask] == batch.y[batch.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(batch.test_mask.sum())  # Derive ratio of correct predictions.

        self.log('test_acc', test_acc, on_epoch=True, prog_bar=True, logger=True)
        print("This is the test accuracy")
        print(test_acc)
        return test_acc

    # Transductive: Load whole graph, mask out when calculating loss
    def prepare_data(self):
        self.train_ds = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)
        self.val_ds = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)
        self.test_ds = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)
