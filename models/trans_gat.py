import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
from models.gat_layer import GATLayer

# TODO add logging, e.g. tensorboard
class transGAT(pl.LightningModule):
    def __init__(self, dataset, node_features, num_classes, in_heads=8, out_heads=1, head_features=8, l2_reg=0.0005, lr = 0.005, dropout=0.6):
        super(transGAT, self).__init__()
        self.dataset = dataset

        self.head_features = head_features
        self.in_heads = in_heads
        self.out_heads = out_heads

        self.lr = lr
        self.l2_reg = l2_reg
        self.dropout = dropout

        self.node_features = node_features
        self.num_classes = num_classes

        self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.in_heads, add_self_loops=True, dropout=self.dropout) # add self loops? GATLayer(in_channels=self.node_features, out_channels=self.head_features, number_of_heads=self.in_heads, dropout=self.dropout, alpha = 0.2)#
        self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, heads=out_heads, concat=False, dropout=self.dropout) # GATLayer(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, number_of_heads=self.out_heads, concat=False, dropout=self.dropout, alpha = 0.2)#

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # dropout to avoid overfitting as dataset is small
        x = F.dropout(x, p=self.dropout)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1) # think paper only mentions normal softmax? - this is fine, oli says for speed increase


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        # This is minimising cross entropy right?
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        # loss = torch.nn.CrossEntropyLoss(out[batch.train_mask], batch.y[batch.train_mask])
        print(loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     out = self(batch)
    #     pred = out.argmax(dim=1)
    #     correct = float (pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item())
    #     val_acc = (correct / batch.test_mask.sum().item())
    #     # This is minimising cross entropy right?
    #     val_loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
    #     # loss = torch.nn.CrossEntropyLoss(out[batch.train_mask], batch.y[batch.train_mask])
    #     print(val_loss)
    #     return loss

    def test_step(self, batch, batch_idx):
        # Copied from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs
        out = self(batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        # TODO change to torch accuracy metric
        test_correct = pred[batch.test_mask] == batch.y[batch.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(batch.test_mask.sum())  # Derive ratio of correct predictions.

        # correct = float (pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item())
        # test_acc = (correct / batch.test_mask.sum().item())
        # print("correct ", correct)
        self.log('test_acc', test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print("This is the test accuracy")
        print(test_acc)
        return test_acc

    

    # How to apply mask at this stage so we arent loading entire dataset twice? - or do we accc want that
    def train_dataloader(self):
        dataset = Planetoid(root='/tmp/' + self.dataset, name=self.dataset)     
        # print(data.train_mask.sum().item())
        # print(data.val_mask.sum().item())
        # print(data.test_mask.sum().item())   
        return DataLoader(dataset)        
        

    def test_dataloader(self):
        dataset = Planetoid(root='/tmp/' + self.dataset, name=self.dataset)        
        return DataLoader(dataset)        