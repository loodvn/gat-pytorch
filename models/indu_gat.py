import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from torch.nn import Sigmoid
import pytorch_lightning as pl
from models.gat_layer import GATLayer

class induGAT(pl.LightningModule):
    def __init__(self, dataset, node_features, num_classes, in_heads=4, mid_heads=4, out_heads=6, head_features=256, l2_reg=0, lr = 0.005, dropout=0):
        super(induGAT, self).__init__()
        self.dataset = dataset

        self.head_features = head_features
        self.in_heads = in_heads
        self.mid_heads = mid_heads
        self.out_heads = out_heads

        self.lr = lr
        self.l2_reg = l2_reg
        self.dropout = dropout

        self.node_features = node_features
        self.num_classes = num_classes

        self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.in_heads, add_self_loops=True, dropout=self.dropout) 
        self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.head_features, heads=self.mid_heads, dropout=self.dropout) 
        self.gat3 = GATConv(in_channels=self.head_features * self.mid_heads, out_channels=self.num_classes, heads=out_heads, concat=False, dropout=self.dropout)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.gat3(x, edge_index)
        x = x.detach()
        x = Sigmoid()

        return x#Sigmoid(x.detach())


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):
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
        # if self.dataset == 'PPI':
        dataset = PPI(root='/tmp/PPI',split='train')     
        # print(data.train_mask.sum().item())
        # print(data.val_mask.sum().item())
        # print(data.test_mask.sum().item())   
        return DataLoader(dataset)        
        

    def test_dataloader(self):
        # if self.dataset == 'PPI':
        dataset = PPI(root='/tmp/PPI',split='test')    
        return DataLoader(dataset)        