import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import torch.nn as nn
import pytorch_lightning as pl
from models.gat_layer import GATLayer

# TODO improve logging, e.g. tensorboard
# TODO validation
# TODO loading correctly
# pl.seed_everything(42)

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

        # self.patience = 100
        # self.lossMin = 1e10

        self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.in_heads, add_self_loops=True, dropout=self.dropout)#add_self_loops=True, # add self loops? GATLayer(in_channels=self.node_features, out_channels=self.head_features, number_of_heads=self.in_heads, dropout=self.dropout, alpha = 0.2)#
        self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, heads=out_heads, add_self_loops=True, concat=False, dropout=self.dropout) #concat=False, GATLayer(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, number_of_heads=self.out_heads, concat=False, dropout=self.dropout, alpha = 0.2)#


    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # dropout to avoid overfitting as dataset is small
        # print(self.training)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1) 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        print(loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = float (pred[batch.val_mask].eq(batch.y[batch.val_mask]).sum().item())
        val_acc = (correct / batch.val_mask.sum().item())
        # loss_fn = nn.CrossEntropyLoss()
        val_loss = F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask])
        # val_loss = loss_fn(out[batch.val_mask], batch.y[batch.val_mask])
        # print(loss)
        #early stopping
    #     if (val_loss<=lossMin):
    #         lossMin = val_loss
    #         count = 0
    #         model_best.load_state_dict(model.state_dict())
    #         val_acc_best = (correct / data.val_mask.sum().item())
    #     else:
    #         count+=1
    #         if(count>patience):
    #             print("patience lost", end=' ')
    #             break
                
    # val_acc += val_acc_best
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Copied from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs
        out = self(batch)
        # out1 = F.log_softmax(out, dim=1) 
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        
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
        return DataLoader(dataset) #, shuffle=True)
        
    def val_dataloader(self):
        dataset = Planetoid(root='/tmp/' + self.dataset, name=self.dataset)
        # print(data.train_mask.sum().item())
        # print(data.val_mask.sum().item())
        # print(data.test_mask.sum().item())
        return DataLoader(dataset)

    def test_dataloader(self):
        dataset = Planetoid(root='/tmp/' + self.dataset, name=self.dataset)        
        return DataLoader(dataset)