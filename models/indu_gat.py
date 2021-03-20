import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from torch.nn import Sigmoid, Linear, BCEWithLogitsLoss, ModuleList
import pytorch_lightning as pl
from .gat_layer import GATLayer
from sklearn.metrics import f1_score
from models.GATModel import GATModel

# TODO improve logging, e.g. tensorboard
# TODO validation
# TODO loading correctly
# TODO skip connections in middle GATConv layer

# pl.seed_everything(42)

# OW test
# Addition 


class induGAT(GATModel):
    # def __init__(self, dataset, node_features, num_classes, first_layer_heads=4, second_layer_heads=4, third_layer_heads=6, head_features=256, l2_reg=0, lr = 0.005, dropout=0):
    def __init__(self, **config):    
        super().__init__(**config)
        self.criterion = BCEWithLogitsLoss(reduction='mean')

        # Is this deffo correct should we instead be doing (x, 1024) -> (x, 121) rather than (x, 1024) -> (x, 6, 121) thenmean to -> (x, 121)
        # OW / LVN.
        
    def training_step(self, batch, batch_idx):
        # print("batch dims: ", batch.x.size())
        out = self(batch)
    
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('val_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        pred = (out > 0)

        f1 = f1_score(y_pred=pred.detach().cpu().numpy(), y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('val_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(f1)

        return f1

    # This should dynamically choose dataset class - not use PPI by default
    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')