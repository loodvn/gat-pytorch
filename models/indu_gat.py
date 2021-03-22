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
    def __init__(self, **config):    
        super().__init__(**config)
        self.criterion = BCEWithLogitsLoss(reduction='mean')

    def training_step(self, batch, batch_idx):
        
        l1_lambda = 0.001
        # Get the outputs from the forwards function, the edge index and the tensor of attention weights.
        out, edge_index, attention_weights_list = self.forward_and_return_attention(batch.x, batch.edge_index, True)

        # 
    
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)

        # Idea is here to add to the loss function.
        # loss += 

        self.log('train_loss', loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        self.log('val_loss', loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('val_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('test_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return f1

    # This should dynamically choose dataset class - not use PPI by default
    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')
