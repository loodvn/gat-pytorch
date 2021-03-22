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

        # Is this deffo correct should we instead be doing (x, 1024) -> (x, 121) rather than (x, 1024) -> (x, 6, 121) thenmean to -> (x, 121)
        # OW / LVN.
        
       
    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        self.layer_step = 2 if self.add_skip_connection else 1

        for i in range(0, len(self.gat_model), self.layer_step):
            if i != 0:
                x = F.elu(x)
            # If skip connection the perform the GAT layer and add this to the skip connection values.
            if self.add_skip_connection:
                x = self.perform_skip_connection(
                    skip_connection_layer=self.gat_model[i+1], 
                    input_node_features=x, 
                    gat_output_node_features=self.gat_model[i](x, edge_index), 
                    head_concat=self.gat_model[i].concat)
            else:
                x = self.gat_model[i](x, edge_index)
            
            # In either can then perform a elu activation.
        return x
    
    def perform_skip_connection(self, skip_connection_layer, input_node_features, gat_output_node_features, head_concat):
        # print("Layer: {}".format(layer))
        # print("Input shape:")
        # print(input_node_features.shape)
        # print("Output shape: ")
        # print(output_node_features.shape)

        if input_node_features.shape[-1] == gat_output_node_features.shape[-1]:
            # This is fine we can just add these and return.
            gat_output_node_features += input_node_features
        else:
            if head_concat:
                gat_output_node_features += skip_connection_layer(input_node_features)
            else:
                # Remove the hard coding.
                # OW: TODO: need to pass these in I think.
                skip_output = skip_connection_layer(input_node_features).view(-1, 6, 121)
                gat_output_node_features += skip_output.mean(dim=1)
        
        return gat_output_node_features

    def training_step(self, batch, batch_idx):
        out = self(batch)
    
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        self.log('train_loss', loss.detach().cpu().numpy(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1.detach().cpu().numpy(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        self.log('val_loss', loss.detach().cpu().numpy(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('val_f1_score', f1.detach().cpu().numpy(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('test_f1_score', f1.detach().cpu().numpy(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return f1

    # This should dynamically choose dataset class - not use PPI by default
    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')
