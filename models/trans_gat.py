from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

from .gat_layer import GATLayer


# pl.seed_everything(42)


class transGAT(pl.LightningModule):
    def __init__(self,
                 layer_type: str,
                 dataset: str,
                 num_classes: int,
                 num_input_node_features: int,
                 num_layers: int,
                 num_heads_per_layer: List[int],
                 heads_concat_per_layer: List[bool],
                 head_output_features_per_layer: List[int],
                 add_skip_connection: bool,
                 dropout: float,
                 l2_reg: float,
                 learning_rate: float,
                 train_batch_size: int,
                 num_epochs: int,
                 **kwargs
                 ):
        """TODO docstring"""
        super(transGAT, self).__init__()

        # Decide whether we are using our layer or the default implementation for PyTorch Geometric of GAT.
        # See: (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)
        self.layer_type = layer_type
        self.add_skip_connection = add_skip_connection

        self.dataset_name = dataset
        self.num_layers = num_layers
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.input_node_features = num_input_node_features
        self.num_classes = num_classes
        self.train_batch_size = int(train_batch_size)
        self.num_epochs = int(num_epochs)
        # In order to make the number of heads consistent as this is used in the in_channels for our GAT layer we have prepended the list given by the user
        # with a 1 to signal that in the first layer, the input is just 1 * num_input_node_features
        self.num_heads_per_layer = [1] + num_heads_per_layer
        self.head_output_features_per_layer = head_output_features_per_layer
        self.heads_concat_per_layer = heads_concat_per_layer

        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        # Collect the layers into a list and then place together into a Sequential model.
        layers = []
        for i in range(0, self.num_layers):
            # Depending on the implimentation layer type depends what we do.
            if self.layer_type == "Ours":
                gat_layer = GATLayer(
                    in_features=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                    out_features=self.head_output_features_per_layer[i + 1],
                    num_heads=self.num_heads_per_layer[i + 1],
                    concat=self.heads_concat_per_layer[i],
                    dropout=self.dropout,
                    bias=False,
                    add_self_loops=True,
                    const_attention=False
                )
            else:
                gat_layer = GATConv(
                    in_channels=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                    out_channels=self.head_output_features_per_layer[i + 1],
                    heads=self.num_heads_per_layer[i + 1],
                    add_self_loops=True,
                    dropout=self.dropout,
                    concat=self.heads_concat_per_layer[i]
                )
            layers.append(gat_layer)

            # In either case if we need to add skip connections we can do this outside of the layer.
            if self.add_skip_connection:
                # If we concat then the output shape will be the number of heads. Otherwise we take a mean over each head and therefore can omit this.
                if self.heads_concat_per_layer[i]:
                    skip_layer = nn.Linear(
                        in_features=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                        out_features=self.head_output_features_per_layer[i + 1] * self.num_heads_per_layer[i + 1],
                        bias=False
                    )
                else:
                    skip_layer = nn.Linear(
                        in_features=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                        out_features=self.num_heads_per_layer[i + 1] * self.head_output_features_per_layer[i + 1],
                        bias=False
                    )
                layers.append(skip_layer)

        # Once this is finished we can create out network by unpacking the layers into teh Sequential module class.
        self.gat_model = nn.ModuleList(layers)
        print(self.gat_model)

    # def reset_parameters(self):
    #     self.gat1.reset_parameters()
    #     self.gat2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        self.layer_step = 2 if self.add_skip_connection else 1

        for i in range(0, len(self.gat_model), self.layer_step):
            if i != 0:
                x = F.elu(x)
            # If skip connection the perform the GAT layer and add this to the skip connection values.
            if self.add_skip_connection:
                x = self.perform_skip_connection(
                    skip_connection_layer=self.gat_model[i + 1],
                    input_node_features=x,
                    gat_output_node_features=self.gat_model[i](x, edge_index),
                    head_concat=self.gat_model[i].concat)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.gat_model[i](x, edge_index)
        return x

    def forward_and_return_attention(self, data, return_attention_coeffs=True):
        x, edge_index = data.x, data.edge_index
        self.layer_step = 2 if self.add_skip_connection else 1
        attention_weights_list = []

        for i in range(0, len(self.gat_model), self.layer_step):
            if i != 0:
                x = F.elu(x)
            # If skip connection the perform the GAT layer and add this to the skip connection values.
            if self.add_skip_connection:
                gat_layer_output, (edge_index, layer_attention_weight) = self.gat_model[i](x, edge_index,
                                                                                         return_attention_coeffs)
                attention_weights_list.append(layer_attention_weight)
                x = self.perform_skip_connection(
                    skip_connection_layer=self.gat_model[i + 1],
                    input_node_features=x,
                    gat_output_node_features=gat_layer_output,
                    head_concat=self.gat_model[i].concat)
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x, edge_index, layer_attention_weight = self.gat_model[i](x, edge_index, return_attention_coeffs)
                attention_weights_list.append(layer_attention_weight)
        return x, edge_index, attention_weights_list

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
        self.log('train_loss', loss, on_epoch=True, prog_bar=True,
                 logger=True)  # There's only one step in epoch so we log on epoch
        # TODO log histogram of attention weights?
        return loss

    def validation_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        val_loss = self.criterion(out[batch.val_mask], batch.y[batch.val_mask])

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = float(pred[batch.val_mask].eq(batch.y[batch.val_mask]).sum().item())
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
