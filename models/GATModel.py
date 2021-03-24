from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv

from .gat_layer import GATLayer
# pl.seed_everything(42)
from run_config import LayerType


class GATModel(pl.LightningModule):
    def __init__(self, 
                 layer_type: LayerType,
                 dataset: str,
                 num_classes: int,
                 num_input_node_features: int,
                 num_layers: int,
                 num_heads_per_layer: List[int],
                 heads_concat_per_layer: List[bool],
                 head_output_features_per_layer: List[int],
                 add_skip_connection: List[bool],
                 dropout: float,
                 l2_reg: float,
                 learning_rate: float,
                 train_batch_size: int,
                 num_epochs: int,
                 const_attention: bool,
                 **kwargs):
        """[summary]
        # UPDATE THIS!!
        Args:
            config (dict): 
                - layer_type: str
                - num_input_node_features: int,
                - num_layers: int
                - num_heads_per_layer: List[int]
                - heads_concat_per_layer: List[bool]
                - head_output_features_per_layer: List[int]
                - add_skip_connection: bool 
                - dropout: float
                - l2_reg: float
                - learning_rate: float
                - train_batch_size: int
                - num_epochs: int
        """
        super(GATModel, self).__init__()

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
        self.const_attention = const_attention
        # In order to make the number of heads consistent as this is used in the in_channels for our GAT layer we have prepended the list given by the user
        # with a 1 to signal that in the first layer, the input is just 1 * num_input_node_features
        self.num_heads_per_layer = [1] + num_heads_per_layer
        self.head_output_features_per_layer = head_output_features_per_layer
        self.heads_concat_per_layer = heads_concat_per_layer

        self.train_ds, self.val_ds, self.test_ds = None, None, None
        
        # Collect the layers into a list and then place together into a ModuleList
        gat_layers = []
        skip_layers = []
        for i in range(0, self.num_layers):
            # Depending on the implimentation layer type depends what we do.
            if self.layer_type == LayerType.GATLayer:
                gat_layer = GATLayer(
                    in_features=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                    out_features=self.head_output_features_per_layer[i+1],
                    num_heads=self.num_heads_per_layer[i+1],
                    concat=self.heads_concat_per_layer[i],
                    dropout=self.dropout,
                    bias=False,
                    add_self_loops=True,
                    const_attention=self.const_attention
                )
            elif self.layer_type == LayerType.PyTorch_Geometric:
                gat_layer = GATConv(
                    in_channels=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                    out_channels=self.head_output_features_per_layer[i+1],
                    heads=self.num_heads_per_layer[i+1],
                    concat=self.heads_concat_per_layer[i],
                    dropout=self.dropout,
                    add_self_loops=True,
                    bias=False,
                )
            else:
                raise ValueError(f"Incorrect layer type passed in: {self.layer_type}. Must be one of {list(LayerType)}")

            gat_layers.append(gat_layer)

            # In either case if we need to add skip connections we can do this outside of the layer.
            # REASONING: IN ORDER TO KEEP THE SAME INTERFACE WE USE THE SKIP CONNECTIONS OUTSIDE OF THE GAT LAYER DEF.
            # These linear projections add a lot of extra capacity - basically the same size as the weight matrix W.
            if self.add_skip_connection[i]:
                skip_in = self.num_heads_per_layer[i] * self.head_output_features_per_layer[i]

                if self.heads_concat_per_layer[i]:
                    # If concatenating: add a linear projection from NH(l-1) * F_OUT(l-1) -> NH(l) * F_OUT(l)
                    skip_out = self.num_heads_per_layer[i + 1] * self.head_output_features_per_layer[i + 1]
                else:
                    # Add a linear projection from NH(l-1) * F_OUT(l-1) to NH(l) * F_OUT(l).
                    # TODO we can actually just use F_OUT, perhaps too much capacity
                    skip_out = self.num_heads_per_layer[i + 1] * self.head_output_features_per_layer[i + 1]

                if skip_in == skip_out:
                    skip_layer = nn.Identity()
                else:
                    skip_layer = nn.Linear(in_features=skip_in, out_features=skip_out, bias=False)

                skip_layers.append(skip_layer)
        
        # Once this is finished we can create out network by unpacking the layers into teh Sequential module class.
        self.gat_layer_list = nn.ModuleList(gat_layers)
        self.skip_layer_list = nn.ModuleList(skip_layers)
        print("GAT Layers", self.gat_layer_list)
        print("Skip Layers", self.skip_layer_list)

    # def reset_parameters(self):
    #     self.gat1.reset_parameters()
    #     self.gat2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        num_layers = len(self.gat_layer_list)
        skip_count = 0

        for i in range(0, num_layers):
            # Store layer input for skip connection
            layer_input = x

            x = F.dropout(x, p=self.dropout, training=self.training)
            # Get outputs from GAT layer
            x = self.gat_layer_list[i](x, edge_index)

            # Add a skip connection between the input and GAT layer output
            if self.add_skip_connection[i]:
                # Use linear projection if layer_input.dim() != x.dim(); otherwise skip_layer == nn.Identity()
                skip_output = self.skip_layer_list[skip_count](layer_input)
                skip_count += 1
                if self.heads_concat_per_layer[i]:
                    x = x + skip_output
                else:
                    # Aggregate by mean
                    skip_output = skip_output.view(-1, self.num_heads_per_layer[i + 1],
                                                   self.head_output_features_per_layer[i + 1])
                    x = x + skip_output.mean(dim=1)

            # ELU between layers
            if i != num_layers - 1:
                x = F.elu(x)

        return x

    def forward_and_return_attention(self, data, return_attention_weights=True):
        x, edge_index = data.x, data.edge_index

        attention_weights_list = []
        num_layers = len(self.gat_layer_list)
        skip_count = 0

        for i in range(0, num_layers):
            # Store layer input for skip connection
            layer_input = x

            x = F.dropout(x, p=self.dropout, training=self.training)
            # Get outputs from GAT layer
            x, (edge_index, layer_attention_weight) = self.gat_layer_list[i](x, edge_index, return_attention_weights=return_attention_weights)
            attention_weights_list.append(layer_attention_weight)

            # Add a skip connection between the input and GAT layer output
            if self.add_skip_connection[i]:
                # Use linear projection if layer_input.dim() != x.dim(); otherwise skip_layer == nn.Identity()
                print()
                skip_output = self.skip_layer_list[skip_count](layer_input)
                skip_count += 1
                if self.heads_concat_per_layer[i]:
                    x = x + skip_output
                else:
                    # Aggregate by mean
                    skip_output = skip_output.view(-1, self.num_heads_per_layer[i + 1],
                                                   self.head_output_features_per_layer[i + 1])
                    x = x + skip_output.mean(dim=1)

            # ELU between layers
            if i != num_layers - 1:
                x = F.elu(x)

        return x, edge_index, attention_weights_list

    def perform_skip_connection(self, skip_connection_layer, layer_input, layer_output, concat, heads_out, features_out):
        if layer_input.shape[-1] == layer_output.shape[-1]:
            # This is fine we can just add these and return.
            layer_output += layer_input
        else:
            if concat:
                layer_output += skip_connection_layer(layer_input)
            else:
                skip_output = skip_connection_layer(layer_input).view(-1, heads_out, features_out)
                layer_output += skip_output.mean(dim=1)
        
        return layer_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_ds)

    def test_dataloader(self):
        return DataLoader(self.test_ds)
