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
            if self.add_skip_connection[i]:
                # If we concat then the output shape will be the number of heads. Otherwise we take a mean over each head and therefore can omit this.
                if self.heads_concat_per_layer[i]:
                    skip_layer = Linear(
                        in_features=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                        out_features=self.head_output_features_per_layer[i+1] * self.num_heads_per_layer[i+1],
                        bias=False
                    )
                else:
                    skip_layer = Linear(
                        in_features=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                        out_features=self.num_heads_per_layer[i+1] * self.head_output_features_per_layer[i+1],
                        bias=False
                    )
                    
                layers.append(skip_layer)
        
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
                x = self.perform_skip_connection(
                    skip_connection_layer=self.skip_layer_list[skip_count],
                    input_node_features=layer_input,
                    gat_output_node_features=x,
                    head_concat=self.gat_layer_list[i].concat,
                    number_of_heads=self.gat_layer_list[i].num_heads,
                    output_node_features=self.gat_layer_list[i].out_features)
                skip_count += 1

            # ELU between layers
            if i != num_layers - 1:
                x = F.elu(x)

        return x

    def forward_and_return_attention(self, data, return_attention_coeffs=True):
        x, edge_index = data.x, data.edge_index
        attention_weights_list = []

        num_layers = len(self.gat_layer_list)
        skip_count = 0

        for i in range(0, num_layers):
            # Store layer input for skip connection
            layer_input = x

            x = F.dropout(x, p=self.dropout, training=self.training)
            # Get outputs from GAT layer
            x, (edge_index, layer_attention_weight) = self.gat_layer_list[i](x, edge_index, return_attention_coeffs)
            attention_weights_list.append(layer_attention_weight)

            # Add a skip connection between the input and GAT layer output
            if self.add_skip_connection[i]:
                x = self.perform_skip_connection(
                    skip_connection_layer=self.skip_layer_list[skip_count],
                    input_node_features=layer_input,
                    gat_output_node_features=x,
                    head_concat=self.gat_layer_list[i].concat,
                    number_of_heads=self.gat_layer_list[i].num_heads,
                    output_node_features=self.gat_layer_list[i].out_features)
                skip_count += 1

            # ELU between layers
            if i != num_layers - 1:
                x = F.elu(x)

        return x, edge_index, attention_weights_list

    def perform_skip_connection(self, skip_connection_layer, input_node_features, gat_output_node_features, head_concat, number_of_heads, output_node_features):
        if input_node_features.shape[-1] == gat_output_node_features.shape[-1]:
            # This is fine we can just add these and return.
            gat_output_node_features += input_node_features
        else:
            if head_concat:
                gat_output_node_features += skip_connection_layer(input_node_features)
            else:
                # Remove the hard coding.
                # OW: TODO - need to pass these in I think.
                skip_output = skip_connection_layer(input_node_features).view(-1, number_of_heads, output_node_features)
                gat_output_node_features += skip_output.mean(dim=1)
        
        return gat_output_node_features

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_ds)

    def test_dataloader(self):
        return DataLoader(self.test_ds)
