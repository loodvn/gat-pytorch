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

# TODO improve logging, e.g. tensorboard
# TODO validation
# TODO loading correctly
# TODO skip connections in middle GATConv layer

# pl.seed_everything(42)

# OW test
# Addition 


class induGAT(pl.LightningModule):
    # def __init__(self, dataset, node_features, num_classes, first_layer_heads=4, second_layer_heads=4, third_layer_heads=6, head_features=256, l2_reg=0, lr = 0.005, dropout=0):
    def __init__(self, config):    
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
        super(induGAT, self).__init__()


        # Decide whether we are using our layer or the default implimentation for PyTorch Geometric of GAT.
        # See: (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv)
        self.layer_type = config.get('layer_type')
        self.add_skip_connection = config.get('add_skip_connection')

         # These will be initialised in prepare_data
        self.train_ds, self.val_ds, self.test_ds = None, None, None

        # Retrieve the hyper parameters for the network.
        self.num_layers = config.get('num_layers')
        self.lr = config.get('learning_rate')
        self.l2_reg = config.get('l2_reg')
        self.dropout = config.get('dropout')
        self.input_node_features = config.get('num_input_node_features')
        self.num_classes = config.get('num_classes')
        self.train_batch_size = int(config.get('train_batch_size'))
        self.num_epochs = int(config.get('num_epochs'))
        # In order to make the number of heads consistent as this is used in the in_channels for our GAT layer we have prepended the list given by the user
        # with a 1 to signal that in the first layer, the input is just 1 * num_input_node_features
        self.num_heads_per_layer = [1] + config.get('num_heads_per_layer')
        self.head_output_features_per_layer = config.get('head_output_features_per_layer')
        self.heads_concat_per_layer = config.get('heads_concat_per_layer')

        # REASONING: IN ORDER TO KEEP THE SAME INTERFACE WE USE THESKIP CONNECTIONS OUTSIDE OF THE GAT LAYER DEF.

        # Collect the layers into a list and then place together into a Sequential model.
        layers = []
        for i in range(0, self.num_layers):
            # Depending on the implimentation layer type depends what we do.
            if self.layer_type == "Ours":
                gat_layer = GATLayer(
                    in_features=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i], 
                    out_features=self.head_output_features_per_layer[i+1], 
                    num_heads=self.num_heads_per_layer[i+1], 
                    concat=self.heads_concat_per_layer[i], 
                    dropout=self.dropout, 
                    bias=False, 
                    add_self_loops=True, 
                    const_attention=False
                )
            else:
                gat_layer = GATConv(
                    in_channels=self.num_heads_per_layer[i] * self.head_output_features_per_layer[i],
                    out_channels=self.head_output_features_per_layer[i+1], 
                    heads=self.num_heads_per_layer[i+1], 
                    add_self_loops=True,
                    concat=self.heads_concat_per_layer[i]
                ) 
            layers.append(gat_layer)

            # In either case if we need to add skip connections we can do this outside of the layer.
            if self.add_skip_connection:
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
        self.gat_model = ModuleList(layers)
        print(self.gat_model)

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
                # OW: TODO - need to pass these in I think.
                skip_output = skip_connection_layer(input_node_features).view(-1, 6, 121)
                gat_output_node_features += skip_output.mean(dim=1)
        
        return gat_output_node_features

    def forward_and_return_attention(self, data, return_attention_coeffs=True):
        x, edge_index = data.x, data.edge_index
        self.layer_step = 2 if self.add_skip_connection else 1
        attention_weights_list = []

        for i in range(0, len(self.gat_model), self.layer_step):
            if i != 0:
                x = F.elu(x)
            # If skip connection the perform the GAT layer and add this to the skip connection values.
            if self.add_skip_connection:
                gat_layer_output, edge_index, layer_attention_weight = self.gat_model[i](x, edge_index, return_attention_coeffs)
                attention_weights_list.append(layer_attention_weight)
                x = self.perform_skip_connection(
                    skip_connection_layer=self.gat_model[i+1], 
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
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=True)        
    
    def val_dataloader(self):
        return DataLoader(self.val_ds)

    def test_dataloader(self):
        return DataLoader(self.test_ds)

if __name__ == "__main__":
    import time
    start = time.time()
    gat = induGAT(dataset='PPI', node_features=50, num_classes=121, lr=0.005, l2_reg=0)
    trainer = pl.Trainer(max_epochs=100)#, limit_train_batches=0.1)
    trainer.fit(gat)
    trainer.test()
    end = time.time()