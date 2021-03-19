import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from torch.nn import Sigmoid, Linear, BCEWithLogitsLoss
import pytorch_lightning as pl
from models.gat_layer import GATLayer
from sklearn.metrics import f1_score

# TODO improve logging, e.g. tensorboard
# TODO validation
# TODO loading correctly
# TODO skip connections in middle GATConv layer

# pl.seed_everything(42)

# OW test
# Addition 


class induGAT(pl.LightningModule):
    def __init__(self, dataset, node_features, num_classes, first_layer_heads=4, second_layer_heads=4, third_layer_heads=6, head_features=256, l2_reg=0, lr = 0.005, dropout=0):
        super(induGAT, self).__init__()
        self.dataset = dataset

        self.head_features = head_features
        self.first_layer_heads = first_layer_heads
        self.second_layer_heads = second_layer_heads
        self.third_layer_heads = third_layer_heads

        self.lr = lr
        self.l2_reg = l2_reg
        self.dropout = dropout

        # These now dont need to be passed in and can be set in dataloader
        self.node_features = node_features
        self.num_classes = num_classes

        self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.first_layer_heads, add_self_loops=True) 
        self.skip_conn_1 = Linear(in_features=self.node_features, out_features=self.head_features * self.first_layer_heads, bias=False)

        self.gat2 = GATConv(in_channels=self.head_features * self.first_layer_heads, out_channels=self.head_features, heads=self.second_layer_heads, add_self_loops = True)
        self.skip_conn_2 = Linear(in_features=self.head_features * self.first_layer_heads, out_features=self.head_features * self.second_layer_heads, bias=False)

        self.gat3 = GATConv(in_channels=self.head_features * self.second_layer_heads, out_channels=self.num_classes, add_self_loops=True, heads=third_layer_heads, concat=False)
        self.skip_conn_3 = Linear(in_features=self.head_features * self.second_layer_heads, out_features=self.num_classes * self.third_layer_heads, bias=False)
        # Is this deffo correct should we instead be doing (x, 1024) -> (x, 121) rather than (x, 1024) -> (x, 6, 121) thenmean to -> (x, 121)
        # OW / LVN.

        self.skip_connection = Linear(self.head_features * self.in_heads, self.head_features * self.mid_heads, bias=False)
        
        # These will be initialised in prepare_data
        self.train_ds, self.val_ds, self.test_ds = None, None, None


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.skip_connection(input_node_features=x, output_node_features=self.gat1(x, edge_index), layer=1)
        x = F.elu(x)
        x = self.skip_connection(input_node_features=x, output_node_features=self.gat2(x, edge_index), layer=2)
        x = F.elu(x)
        x = self.skip_connection(input_node_features=x, output_node_features=self.gat3(x, edge_index), layer=3)
        return x
    
    def skip_connection(self, input_node_features, output_node_features, layer):
        # print("Layer: {}".format(layer))
        # print("Input shape:")
        # print(input_node_features.shape)
        # print("Output shape: ")
        # print(output_node_features.shape)

        if input_node_features.shape[-1] == output_node_features.shape[-1]:
            # This is fine we can just add these and return.
            output_node_features += input_node_features
        else:
            # Need to project as FIN != FOUT.
            if layer == 1:
                # print("Post skip conn for layer 1 shape: ")
                # print(self.skip_conn_1(input_node_features).shape)
                # print("After reshaping view")
                # print(self.skip_conn_1(input_node_features).view(-1, self.first_layer_heads, self.head_features).shape)
                output_node_features += self.skip_conn_1(input_node_features)
            elif layer == 2:
                # print("Post skip conn for layer 2 shape: ")
                # print(self.skip_conn_2(input_node_features).shape)
                # print("After reshaping view")
                # print(self.skip_conn_2(input_node_features).view(-1, self.second_layer_heads, self.head_features).shape)
                output_node_features += self.skip_conn_2(input_node_features)
            else:
                # print("Post skip conn for layer 3 shape: ")
                # print(self.skip_conn_3(input_node_features).shape)
                # print("After reshaping view")
                # Again - check.
                # MEAN BEFORE SUMMING - OW / LVD
                skip_output = self.skip_conn_3(input_node_features).view(-1, self.third_layer_heads, self.num_classes)
                # print(skip_output.shape)
                # print("Skip output after mean")
                # print(skip_output.mean(dim=1).shape)
                output_node_features += skip_output.mean(dim=1)
        
        # if layer != 3:
        #     output_node_features = output_node_features.view(-1, self.second_layer_heads * self.head_features)
        # else:
        #     output_node_features = output_node_features

        return output_node_features


    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        out = self(batch)
        # print("sigmoid error: ", (out[0]-batch.y[0]))
        loss_fn = BCEWithLogitsLoss(reduction='mean') #F.binary_cross_entropy(out, batch.y)
        loss = loss_fn(out, batch.y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # loss = F.binary_cross_entropy(out, batch.y)
        loss_fn = BCEWithLogitsLoss(reduction='mean') #F.binary_cross_entropy(out, batch.y)
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
        # TODO can add accuracy/precision/recall although not sure how that aggregates in multilabel setting

        return f1

    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        # N = len(train_ds.dataset)
        # train_size = 0.75*N
        # # Split into train and val
        # self.train_ds, self.val_ds = torch.utils.data.random_split(train_ds, [train_size, N-train_size])

        self.test_ds = PPI(root='/tmp/PPI', split='test')

    # Only for PPI dataset at this stage - move into train.py
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, shuffle=True)        
    
    def val_dataloader(self):
        return DataLoader(self.val_ds) 

    def test_dataloader(self):
        return DataLoader(self.test_ds)

if __name__ == "__main__":
    gat = induGAT(dataset='PPI', node_features=50, num_classes=121, lr=0.005, l2_reg=0)
    trainer = pl.Trainer(max_epochs=10)#, limit_train_batches=0.1)
    trainer.fit(gat)
    trainer.test()