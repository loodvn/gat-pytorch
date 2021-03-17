import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
from torch.nn import Sigmoid
from torch.nn import BCEWithLogitsLoss
import pytorch_lightning as pl
from gat_layer import GATLayer
from sklearn.metrics import f1_score

# TODO improve logging, e.g. tensorboard
# TODO validation
# TODO loading correctly
# TODO skip connections in middle GATConv layer

# pl.seed_everything(42)

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

        self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, add_self_loops=True, heads=self.in_heads)#, add_self_loops=True, dropout=self.dropout) 
        self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.head_features, heads=self.mid_heads) #dropout=self.dropout) 
        self.gat3 = GATConv(in_channels=self.head_features * self.mid_heads, out_channels=self.num_classes, heads=out_heads, concat=False)#, dropout=self.dropout)

        # These will be initialised in prepare_data
        self.train_ds, self.val_ds, self.test_ds = None, None, None


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        # layer1_skip = x  # add skip connection between layer 1 output and layer 3 input
        # print(layer1_skip)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        # print(layer1_skip)
        x = self.gat3(x , edge_index) #+ layer1_skip
        # print(x)
        # s = Sigmoid()
        # x = s(x)
        # print(x)

        return x

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)#, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):
        out = self(batch)
        # print("sigmoid error: ", (out[0]-batch.y[0]))
        loss_fn = BCEWithLogitsLoss(reduction='mean') #F.binary_cross_entropy(out, batch.y)
        loss = loss_fn(out, batch.y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out > 0, y_true=batch.y, average='micro')
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # loss = F.binary_cross_entropy(out, batch.y)
        loss_fn = BCEWithLogitsLoss(reduction='mean') #F.binary_cross_entropy(out, batch.y)
        loss = loss_fn(out, batch.y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out > 0, y_true=batch.y, average="micro")
        self.log('val_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def test_step(self, batch, batch_idx):
        out = self(batch)
        pred = (out > 0)

        f1 = f1_score(y_pred=pred, y_true=batch.y, average="micro")
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
        return DataLoader(self.train_ds, batch_size=2, shuffle=True)        
    
    def val_dataloader(self):
        return DataLoader(self.val_ds) 

    def test_dataloader(self):
        return DataLoader(self.test_ds)

if __name__ == "__main__":
    gat = induGAT(dataset='PPI', node_features=50, num_classes=121, lr=0.005, l2_reg=0)
    trainer = pl.Trainer(max_epochs=10)#, limit_train_batches=0.1)
    trainer.fit(gat)
    trainer.test()