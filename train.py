import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl


# Could rather make an inductiveGAT and transductiveGAT?
# TODO add logging, e.g. tensorboard
class GATCora(pl.LightningModule):
    def __init__(self, heads=8, gat1_features=8):  # Useful so that we can play around with number of heads and features.
        super(GATCora, self).__init__()
        # From GAT paper, Section 3.3
        self.gat1_features = gat1_features
        self.heads = heads

        self.num_node_features = 1433
        self.num_classes = 7

        self.gat1 = GATConv(in_channels=self.num_node_features, out_channels=self.gat1_features, heads=self.heads)
        self.gat2 = GATConv(in_channels=gat1_features * self.heads, out_channels=self.num_classes, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        x = F.log_softmax(x, dim=1)  # TODO ELU activation in gat2 already

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        out = self(batch)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])

        return loss

    def test_step(self, batch, batch_idx):
        # Copied from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs
        out = self(batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        # TODO change to torch accuracy metric
        test_correct = pred[batch.test_mask] == batch.y[batch.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(batch.test_mask.sum())  # Derive ratio of correct predictions.

        return test_acc

    def train_dataloader(self):
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        return DataLoader(dataset)


def train_cora():
    gat_cora = GATCora()

    trainer = pl.Trainer(max_epochs=4)
    trainer.fit(gat_cora)


if __name__ == "__main__":
    # TOOD argparsing, could do one for each dataset?

    train_cora()
