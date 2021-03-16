import pickle
import numpy as np
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import add_self_loops
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl
import torch_geometric
from scipy.stats import entropy
import typing
from typing import List


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

        self.attention_weights_layer1 = None
        self.attention_weights_layer2 = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x, attention_weights_layer1 = self.gat1(x, edge_index, return_attention_weights=True)
        self.attention_weights_layer1 = attention_weights_layer1
        x, attention_weights_layer2 = self.gat2(x, edge_index, return_attention_weights=True)
        self.attention_weights_layer2 = attention_weights_layer2
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


def load_cora_data():
    cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
    cora_graph = cora_dataset[0]
    cora_graph.edge_index, _ = add_self_loops(cora_graph.edge_index, num_nodes=cora_graph.x.shape[0])
    return cora_graph


def train_cora_ow():
    trainer = pl.Trainer(max_epochs=10)

    gat_cora = GATCora()
    acc = trainer.fit(gat_cora)

    layer1_attention_weights = gat_cora.attention_weights_layer1[1]
    layer2_attention_weights = gat_cora.attention_weights_layer2[1]

    print('layer 1 attention weights shape: \n{}'.format(layer1_attention_weights))
    print('layer 2 attention weights shape: \n{}'.format(layer2_attention_weights))

    return [layer1_attention_weights.detach(), layer2_attention_weights.detach()]

def prep_model_and_data_for_analysis(pretrained_model_location=None, dataset_name=None):

    if dataset_name.lower() == 'cora':
        # Graph the Cora Dataset and then add the self loops.
        cora_graph = load_cora_data()
        cora_graph.edge_index, _ = add_self_loops(cora_graph.edge_index, num_nodes=cora_graph.x.shape[0])
        print(type(cora_graph))

    return cora_graph
