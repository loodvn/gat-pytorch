import torch
import sys
import torch.nn.functional as F
import argparse
from torch_geometric.datasets import PPI
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATConv
from torch.nn import Sigmoid, Linear, BCEWithLogitsLoss
import pytorch_lightning as pl
from gat_layer import GATLayer
from sklearn.metrics import f1_score

# TODO improve logging, e.g. tensorboard
# TODO validation
# TODO loading correctly
# TODO skip connections in middle GATConv layer

# pl.seed_everything(42)

# Taken from AI Emp.
# OW - will remove just trying to make sure it works
class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).

    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it

    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list
        print("creating dataset")

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        print(idx)
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """
    print(batch)
    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    print("graph coalate function")
    print(node_features.shape)
    print(node_labels.shape)
    print(edge_index.shape)

    return node_features, node_labels, edge_index


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
        self.skip_connection = Linear(self.head_features * self.in_heads, self.head_features * self.mid_heads, bias=False)

        # These will be initialised in prepare_data
        self.train_ds, self.val_ds, self.test_ds = None, None, None


    def forward(self, data):
        x, edge_index = data[0], data[2]
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        skip_values = self.skip_connection(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.gat3(x + skip_values, edge_index) #+ layer1_skip
        return x

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)#, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):
        print(batch)
        print(batch_idx)
        out = self(batch)
        # print("sigmoid error: ", (out[0]-batch.y[0]))
        loss_fn = BCEWithLogitsLoss(reduction='mean') #F.binary_cross_entropy(out, batch.y)
        loss = loss_fn(out, batch[1])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out > 0, y_true=batch[1], average='micro')
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        # loss = F.binary_cross_entropy(out, batch.y)
        loss_fn = BCEWithLogitsLoss(reduction='mean') #F.binary_cross_entropy(out, batch.y)
        loss = loss_fn(out, batch[1])
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out > 0, y_true=batch[1], average="micro")
        self.log('val_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def test_step(self, batch, batch_idx):
        out = self(batch)
        pred = (out > 0)

        f1 = f1_score(y_pred=pred, y_true=batch[1], average="micro")
        self.log('val_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        print(f1)
        # TODO can add accuracy/precision/recall although not sure how that aggregates in multilabel setting

        return f1

    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')

    # Only for PPI dataset at this stage - move into train.py
    def train_dataloader(self):
        # print("Calling val train loader.")
        node_features_list, node_label_list, edge_index_list = reconfig_data(self.train_ds)
        print('finished reconfiguring training data')
        # print("Edge data being passed:")        
        print("data loader train object: ")
        print(GraphDataLoader(node_features_list, node_label_list, edge_index_list, batch_size=2, shuffle=True).__dict__)
        return GraphDataLoader(node_features_list, node_label_list, edge_index_list, batch_size=2, shuffle=True)        
    
    def val_dataloader(self):
        # print("Calling val data loader.")
        node_features_list, node_label_list, edge_index_list = reconfig_data(self.val_ds)
        print('finished reconfiguring val data')

        # print("Edge data being passed:")
        # for e in edge_index_list:
        #     print(e.shape)
        return GraphDataLoader(node_features_list, node_label_list, edge_index_list, batch_size=1) 

    def test_dataloader(self):
        # print("Calling val test loader.")
        node_features_list, node_label_list, edge_index_list = reconfig_data(self.test_ds)
        print('finished reconfiguring test data')

        # print("Edge data being passed:")      
        # for e in edge_index_list:
        #     print(e.shape)
        return GraphDataLoader(node_features_list, node_label_list, edge_index_list, batch_size=1)




def reconfig_data(loaded_dataset):
    # Takes in something like self.val_ds and returns the lists we need.
    node_features_list = [] 
    for tensor_index, i in enumerate(loaded_dataset.slices['x'].numpy()):
        if i+1 < len(loaded_dataset.slices['x'].numpy()):
            node_features_list.append(loaded_dataset.data.x[tensor_index:loaded_dataset.slices['x'].numpy()[i+1]])
        else:
            node_features_list.append(loaded_dataset.data.x[tensor_index:])
    node_label_list = []
    for tensor_index, i in enumerate(loaded_dataset.slices['y'].numpy()):
        if i+1 < len(loaded_dataset.slices['y'].numpy()):
            node_label_list.append(loaded_dataset.data.y[tensor_index:loaded_dataset.slices['y'].numpy()[i+1]])
        else:
            node_label_list.append(loaded_dataset.data.y[tensor_index:])
    edge_index_list = []
    for tensor_index, i in enumerate(loaded_dataset.slices['edge_index'].numpy()):
        if i+1 < len(loaded_dataset.slices['edge_index'].numpy()):
            edge_index_list.append(loaded_dataset.data.edge_index[0:2, tensor_index:loaded_dataset.slices['edge_index'].numpy()[i+1]])
        else:
            edge_index_list.append(loaded_dataset.data.edge_index[0:2, tensor_index:])
    # print("Node label: ")
    # print(node_label_list)
    # print("Features list: ")
    # print(node_features_list)
    # print("Edge List: ")
    # print(edge_index_list)
    return (node_features_list, node_label_list, edge_index_list)


if __name__ == "__main__":
    gat = induGAT(dataset='PPI', node_features=50, num_classes=121, lr=0.005, l2_reg=0)
    trainer = pl.Trainer(max_epochs=10)#, limit_train_batches=0.1)
    trainer.fit(gat)
    trainer.test()