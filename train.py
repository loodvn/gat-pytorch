import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv
import pytorch_lightning as pl

# class induGAT(pl.LightningModule)
#     def __init__(self, dataset, node_features, num_classes, in_heads=4, out_heads=6, head_features=8, l2_reg=0.0005, dropout=0.6):
#         super(transGAT, self).__init__()
#         self.dataset = dataset

#         # From GAT paper, Section 3.3
#         self.head_features = head_features
#         self.in_heads = in_heads
#         self.out_heads = out_heads

#         self.l2_reg = l2_reg
#         self.dropout = dropout

#         # These are cora specific and shouldnt be explicitly declared
#         self.node_features = node_features
#         self.num_classes = num_classes

#         # Is out for layer 1 correct?
#         self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.in_heads, add_self_loops=True, dropout=self.dropout) # add self loops?
#         self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, heads=out_heads, concat=False, dropout=self.dropout)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         # dropout to avoid overfitting as dataset is small
#         x = F.dropout(x, p=self.dropout)
#         x = self.gat1(x, edge_index)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout)
#         x = self.gat2(x, edge_index)

#         return F.log_softmax(x, dim=1) # think paper only mentions normal softmax? - this is fine, ollie says for speed increase


# inductive and transductive in one?
# TODO add logging, e.g. tensorboard
class transGAT(pl.LightningModule):
    def __init__(self, dataset, node_features, num_classes, in_heads=8, out_heads=1, head_features=8, l2_reg=0.0005, dropout=0.6):
        super(transGAT, self).__init__()
        self.dataset = dataset

        # From GAT paper, Section 3.3
        self.head_features = head_features
        self.in_heads = in_heads
        self.out_heads = out_heads

        self.l2_reg = l2_reg
        self.dropout = dropout

        # These are cora specific and shouldnt be explicitly declared
        self.node_features = node_features
        self.num_classes = num_classes

        # Is out for layer 1 correct?
        self.gat1 = GATConv(in_channels=self.node_features, out_channels=self.head_features, heads=self.in_heads, add_self_loops=True, dropout=self.dropout) # add self loops?
        self.gat2 = GATConv(in_channels=self.head_features * self.in_heads, out_channels=self.num_classes, heads=out_heads, concat=False, dropout=self.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # dropout to avoid overfitting as dataset is small
        x = F.dropout(x, p=self.dropout)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1) # think paper only mentions normal softmax? - this is fine, oli says for speed increase


    def configure_optimizers(self):
        # Need to use AdamW instead? - see paper: https://arxiv.org/abs/1711.05101
        # swap in with self.lr....
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.0005)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=0.005, weight_decay=0.0005)
        return optimizer

    def training_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
        # # l2 regularisation
        out = self(batch)
        # This is minimising cross entropy right?
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        # loss = torch.nn.CrossEntropyLoss(out[batch.train_mask], batch.y[batch.train_mask])
        print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        # Copied from https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#learning-methods-on-graphs
        out = self(batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        # TODO change to torch accuracy metric
        # test_correct = pred[batch.test_mask] == batch.y[batch.test_mask]  # Check against ground-truth labels.
        # test_acc = int(test_correct.sum()) / int(batch.test_mask.sum())  # Derive ratio of correct predictions.

        correct = float (pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item())
        test_acc = (correct / batch.test_mask.sum().item())
        print("correct ", correct)
        print("This is the test accuracy")
        print(test_acc)
        return test_acc

    

    # How to apply mask at this stage so we arent loading entire dataset twice? - or do we accc want that
    def train_dataloader(self):
        dataset = Planetoid(root='/tmp/' + self.dataset, name=self.dataset)     
        # print(data.train_mask.sum().item())
        # print(data.val_mask.sum().item())
        # print(data.test_mask.sum().item())   
        return DataLoader(dataset)        
        

    def test_dataloader(self):
        dataset = Planetoid(root='/tmp/' + self.dataset, name=self.dataset)        
        return DataLoader(dataset)        


def train(dataset, node_features, num_classes, max_epochs):
    if dataset == 'PPI':
        gat = induGAT(dataset, node_features, num_classes)
    else:
        gat = transGAT(dataset, node_features, num_classes)

    trainer = pl.Trainer(max_epochs=max_epochs)
    
    trainer.fit(gat)

    trainer.test()
    

if __name__ == "__main__":
    # TOOD argparsing, could do one for each dataset?
    dataset = 'Cora'
    max_epochs = 100
    task_type = 'transductive'

    if dataset == 'Cora':
        node_features = 1433
        num_classes = 7
    elif dataset == 'Pubmed':
        node_features = 500
        num_classes = 3
    elif dataset == 'Citeseer':
        node_features = 3703
        num_classes = 6  
    elif dataset == 'PPI':
        node_features = 50
        num_classes = 121
        task_type = 'inductive'
    
    train(dataset, node_features, num_classes, max_epochs)
    
