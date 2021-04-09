import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from models.GATModel import GATModel


class PlanetoidGAT(GATModel):
    def __init__(self, attention_reward=0.0, track_grads=False, **config):
        super().__init__(**config)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.attention_reward = attention_reward
        self.track_grads = track_grads

    def training_step(self, batch, batch_idx):
        out, edge_index, attention_list = self.forward_and_return_attention(batch, True)

        # We can then add a norm over the attention weights.
        attention_norm = self.calc_attention_norm(edge_index, attention_list)

        self.log("train_attention_norm", attention_norm.detach().cpu())

        norm_loss = self.attention_reward * attention_norm
        self.log("train_norm_loss", norm_loss.detach().cpu())

        # Try negative penalty (/reward) on the attention norm to encourage using attention mechanism
        loss = self.loss_fn(out[batch.train_mask], batch.y[batch.train_mask]) + norm_loss

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)  # There's only one step in epoch so we log on epoch
        return loss

    def validation_step(self, batch, batch_idx):  
        out = self(batch)
        val_loss = self.loss_fn(out[batch.val_mask], batch.y[batch.val_mask])

        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = float(pred[batch.val_mask].eq(batch.y[batch.val_mask]).sum().item())
        val_acc = (correct / batch.val_mask.sum().item())
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        # out, edge_index, attention_list = self.forward_and_return_attention(batch, return_attention_coeffs=True)
        out = self(batch)
        # self.attention_weights_list = attention_list
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        test_correct = pred[batch.test_mask] == batch.y[batch.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(batch.test_mask.sum())  # Derive ratio of correct predictions.

        self.log('test_acc', test_acc, on_epoch=True, prog_bar=True, logger=True)
        return test_acc

    # Transductive: Load whole graph, mask out when calculating loss
    def prepare_data(self):
        self.train_ds = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)
        self.val_ds = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)
        self.test_ds = Planetoid(root='/tmp/' + self.dataset_name, name=self.dataset_name)



