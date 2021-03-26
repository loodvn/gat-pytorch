import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.data import DataLoader
import models.GATModel

class PatternGAT(models.GATModel.GATModel):
    def __init__(self, **config):    
        super().__init__(**config)
        data = [4.65]
        dataset_balance = torch.tensor(data)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=dataset_balance)
        self.track_grads=False

    def training_step(self, batch, batch_idx):
        out = self(batch)
        out = torch.squeeze(out)

        target = (batch.y).float()

        loss = self.loss_fn(out, target)

        out = (out > 0)

        train_correct = out == target  # Check against ground-truth labels.
        train_acc = int(train_correct.sum()) / int(len(target))  # Derive ratio of correct predictions.


        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', train_acc, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        out = torch.squeeze(out)

        target = (batch.y).float()

        loss = self.loss_fn(out, target)
        
        out = (out > 0)

        val_correct = out == target  # Check against ground-truth labels.
        val_acc = int(val_correct.sum()) / int(len(target))  # Derive ratio of correct predictions.

        self.log('val_loss', loss, on_step=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        out = torch.squeeze(out)
        out = (out > 0)
        target = (batch.y).float()

        test_correct = out == target  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(len(target))  # Derive ratio of correct predictions.

        self.log('test_acc', test_acc, on_epoch=True, prog_bar=True, logger=True)
        return test_acc
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=0.000001)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def prepare_data(self):
        self.train_ds = GNNBenchmarkDataset(root='/tmp/Pattern', name="PATTERN", split='train')
        self.val_ds = GNNBenchmarkDataset(root='/tmp/Pattern', name="PATTERN", split='val')
        self.test_ds = GNNBenchmarkDataset(root='/tmp/Pattern', name="PATTERN", split='test')