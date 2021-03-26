import torch
from sklearn.metrics import balanced_accuracy_score
from torch_geometric.datasets import GNNBenchmarkDataset

import models.GATModel


class PatternGAT(models.GATModel.GATModel):
    def __init__(self, **config):
        super().__init__(**config)
        # 0.1765 of the train dataset (209900. / 1189120.0 over all graphs) is from the positive class
        self.prop_pos = 0.1765
        data = [1/self.prop_pos]  # previously [4.65]
        dataset_balance = torch.tensor(data)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=dataset_balance)
        self.track_grads=False

    def training_step(self, batch, batch_idx):
        out = self(batch)
        out = torch.squeeze(out)

        target = (batch.y).float()

        loss = self.loss_fn(out, target)

        out = (out > 0)  # If logits are > 0, then sigmoid probabilities would be > 0.5

        train_acc = self.balanced_acc(target.detach().cpu().numpy(), out.detach().cpu().numpy())

        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_weighted_acc', train_acc, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        out = torch.squeeze(out)

        target = (batch.y).float()

        loss = self.loss_fn(out, target)

        out = (out > 0)

        val_acc = self.balanced_acc(target.detach().cpu().numpy(), out.detach().cpu().numpy())

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_weighted_acc', val_acc, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)
        out = torch.squeeze(out)
        out = (out > 0)
        target = (batch.y).float()

        test_acc = self.balanced_acc(target.detach().cpu().numpy(), out.detach().cpu().numpy())

        self.log('test_acc', test_acc, prog_bar=True, logger=True)
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

    def balanced_acc(self, y_true, y_pred):
        # Note: Please ensure that you detach().cpu() tensors before calling
        # https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score
        # "each sample is weighted according to the inverse prevalence of its true class"
        sample_weights = 1 / self.prop_pos * (y_pred == 1.) + (1 / (1 - self.prop_pos) * (y_pred == 0.))
        balanced_acc = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        return balanced_acc
