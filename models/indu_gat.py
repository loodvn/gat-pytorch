from torch_geometric.datasets import PPI
from torch.nn import BCEWithLogitsLoss
import pytorch_lightning as pl

from sklearn.metrics import f1_score
from models.GATModel import GATModel

# pl.seed_everything(42)

# OW test
# Addition 


class induGAT(GATModel):
    def __init__(self, **config):    
        super().__init__(**config)
        self.criterion = BCEWithLogitsLoss(reduction='mean')

    def training_step(self, batch, batch_idx):
        out = self(batch)
    
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        self.log('train_loss', loss.detach().cpu(), prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        self.log('val_loss', loss.detach().cpu(), prog_bar=True, logger=True)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('val_f1_score', f1, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('test_f1_score', f1, prog_bar=True, logger=True)

        return f1

    # This should dynamically choose dataset class - not use PPI by default
    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')
