from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
from torch_geometric.datasets import PPI

from models.GATModel import GATModel


class PPI_GAT(GATModel):
    def __init__(self, attention_penalty=0.0, track_grads=False, **config):
        super().__init__(**config)
        self.loss_fn = BCEWithLogitsLoss(reduction='mean')
        self.attention_penalty = attention_penalty
        self.track_grads = track_grads

    def training_step(self, batch, batch_idx):
        # Get the outputs from the forwards function, the edge index and the tensor of attention weights.
        out, edge_index, attention_list = self.forward_and_return_attention(batch, return_attention_weights=True)

        loss = self.loss_fn(out, batch.y)

        # Penalise deviation from constant attention (const-GAT), for analysing the gain of attention
        attention_norm = self.calc_attention_norm(edge_index, attention_list)
        self.log("train_attention_norm", attention_norm.detach().cpu())

        norm_loss = self.attention_penalty * attention_norm
        # print("Norm Loss: {}".format(norm_loss.detach().cpu()))
        # print("bce loss: ", loss.detach().cpu())
        # print(f"norm loss with lambda = {self.attention_penalty}", norm_loss.detach().cpu())
        self.log("train_norm_loss", norm_loss.detach().cpu())

        # Only add norm if we have a positive attention penalty - might cause weirdness (TM) otherwise
        if self.attention_penalty != 0.0:
            loss = loss + norm_loss
        # print("Total Loss: {}".format(loss.detach().cpu()))

        self.log('train_loss', loss.detach().cpu(), prog_bar=True, logger=True)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss = self.loss_fn(out, batch.y)
        self.log('val_loss', loss.detach().cpu(), prog_bar=True, logger=True)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('val_f1_score', f1, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('test_f1_score', f1, prog_bar=True, logger=True)

        return f1

    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')
