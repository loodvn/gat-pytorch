import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import f1_score

from models.GATModel import GATModel


# pl.seed_everything(42)

# OW test
# Addition
from models.utils import sum_over_neighbourhood, explicit_broadcast


class induGAT(GATModel):
    def __init__(self, attention_penalty=0.0, **config):
        super().__init__(**config)
        self.loss_fn = BCEWithLogitsLoss(reduction='mean')
        self.attention_penalty = attention_penalty

    def training_step(self, batch, batch_idx):

        # Get the outputs from the forwards function, the edge index and the tensor of attention weights.
        out, edge_index, attention_list = self.forward_and_return_attention(batch, return_attention_coeffs=True)  # attention_weights_list

        loss = self.loss_fn(out, batch.y)

        # Penalise deviation from constant attention (const-GAT), for analysing the gain of attention
        attention_norm = self.calc_attention_norm(edge_index, attention_list)
        self.log("train_attention_norm", attention_norm.detach().cpu())

        norm_loss = self.attention_penalty * attention_norm
        print("Norm Loss: {}".format(loss.detach().cpu()))
        # print("bce loss: ", loss.detach().cpu())
        # print(f"norm loss with lambda = {self.attention_penalty}", norm_loss.detach().cpu())
        self.log("train_norm_loss", norm_loss.detach().cpu())

        loss = loss + norm_loss
        print("Total Loss: {}".format(loss.detach().cpu()))

        self.log('train_loss', loss.detach().cpu(), prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1, prog_bar=True, logger=True)

        return loss

    # Useful for checking if gradients are flowing
    # def on_after_backward(self):
    #     print("On backwards")
    #     # print(self.attention_reg_sum.grad)
    #     # print(self.gat_model[0].W.weight.grad)
    #     # print(self.gat_model[0].a.weight.grad)
    #     # print(self.gat_model[0].normalised_attention_coeffs.grad)

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

    # This should dynamically choose dataset class - not use PPI by default
    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')

    def calc_attention_norm(self, edge_index, attention_list):
        # (incoming) Neighbourhood: Edges that share a target node
        neighbourhood_indices = edge_index[1]

        first_attention = attention_list[0]

        # Get degrees, shaped as (E,), so that we can reshape for every layer
        degrees = sum_over_neighbourhood(
            torch.ones_like(first_attention[:, 0]),
            neighbourhood_indices=neighbourhood_indices,
            aggregated_shape=first_attention[:, 0].size(),
            broadcast_back=True,
        )
        # print("new degrees shape", degrees.size())

        num_layers = len(attention_list)
        attention_norm = torch.tensor(0.0, device=self.device)
        # Calculate attention penalty for each layer (can parallelise?)
        for i in range(num_layers):
            tmp_degrees = explicit_broadcast(degrees, attention_list[i])
            unnormalised_attention = attention_list[i] * tmp_degrees

            attention_minus_const = unnormalised_attention - 1.0

            # print(f"unnormalised_attention {i}", unnormalised_attention.detach().cpu(), unnormalised_attention.size())
            # print(f"attention_minus const {i}", attention_minus_const.detach().cpu())
            # Tensorboard must be passed in as a logger (can't use default logging for this)
            if self.logger is not None:
                tensorboard: TensorBoardLogger = self.logger
                tensorboard.experiment.add_histogram(f"unnormalised_attention_layer_{i}",
                                                     unnormalised_attention.detach().cpu())
                tensorboard.experiment.add_histogram(f"attention_minus_const_layer_{i}",
                                                     attention_minus_const.detach().cpu())

            norm_i = torch.norm(attention_minus_const, p=1)
            norm_i = norm_i / neighbourhood_indices.size(0)  # Can also get average norm per edge
            attention_norm = attention_norm + norm_i

        # print("attention norm total:", attention_norm.detach().cpu())
        attention_norm = attention_norm / torch.tensor(num_layers, device=self.device)

        # CLIP GRAD.
        attention_norm = torch.minimum(attention_norm, torch.tensor([10.0], device=self.device))
        # print("attention norm / layers:", attention_norm.detach().cpu())

        return attention_norm


if __name__ == "__main__":
    import run_config
    dataset = "PPI"
    config = run_config.data_config[dataset]
    model = induGAT(dataset=dataset, **config)

    print("Debug: Running a batch directly through indu_gat")
    ds = PPI(root='/tmp/PPI', split='train')
    dl = DataLoader(ds)
    # Get 1 batch from Cora
    batch = next(iter(dl))
    print(batch)

    # Get loss from indu_gat
    loss = model.training_step(batch, 0)

    loss.backward()

    #
    # # Run through GATModel's forward func
    # print("running GATModel forward func")
    # out, edge_index, first_attention, att_reg = model.forward_and_return_attention(batch)