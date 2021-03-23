import torch
from sklearn.metrics import f1_score
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI

from models.GATModel import GATModel


# pl.seed_everything(42)

# OW test
# Addition 


class induGAT(GATModel):
    def __init__(self, **config):    
        super().__init__(**config)
        self.criterion = BCEWithLogitsLoss(reduction='mean')

    def training_step(self, batch, batch_idx):
        
        l1_lambda = 0.001

        # Get the outputs from the forwards function, the edge index and the tensor of attention weights.
        out, edge_index, first_attention, _ = self.forward_and_return_attention(batch, return_attention_coeffs=True)  # attention_weights_list

        loss_fn = BCEWithLogitsLoss(reduction='mean')
        loss = loss_fn(out, batch.y)

        # Penalise deviation from constant attention (const-GAT), for analysing the gain of attention

        # Copied gat_layer.aggregate_neighbourhood below
        neighbourhood_indices = edge_index[1]

        # Create a new tensor in which to store the aggregated values. Created using the values tensor, so that the dtype and device match
        degrees = first_attention.new_zeros(first_attention.size())

        # scatter_add requires target to match src's shape, e.g. needs to be of size (E, NH), not (E,)
        target_idx = neighbourhood_indices.unsqueeze(-1).expand_as(first_attention)

        # Sum all elements according to the neighbourhood index. e.g. index=[0, 0, 0, 1, 1, 2], src=[1, 2, 3, 4, 5, 6] -> [1+2+3, 4+5, 6]
        degrees.scatter_add_(dim=0, index=target_idx, src=torch.ones_like(first_attention))  # shape: (E,NH) -> (N,NH)

        # Broadcast back up to (E,NH) so that we can calculate softmax by dividing each edge by denominator
        degrees = torch.index_select(degrees, dim=0, index=neighbourhood_indices)
        print("new degrees shape", degrees.size())

        unnormalised_attention = first_attention * degrees

        print("unnormalised_attention", unnormalised_attention.detach().cpu(), unnormalised_attention.size())

        attention_minus_const = unnormalised_attention - 1.0

        print("attention_minus const", attention_minus_const.detach().cpu())

        norm_loss = l1_lambda * torch.norm(attention_minus_const, p=1)
        print("bce loss: ", loss.detach().cpu())
        print(f"norm loss with lambda = {l1_lambda.detach().cpu()}", norm_loss.detach().cpu())

        loss = loss + norm_loss
        print("total_loss", loss.detach().cpu())

        self.log('train_loss', loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average='micro')
        self.log('train_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_after_backward(self):
        print("On backwards")
        # print(self.attention_reg_sum.grad)
        # print(self.gat_model[0].W.weight.grad)
        # print(self.gat_model[0].a.weight.grad)
        # print(self.gat_model[0].normalised_attention_coeffs.grad)

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        loss_fn = BCEWithLogitsLoss(reduction='mean') 
        loss = loss_fn(out, batch.y)
        self.log('val_loss', loss.detach().cpu(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('val_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch)

        f1 = f1_score(y_pred=out.detach().cpu().numpy() > 0, y_true=batch.y.detach().cpu().numpy(), average="micro")
        self.log('test_f1_score', f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return f1

    # This should dynamically choose dataset class - not use PPI by default
    def prepare_data(self):
        self.train_ds = PPI(root='/tmp/PPI', split='train')
        self.val_ds = PPI(root='/tmp/PPI', split='val')
        self.test_ds = PPI(root='/tmp/PPI', split='test')


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
    # # Check normalised attention coeffs
    # print("normalised att: ", model.gat_model[0].normalised_attention_coeffs)
    print("W grad: ", model.gat_model[0].W.weight.grad)
    #
    # # Run through GATModel's forward func
    # print("running GATModel forward func")
    # out, edge_index, first_attention, att_reg = model.forward_and_return_attention(batch)