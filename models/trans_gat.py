import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid

from models.GATModel import GATModel


# pl.seed_everything(42)
from models.utils import sum_over_neighbourhood, explicit_broadcast


class transGAT(GATModel):
    def __init__(self, attention_reward=0.0, **config):
        super().__init__(**config)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.attention_reward = attention_reward

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

    # Useful for checking if gradients are flowing
    # def on_after_backward(self):
    #     print("On backwards")
    #     # print(self.attention_reg_sum.grad)
    #     print("w: ", self.gat_layer_list[0].W.weight.grad)
    #     print("a:", self.gat_layer_list[0].a.weight.grad)
    #     print("normalised", self.gat_layer_list[0].normalised_attention_coeffs.grad)
    #     # print(self.gat_model[0].attention_reg_sum)
    # grad_fn = loss.grad_fn
    # for i in range(10):
    #     grad_fn = grad_fn.next_functions[0][0]
    #     print("loss trace:", loss.grad_fn.next_functions[0][0])

    def validation_step(self, batch, batch_idx):  # In Cora, there is only 1 batch (the whole graph)
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
        # OW: TODO - Use these attention weights.
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
            if self.logger is not None:  # TODO can log individual heads later
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
        # print("attention norm / layers:", attention_norm.detach().cpu())

        return attention_norm


if __name__ == "__main__":
    import run_config
    config = run_config.data_config['Cora']
    model = transGAT(dataset="Cora", **config)

    ds = Planetoid(root='/tmp/Cora', name="Cora")
    dl = DataLoader(ds)
    # Get 1 batch from Cora
    batch = next(iter(dl))
    print(batch)

    # Get loss from trans_gat
    loss = model.training_step(batch, 0)

    loss.backward()

    # Check normalised attention coeffs
    print("normalised att: ", model.gat_layer_list[0].normalised_attention_coeffs)

    # Run through GATModel's forward func
    print("running GATModel forward func")
    out, edge_index, attention_list = model.forward_and_return_attention(batch)
    print("debug: first attention head:", attention_list[0])



