import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv


def train_cora():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    class GATCora(torch.nn.Module):
        def __init__(self):
            # From GAT paper, Section 3.3
            super(GATCora, self).__init__()
            self.gat1 = GATConv(in_channels=dataset.num_node_features, out_channels=8, heads=8)
            self.gat2 = GATConv(in_channels=8*8, out_channels=dataset.num_classes, concat=False)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            x = self.gat1(x, edge_index)
            x = self.gat2(x, edge_index)
            x = F.log_softmax(x)  # TODO ELU activation in gat2 already

            return x

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATCora()
    data = dataset[0].to(device)
    print("tmp: data", data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print("test")
    model.train()
    for epoch in range(1):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    print("eval")
    model.eval()
    for epoch in range(1):
        out = model(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.

    print(f"test acc: {test_acc}")


if __name__ == "__main__":
    # TOOD argparsing, could do one for each dataset?

    train_cora()
