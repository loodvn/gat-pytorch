import torch


class GATLayer(torch.nn.Module): # TODO could also create a MessagePassing class first and use that, similar to how torch_geometric's GATConv class works
    """
    This is a GAT.
    """

    # Extra possible params from torch_geometric: (add_self_loops: bool = True, bias: bool = True, **kwargs for MessagePassing)
    def __init__(self, in_channels, out_channels, heads=1, concat=True, elu_slope=0.2, dropout=0.0):
        super(GATLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # TODO some validation on args

        self.reset_parameters()

    def forward(self, x, edge_index):
        pass

