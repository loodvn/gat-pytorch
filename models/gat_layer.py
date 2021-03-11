import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(torch.nn.Module): # TODO could also create a MessagePassing class first and use that, similar to how torch_geometric's GATConv class works
    """
    This is a GAT.
    """

    # Extra possible params from torch_geometric: (add_self_loops: bool = True, bias: bool = True, **kwargs for MessagePassing)
    def __init__(self, in_channels, out_channels, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()

        # model should be initialised using Glorot(Xavier initialisation) which means:
        # leakyReLu with negative slope = 0.2
        # there is also an in-built initialisation in pytorch for Xavier initialisation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout # should be 0.6
        self.alpha = alpha # used for leakyRelu
        self.concat = concat # except for the last layer

        self.w = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_uniform(self.w.data, gain=1.2)  # not sure what the gain should be, probably need to
                                                         # experiment for this

        self.vector_a = nn.Parameter(torch.zeros(size=(2*out_channels, 1)))
        nn.init.xavier_uniform(self.vector_a.data, gain=1.2)
        self.leakyRelu = nn.LeakyReLU(self.alpha)

        # TODO some validation on args

        self.reset_parameters()

    def forward(self, x, adjacent):
        h = torch.mm(x, self.w)
        n = h.size()[0]

        # attention


