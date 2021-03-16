import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    # TODO could also create a MessagePassing class first and use that, similar to how torch_geometric's GATConv class works
    # i.e. class MessagePassing(torch.nn.Module): ...
    """
    GAT layer implementation
    """

    # Extra possible params from torch_geometric: (add_self_loops: bool = True, bias: bool = True, **kwargs for MessagePassing)
    def __init__(self, in_channels, out_channels, dropout, alpha, number_of_heads, concat=True, include_skip_connection = False):
        super(GATLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout  # should be 0.6
        self.alpha = alpha  # should be 0.2
        self.number_of_heads = number_of_heads
        self.concat = concat
        self.include_skip_connection = include_skip_connection # useful for the inductive implementation

        # model should be initialised using Glorot(Xavier initialisation) which means:
        # leakyReLu with negative slope = 0.2
        # there is also an in-built initialisation in pytorch for Xavier initialisation

        self.weights = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        gain = nn.init.calculate_gain('relu')  # this should return sqrt(2), which is the recommended value for the
                                               # non-linearity function relu

        nn.init.xavier_uniform_(self.weights.data, gain=gain)
        self.attention = nn.Parameter(torch.zeros(size=(2 * out_channels, 1))) # 2 because we will concatenate the two
                                                                               # representations and we need to match the
                                                                               # dimension

        nn.init.xavier_uniform_(self.attention.data, gain=gain)
        1
        if include_skip_connection_skip_connection:
            self.skip_projections = nn.Linear(in_channels, no_of_heads * out_channels, bias=False)

        self.leakyReLu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # not very sure what to put here maybe the xavier weights?

    def forward(self, x, immediate_neighbor):
        h = torch.mm(x, self.weights)  # the initial linear transformation
        n = h.size()[0]  # extracting the size of h (F in the paper)

        # attention
        input_attention = torch.cat([h.repeat(1, n).view(n * n, -1), h.repeat(n, 1)], dim=1).view(n, -1, 2 * self.out_channels)

        attention_coefficients = self.leakyReLu(torch.matmul(input_attention, self.attention))
        attention_coefficients = attention_coefficients.squeeze(2)

        # masked attention
        negative_vector = -10e20*torch.ones_like(attention_coefficients)
        masked_attention = torch.where(immediate_neighbor > 0, attention_coefficients, negative_vector) # to avoid any
                                                                # computation with 0, as that might create problems with
                                                                # backpropagation and attention updates

        masked_attention = F.softmax(masked_attention, dim=1)
        masked_attention = F.dropout(masked_attention, self.dropout, training=self.training)
        h_hat = torch.matmul(masked_attention, h)

        # should return the weights in the forward to help with the visualisation
        weights = self.weights

        if self.concat:
            return F.elu(h_hat)
        else:
            return h_hat

    def skip_connections(self, input_node_features, output_node_features):

        if self.include_skip_connection:
            if output_node_features.shape[-1] == input_node_features.shape[-1]:
                output_node_features += input_node_features.unsqueeze(1)
            else:
                output_node_features += self.skip_projection(input_nodes_features).view(-1, self.number_of_heads, self.out_channels)

        if self.concat:
            output_node_features = output_node_features.view(-1, self.number_of_heads * self.out_channels)
        else:
            output_node_features = output_node_features.mean(dim=1)

