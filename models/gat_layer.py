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
    def __init__(self,in_channels, out_channels, dropout, alpha, concat = True):
        super(GATLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout # should be 0.6
        self.alpha = alpha # should be 0.2
        self.concat = concat

        # model should be initialised using Glorot(Xavier initialisation) which means:
        # leakyReLu with negative slope = 0.2
        # there is also an in-built initialisation in pytorch for Xavier initialisation

        self.weights = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        gain = nn.init.calculate_gain('relu') # this should return sqrt(2), which is the recommended value for the
                                              # non-linearity function relu

        nn.init.xavier_uniform_(self.weights.data, gain=gain)
        self.attention = nn.Parameter(torch.zeros(size=(2*out_channels,1))) # 2 because we will concatenate the two
                                                                            # representations and we need to match the
                                                                            # dimension

        nn.init.xavier_uniform_(self.attention.data, gain=gain)
        self.leakyReLu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        # not very sure what to put here maybe the xavier weights?

    def forward(self, x, immediate_neighbor):
        h = torch.mm(x, self.weights) # the initial linear transformation
        n = h.size()[0] # extracting the size of h (F in the paper)

        # attention
        input_attention = torch.cat([h.repeat(1, n).view(n*n, -1), h.repeat(n, 1)], dim=1).view(n, -1,
                                                                                                2*self.out_channels)
        attention_coefficients = self.leakyReLu(torch.matmul(input_attention, self.attention).squeeze(right_dimension))

                                                        # squeeze output in the right_dimension - not sure if it should
                                                        # be dimension 1 or 2, need to look at some data
        # masked attention

        #negative_vector = -10e20*torch.ones_like(attention_coefficients)
        masked_attention = torch.where(immediate_neighbor > 0, attention_coefficients, 0.)

                                         # we have attention only for immediate neighbors, otherwise we replace them
                                         # with 0. NOTE: instead of 0 maybe we can put something really negative,
                                         # like negative_vector
        # for immediate neighbors
        # otherwise we replace with 0
        # here instead of 0 maybe

        masked_attention = F.softmax(masked_attention, dim=1)
        masked_attention = F.dropout(masked_attention, self.dropout, training=self.training)
        h_hat = torch.matmul(masked_attention, h)

        # should return the weights in the forward to help with the visualisation
        weights = self.weights

        if self.concat:
            return F.elu(h_hat)
        else:
            return h_hat

