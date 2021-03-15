import torch.nn as nn

class GATLayer_2(nn.Module):

    def __init__(self, input_features, output_features, no_heads, dropout=0.6, concat=True):

        super(GATLayer_2, self).__init__()

        self.no_heads = no_heads
        self.input_features = input_features
        self.output_features = output_features
        self.ls = nn.Linear(input_features, no_heads*output_features, bias=False) # ls stands for linear sum which is
                                                                                  # the first linear projection defined
                                                                                  # in the paper
        # dimension here is 1 to denote that we are calculating the attention of 1 node
        # second dimension denotes the number of heads, which have independent feature representation
        # third dimension is just the number of output features from previous layers/representations

        self.self_attention_coefficient = nn.Parameter(torch.Tensor(1,no_heads, output_features)) #self-attention needs
                                                                                    # to be defined even if there is no
                                                                                    # self-loop
        self.neighbor_attention_coefficient = nn.Parameter(torch.Tensor(1,no_heads, output_features))


        #could have activation as an input to the class
        #activation = nn.ELU()
        #self.activation = activation
        self.leakyReLu = nn.LeakyRelu(0.2) # negative slope is 0.2 from the paper
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
        self.concat = concat
        self.reset_parameters()

    def forward(self, input):
        node_features, edge_index = input
        total_nodes = node_features.shape[0] # first dimension gives the total number of nodes from the input features
        node_features = self.dropout(node_features) # we apply a dropout on all input features as described in the paper
        node_features_projection = self.

