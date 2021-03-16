import torch.nn as nn


class GATLayer_2(nn.Module):

    def __init__(self, input_features, output_features, no_heads, dropout=0.6, concat=True, activation=nn.ELU()):

        super(GATLayer_2, self).__init__()

        self.no_heads = no_heads
        self.input_features = input_features
        self.output_features = output_features
        self.ls = nn.Linear(input_features, no_heads * output_features, bias=False) # ls stands for linear sum which is
                                                                                    # the first linear projection defined
                                                                                    # in the paper

        # Dimension below is 1 to denote that we are calculating the attention of 1 node.
        # Second dimension denotes the number of heads, which have independent feature representation for each node.
        # Third dimension is just the number of output features from previous layers/representations

        self.self_attention_coefficient = nn.Parameter(torch.Tensor(1, no_heads, output_features))# self-attention needs
                                                                                    # to be defined even if there is no
                                                                                   # self-loop since we want to calculate
                                                                                   # self attention as well

        self.neighbor_attention_coefficient = nn.Parameter(torch.Tensor(1, no_heads, output_features))
        self.activation = activation        # ELU
        self.leakyReLu = nn.LeakyRelu(0.2)  # what they use for the paper
        self.dropout = nn.Dropout(dropout)  # dropout layer
        self.concat = concat                # denotes whether we are concatenating feature representations from heads or
                                            # not. If at the last layer, we shouldn't concatenate but we should use the
                                            # mean
        self.reset_parameters()

    def forward(self, input_data, return_attention=False):
        node_features, edge_index = input_data
        node_features = self.dropout(node_features) # we apply a dropout on all input features as described in the paper

        # we projecting the feature representations of the nodes
        projection_node_features = self.ls(node_features)
        projection_node_features = projection_node_features.view(-1, self.no_heads, self.output_features)  # reshape
        projection_node_features = self.dropout(projection_node_features)  # dropping the features

        # calculate attention scores between edges
        self_node_attention = torch.sum(projection_node_features * self.self_attention_coefficient, dim=1)
        neighbor_node_attention = torch.sump(rojection_node_features * self.neighbor_attention_coefficient, dim=1)

        # now we calculate the attention for the nodes that have common edges (i.e. are connected)
        # for this to be more readable, we create a helper function "score_calculation" and we call that in here
        source_scores, target_scores, node_projection_matrix = self.score_calculation(self_node_attention,
                                                                neighbor_node_attention,projection_node_features,edge_index)
        edge_scores = self.leakyReLu(source_scores + target_scores)  # here we normalise the scores for edges by summing
                                                                         # over the scores calculated for the source index
                                                                         # and the target index
        total_nodes = node_features.shape[0]  # first dimension gives the total number of nodes from the input features
        edge_attention = self.normalise_neighborhood(edge_scores, edge_index[1], total_nodes)
        edge_attention = self.dropout(edge_attention)

        # element-wise product
        projection_normalised_neighborhood = torch.mul(node_projection_matrix, edge_attention)

        # now summing the weighted & projected feature vectors for all neighbors
        target_node_features = self.node_feature_normalised(projection_normalised_neighborhood, edge_index, total_nodes)

        output_node_features = self.node_features(edge_attention, node_features, target_node_features)

        # We return the node features and the edge_index, will probably be helpful
        # could also return something else if needed, need to see with the visualisation requirements

        return (output_node_features, edge_attention) if return_attention else output_node_features

    def score_calculation(self, source_node_score, target_node_score, projection_matrix, edge_index):

        source_index = edge_index[0]  # grab the source node from the edge index
        target_index = edge_index[1]  # target node from the edge index
        source_scores = source_scores[source_index]
        target_scores = target_scores[target_index]
        node_projection_matrix = projection_matrix[source_index]

        return source_scores, target_scores, node_projection_matrix

    def normalise_neighborhood(self, edge_scores, index, total_nodes):
        """ The function computes the softmax over the neighborhood of a node"""
        edge_scores = edge_scores.exp()  # exponentiate the scores

        # now we need to calculate the sum of the scores of the neighborhood of the edge and then divide this with the
        # exponential edge scores for every edge calculated from before
        neighborhood_score = self.neighborhood_edge_sum(edge_scores, index, total_nodes)
        edge_attention = edge_scores / neighborhood_score
        # here we change the shape from (E,no_heads) to (E,no_heads,1) for element-wise multiplication
        edge_attention = edge_attention.unsqueeze(-1)

        return edge_attention

    def neighborhood_edge_sum(self, edge_scores, index, nodes):
        """The function returns the sum of the scores of the neighborhood of a particular node.
        This will be used in the softmax attention calculation as the denominator """
        target_index = self.index_helper_function(index, edge_scores)
        nodes = list(edge_scores.shape)[0]  # recursively update the total nodes that will be the input of the next
        # calculation
        sum_of_neighborhood = torch.zeros(list(edge_scores.shape))

        # scatter_add_ sums all the values from the source tensor into the output indices
        sum_of_neighborhood.scatter_add_(0, target_index, edge_scores)

        return sum_of_neighborhood.index_select(0, target_index)

    def index_helper_function(self, first, second):
        """ This function will help with the calculations as it essentially appends singleton dimensions
            until the inserted dimensions match. This will help when calculating neighborhood attention scores
            in cases where the neighbors aren't the same"""

        for _ in range(input.dim(), output.dim()):
            input = input.unsqueeze(-1)
            # return first.expand_as(second)
        return input.expand(second.size())

    def node_feature_normalised(self, projection_normalised_neighborhood, edge_index, node_features_output, nodes):

        target_index = self.index_helper_function(edge_index[1], projection_normalised_neighborhood)
        nodes = list(projection_normalised_neighborhood.shape)[0]
        sum_of_node_features = torch.zeros(list(projection_normalised_neighborhood.shape))
        node_features_output.scatter_add_(0, target_index, projection_normalised_neighborhood)

        return node_features_output

    def node_features(self, output_node_features):

        if self.concat:
            output_node_features = output_node_features.view(-1, self.no_heads * self.output_features)

        else:
            output_node_features = output_node_features.mean(dim=1)

        return self.activation(output_node_features) if self.activation is not None else output_node_features

    def reset_parameters(self):
        nn.init.xavier_uniform(self.ls.weight)

        nn.init.xavier_uniform(self.self_attention_coefficient)
        nn.init.xavier_uniform(self.neighbor_attention_coefficient)
