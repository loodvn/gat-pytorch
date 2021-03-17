import torch
import torch.nn as nn


class GATLayer_2(nn.Module):

    def __init__(self, input_features, output_features, num_heads, dropout=0.6, concat=True, activation=nn.ELU()):

        super(GATLayer_2, self).__init__()

        self.num_heads = num_heads
        self.input_features = input_features
        self.output_features = output_features
        self.ls = nn.Linear(input_features, num_heads * output_features, bias=False) # ls stands for linear sum which is
                                                                                     # the first linear projection defined
                                                                                     # in the paper

        # First dimension below is 1 to denote that we are calculating the attention of 1 node.
        # Second dimension denotes the number of heads, which have independent feature representation for each node.
        # Third dimension is just the number of output features from previous layers/representations

        self.self_attention_coefficient = nn.Parameter(torch.Tensor(1, num_heads, output_features))# self-attention needs
                                                                                    # to be defined even if there is no
                                                                                   # self-loop since we want to calculate
                                                                                   # self attention as well

        self.neighbor_attention_coefficient = nn.Parameter(torch.Tensor(1, num_heads, output_features))
        self.activation = activation        # ELU
        self.leakyReLu = nn.LeakyReLU(0.2)  # what they use for the paper
        self.dropout = nn.Dropout(dropout)  # dropout layer
        self.concat = concat                # denotes whether we are concatenating feature representations from heads or
                                            # not. If at the last layer, we shouldn't concatenate but we should use the
                                            # mean
        self.reset_parameters()

    def forward(self, x, edge_index, return_attention=False):
        node_features = x  # Node features shape: (N, F_IN)
        N = node_features.size(0)
        E = edge_index.size(1)  # number of edges
        node_features = self.dropout(node_features)  # we apply a dropout on all input features as described in the paper

        # we projecting the feature representations of the nodes
        projection_node_features = self.ls(node_features)  # shape: (N, F_IN) * (F_IN, NH*F_OUT) -> (N, NH*F_OUT)
        projection_node_features = projection_node_features.view(-1, self.num_heads, self.output_features)  # shape: (N, NH, F_OUT)
        projection_node_features = self.dropout(projection_node_features)  # dropout on features

        # calculate attention scores between edges
        self_node_attention = (projection_node_features * self.self_attention_coefficient).sum(dim=-1)
        print("projection node features shape is ", projection_node_features.shape)
        print("attention_coefficient shape is", self.self_attention_coefficient.shape)
        print("self_node_attention shape is ", self_node_attention.shape)
        # shape: (N, NH, F_OUT) * (1, NH, F_OUT) -> (N, NH, F_OUT) -> (N, F_OUT)

        # I believe this should be:
        # (N, NH, F_OUT) * (1, NH, F_OUT) -> (N, NH, 1) -> (N, NH)

        #assert self_node_attention.size() == (N, self.output_features)
        assert self_node_attention.size() == (N, self.num_heads)
        # same shape as self-attention
        neighbor_node_attention = (projection_node_features * self.neighbor_attention_coefficient).sum(dim=-1)

        # now we calculate the attention for the nodes that have common edges (i.e. are connected)
        # for this to be more readable, we create a helper function "score_calculation" and we call that in here
        source_scores, target_scores, node_projection_matrix = self.score_calculation(self_node_attention,
                                                                neighbor_node_attention, projection_node_features, edge_index)
        print("source_scores shape", source_scores.shape)
        #("source shape: ", source_scores.shape, "target shape: ", target_scores.shape, "node proj", node_projection_matrix.shape)
        edge_scores = self.leakyReLu(source_scores + target_scores)  # here we normalise the scores for edges by summing
                                                                         # over the scores calculated for the source index
                                                                         # and the target index
        print("edge_scores shape", edge_scores.shape)
        edge_attention = self.normalise_neighborhood(edge_scores, edge_index[1], N)
        edge_attention = self.dropout(edge_attention)

        # element-wise product
        print("error shape", node_projection_matrix.shape, edge_attention.shape)
        projection_normalised_neighborhood = torch.mul(node_projection_matrix, edge_attention)

        # now summing the weighted & projected feature vectors for all neighbors
        target_node_features = self.node_feature_normalised(projection_normalised_neighborhood, self.input_features, edge_index, N)

        output_node_features = self.node_features(target_node_features)

        # We return the node features and the edge_index, will probably be helpful
        # could also return something else if needed, need to see with the visualisation requirements

        return (output_node_features, edge_attention) if return_attention else output_node_features

    def score_calculation(self, source_node_score, target_node_score, projection_matrix, edge_index):

        source_index = edge_index[0]  # grab the source node from the edge index
        target_index = edge_index[1]  # target node from the edge index
        source_node_score = source_node_score.index_select(0, source_index) # could potentially change index_select to
                                                                            # usual indexing
        target_node_score = target_node_score.index_select(0, target_index)
        node_projection_matrix = projection_matrix.index_select(0, source_index)

        return source_node_score, target_node_score, node_projection_matrix

    def normalise_neighborhood(self, edge_scores, index, N):
        """ The function computes the softmax over the neighborhood of a node"""
        edge_scores_exp = edge_scores.exp()  # exponentiate the scores for the softmax calculation

        # now we need to calculate the sum of the scores of the neighborhood of the edge and then divide this with the
        # exponential edge scores for every edge calculated from before
        print("shape of edge_attention before", edge_scores_exp.shape)
        neighborhood_score = self.neighborhood_edge_sum(edge_scores_exp, index, N)
        print("neighborhood_score shape is ", neighborhood_score.shape)
        edge_attention = edge_scores_exp / neighborhood_score
        print("shape of edge_attention is", edge_attention.shape)
        # here we change the shape from (E,no_heads) to (E,no_heads,1) for element-wise multiplication
        edge_attention = edge_attention.unsqueeze(-1)

        return edge_attention

    def neighborhood_edge_sum(self, edge_scores, target_index, nodes):
        """The function returns the sum of the scores of the neighborhood of a particular node.
        This will be used in the softmax attention calculation as the denominator """
        target_index_heads = self.index_helper_function(target_index, edge_scores)
        list(edge_scores.shape)[0] = nodes  # recursively update the total nodes that will be the input of the next
        # calculation
        sum_of_neighborhood = torch.zeros(list(edge_scores.shape))

        # scatter_add_ sums all the values from the source tensor into the output indices
        sum_of_neighborhood.scatter_add_(0, target_index_heads, edge_scores)

        # return sum_of_neighborhood[target_index]
        return sum_of_neighborhood.index_select(0, target_index)

    def index_helper_function(self, input, output):
        """ This function will help with the calculations as it essentially appends singleton dimensions
            until the inserted dimensions match. This will help when calculating neighborhood attention scores
            in cases where the neighbors aren't the same"""

        for _ in range(input.dim(), output.dim()):
            input = input.unsqueeze(-1)
            # return first.expand_as(second)
        return input.expand(output.size())

    def node_feature_normalised(self, projection_normalised_neighborhood, edge_index, node_features_output, nodes):

        target_index_heads = self.index_helper_function(edge_index[1], projection_normalised_neighborhood)
        nodes = list(projection_normalised_neighborhood.shape)[0]
        sum_of_node_features = torch.zeros(list(projection_normalised_neighborhood.shape))
        node_features_output.scatter_add_(0, target_index_heads, projection_normalised_neighborhood)

        return node_features_output

    def node_features(self, output_node_features):

        if self.concat:
            output_node_features = output_node_features.view(-1, self.num_heads * self.output_features)

        else:
            output_node_features = output_node_features.mean(dim=1)

        return self.activation(output_node_features) if self.activation is not None else output_node_features

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ls.weight)  # TODO Check gain and ReLU stuff
        nn.init.xavier_uniform_(self.self_attention_coefficient)
        nn.init.xavier_uniform_(self.neighbor_attention_coefficient)


if __name__ == "__main__":
    print("Debugging: Playing with Cora dataset")
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/cora', name='Cora')
    #print(dataset[0])  # The entire graph is stored in dataset[0]
    model = GATLayer_2(input_features=1433, output_features=2, num_heads=3)  # just playing around with 3 heads and 2 output features
    model.forward(dataset[0].x, dataset[0].edge_index)

