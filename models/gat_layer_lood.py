# tmp
import torch
from torch import nn


class GATLayerLood(nn.Module):
    """Minimal GAT Layer, playing around"""
    def __init__(self, in_features, out_features, num_heads, concat):
        super(GATLayerLood, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.W = nn.Linear(in_features=self.in_features, out_features=self.num_heads*self.out_features, bias=False)
        self.a = nn.Linear(in_features=2*(self.num_heads*self.out_features), out_features=self.num_heads, bias=False)  # Attention coefficients

        self.concat = concat # TODO could have some more aggregation functions

    def forward(self, x, edge_index):
        """
        Compute attention-weighted representations of all nodes in x
        :param x: Feature matrix of size (N, in_features), where N is the number of nodes
        :param edge_index: Edge indices of size (2, E), where E is the number of edges.
        The edges point from the first row to second row, i.e. edge i = [231, 100] will be an edge that points from 231 to 100.

        :return: New node representations of size (N, num_heads*out_features)

        TODO return_attention_weights, aggregation concat
        """

        N = x.size(0)
        E = edge_index.size(1)

        source_edges, target_edges = edge_index

        # Transform features
        node_features = self.W(x).view(N, self.num_heads, self.out_features)  # (N, F_IN) -> (N, NH*F_OUT)

        # Perform attention over neighbourhoods. Done in naive fashion (i.e. compute attention for all nodes)
        source_representations = node_features[source_edges]   # shape: (E, NH, F_OUT)
        target_representations = node_features[target_edges]   # shape: (E, NH, F_OUT)
        assert target_representations.size() == (E, self.num_heads, self.out_features), f"{target_representations.size()} != {(E, self.num_heads, self.out_features)}"

        cat = torch.cat([source_representations, target_representations], dim=-1)
        cat = cat.view(E, self.num_heads*(2*self.out_features))
        attention_weights = self.a(cat)  # shape: (E, NH, 2*F_OUT) -> (E, NH*(2*F_OUT)) -> (E, NH, 1)
        print("att size: ", attention_weights.size())
        attention_weights = attention_weights.squeeze(-1)  # Squeeze last dim
        attention_weights = nn.LeakyReLU()(attention_weights)
        assert attention_weights.size() == (E, self.num_heads), f"{attention_weights.size()} != {(E, self.num_heads)}"

        # TODO can probably multiply logits with representations and then denominator afterwards?
        # Softmax over neighbourhoods
        attention_exp = attention_weights.exp()
        # Calculate the softmax denominator for each neighbourhood (target)
        softmax_denom = attention_exp.new_zeros((N, self.num_heads))  # Create new_zeros from attention_exp, so that the dtype matches nicely
        # Sum all elements according to the target edge. e.g. index=[0, 0, 0, 1, 1, 2], src=[1, 2, 3, 4, 5, 6] -> [1+2+3, 4+5, 6]
        # target needs to match src's shape according to scatter_add, so needs to be of size (E, NH), not (E,)
        target_idx = target_edges.unsqueeze(-1).expand_as(attention_exp)
        assert target_idx.size() == attention_exp.size()
        softmax_denom.scatter_add_(dim=0, index=target_idx, src=attention_exp)  # shape: (E,NH) -> (N,NH)
        assert softmax_denom.size() == (N, self.num_heads), f"{softmax_denom.size()} != {(N, self.num_heads)}"
        # Broadcast back up to (E,1) so that we can calculate softmax
        softmax_denom = torch.index_select(softmax_denom, dim=0, index=target_edges)

        softmax = attention_exp / softmax_denom  # shape: (E, NH) TODO add epsilon for stability?
        print("softmax shape: ", softmax.size())

        # Multiply representations by attention coefficients
        weighted_target_representations = target_representations * softmax.view(E, self.num_heads, 1)  # shape: (E, NH, F_OUT) * (E, NH, 1) -> (E, NH, F_OUT)
        print(weighted_target_representations.size())

        output_features = node_features.new_zeros((N, self.num_heads, self.out_features))
        target_idx = target_edges.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_target_representations)  # TODO this is ugly
        # Aggregate again according to target edge.
        output_features.scatter_add_(dim=0, index=target_idx, src=weighted_target_representations)
        assert output_features.size() == (N, self.num_heads, self.out_features)

        if self.concat:
            output_features = output_features.view(-1, self.num_heads*self.out_features)  # self.num_heads*self.out_features
        else:
            output_features = torch.mean(output_features, dim=-1)  # Aggregate over the different heads

        return output_features

    def forward_i(self, x, edge_index, i=0):
        """
        Forward pass for node i. Useful for understanding.
        forward() should be equivalent to running forward_i for all i?
        """
        N = x.size(0)
        E = edge_index.size(1)

        # Transform features
        node_features = self.W(x)  # (N, F_IN) -> (N, F_OUT)
        assert node_features.size() == (N, self.out_features)

        # Perform attention on all incoming nodes with i TODO extend later
        i_idx = (edge_index[1] == i)
        in_neighbours_idx = edge_index[0][i_idx]
        print("in_neighbours = ", in_neighbours_idx)

        # Repeat node i's representation so that we can concat with all the neighbours
        node_i_stacked = node_features[0].expand_as(node_features[in_neighbours_idx])
        print(node_i_stacked.size(), node_features[in_neighbours_idx].size())
        assert node_i_stacked.size() == node_features[in_neighbours_idx].size()

        # Compute attention weights (scalars)
        attention_weights_i = self.a(torch.cat([node_i_stacked, node_features[in_neighbours_idx]], dim=-1))
        print("att weights: ", attention_weights_i.size(), in_neighbours_idx.size(0))
        assert attention_weights_i.size() == (in_neighbours_idx.size(0), 1)

        attention_weights_i = nn.LeakyReLU()(attention_weights_i)

        softmax_attention = nn.Softmax(dim=0)(attention_weights_i)  # Softmax over j, the different neighbours (dim=0)

        # Use attention weights for this neighbourhood to weight features
        h_i = torch.sum(softmax_attention * node_features[in_neighbours_idx], dim=0)  # (num_neighbourhood, 1) * (num_neighbourhood, F_OUT) -> (F_OUT)
        print('h_i', h_i.size())
        assert h_i.size() == (self.out_features,)

        return h_i


if __name__ == "__main__":
    print("Debugging: Playing with Cora dataset")
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/cora', name='Cora')

    print(dataset[0])  # The entire graph is stored in dataset[0]
    model = GATLayerLood(in_features=1433, out_features=2, num_heads=3, concat=False)  # just playing around with 3 heads and 2 output features
    out = model.forward(dataset[0].x, dataset[0].edge_index)
    print(out.size())
    print(out)
