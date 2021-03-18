import torch
from torch import nn
from .utils import add_remaining_self_loops


class GATLayerLood(nn.Module):
    """Minimal GAT Layer, playing around"""
    def __init__(self, in_features, out_features, num_heads, concat, dropout=0, add_self_loops=True, bias=True, activation=nn.ELU()):
        """
        TODO docstring
        :param in_features:
        :param out_features:
        :param num_heads:
        :param concat:
        :param dropout:
        :param add_self_loops:
        :param bias:
        """
        super(GATLayerLood, self).__init__()

        print(f"tmp in_features={in_features}, out={out_features}, heads{num_heads}, concat={concat}")
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = 0 #dropout
        self.add_self_loops = add_self_loops
        self.bias = bias

        # Weight matrix from paper
        self.W = nn.Linear(in_features=self.in_features, out_features=self.num_heads*self.out_features, bias=self.bias)
        # Attentional mechanism from paper
        self.a = nn.Linear(in_features=self.num_heads*(2*self.out_features), out_features=self.num_heads, bias=self.bias)  # Attention coefficients
        self.normalised_attention_coeffs = None

    def forward(self, x, edge_index, return_attention_coeffs=False):
        """
        Compute attention-weighted representations of all nodes in x

        :param x: Feature matrix of size (N, in_features), where N is the number of nodes
        :param edge_index: Edge indices of size (2, E), where E is the number of edges.
        The edges point from the first row to second row, i.e. edge i = [231, 100] will be an edge that points from 231 to 100.
        :param return_attention_coeffs: Return a tuple (out, (edge_index, normalised_attention_coeffs))

        :return: New node representations of size (N, num_heads*out_features), optionally with attention coefficients
        """
        if self.add_self_loops:
            edge_index = add_remaining_self_loops(edge_index)

        N = x.size(0)
        E = edge_index.size(1)

        source_edges, target_edges = edge_index
        
        # Dropout on input features
        if self.dropout > 0:
            x = nn.Dropout(p=self.dropout)(x)
        
        # Transform features
        node_features = self.W(x).view(N, self.num_heads, self.out_features)  # (N, F_IN) -> (N, NH*F_OUT)
        if self.dropout > 0:
            node_features = nn.Dropout(p=self.dropout)(node_features)

        # Perform attention over neighbourhoods. Done in naive fashion (i.e. compute attention for all nodes)
        source_representations = node_features[source_edges]   # shape: (E, NH, F_OUT)
        target_representations = node_features[target_edges]   # shape: (E, NH, F_OUT)
        assert target_representations.size() == (E, self.num_heads, self.out_features), f"{target_representations.size()} != {(E, self.num_heads, self.out_features)}"

        # Equation (1)
        attention_pairs = torch.cat([source_representations, target_representations], dim=-1)
        # (E, NH, 2*F_OUT) -> (E, NH*(2*F_OUT)): self.a expects an input of size (NH*(2*F_OUT))
        attention_pairs = attention_pairs.view(E, self.num_heads*(2*self.out_features))
        attention_weights = self.a(attention_pairs)  # shape: (E, NH*(2*F_OUT)) -> (E, NH)  # TODO the heads are mixing here (input fully connected to NH output) which is wrong
        attention_weights = nn.LeakyReLU()(attention_weights)
        assert attention_weights.size() == (E, self.num_heads), f"{attention_weights.size()} != {(E, self.num_heads)}"

        # TODO can probably multiply logits with representations and then denominator afterwards?
        # Softmax over neighbourhoods: Equation (2)/(3)
        attention_exp = attention_weights.exp()
        # Calculate the softmax denominator for each neighbourhood (target)
        softmax_denom = attention_exp.new_zeros((N, self.num_heads))  # Create new_zeros from attention_exp, so that the dtype matches nicely
        # Sum all elements according to the target edge. e.g. index=[0, 0, 0, 1, 1, 2], src=[1, 2, 3, 4, 5, 6] -> [1+2+3, 4+5, 6]
        # target needs to match src's shape according to scatter_add, so needs to be of size (E, NH), not (E,)
        target_idx = target_edges.unsqueeze(-1).expand_as(attention_exp)
        assert target_idx.size() == attention_exp.size()
        softmax_denom.scatter_add_(dim=0, index=target_idx, src=attention_exp)  # shape: (E,NH) -> (N,NH)
        assert softmax_denom.size() == (N, self.num_heads), f"{softmax_denom.size()} != {(N, self.num_heads)}"
        # Broadcast back up to (E,NH) so that we can calculate softmax
        softmax_denom = torch.index_select(softmax_denom, dim=0, index=target_edges)

        normalised_attention_coeffs = attention_exp / softmax_denom  # shape: (E, NH) TODO add epsilon for stability?
        self.normalised_attention_coeffs = normalised_attention_coeffs  # Save attention weights

        # Dropout on normalized attention coefficients
        if self.dropout > 0:
            normalised_attention_coeffs = nn.Dropout(p=self.dropout)(normalised_attention_coeffs)

        # Multiply representations by attention coefficients: Equation (4)
        weighted_target_representations = normalised_attention_coeffs.view(E, self.num_heads, 1) * target_representations   # shape: (E, NH, F_OUT) * (E, NH, 1) -> (E, NH, F_OUT)

        output_features = node_features.new_zeros((N, self.num_heads, self.out_features))
        target_idx = target_edges.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_target_representations)  # TODO this is ugly
        # Get the attention-weighted sum of neighbours. Aggregate again according to target edge.
        output_features.scatter_add_(dim=0, index=target_idx, src=weighted_target_representations)
        assert output_features.size() == (N, self.num_heads, self.out_features)

        # Equation (5)/(6)
        if self.concat:
            output_features = output_features.view(-1, self.num_heads*self.out_features)  # self.num_heads*self.out_features
        else:
            output_features = torch.mean(output_features, dim=1)  # Aggregate over the different heads

        print("tmp output features shape: ", output_features.size())

        if return_attention_coeffs:
            return output_features, (edge_index, self.normalised_attention_coeffs)

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
        node_i_stacked = node_features[i].expand_as(node_features[in_neighbours_idx])
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
