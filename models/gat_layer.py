import torch
from torch import nn

from .utils import add_remaining_self_loops


class GATLayer(nn.Module):
    """
    Inspired by both Aleksa Gordic's https://github.com/gordicaleksa/pytorch-GAT and PyTorch Geometric's GATConv layer,
    which we use as reference to test this implementation.

    This implementation follows the equations from the original GAT paper more faithfully, but will be less efficient than other optimised implementations.
    """
    def __init__(self, in_features, out_features, num_heads, concat, dropout=0, add_self_loops=False, bias=False, const_attention=False):
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
        super(GATLayer, self).__init__()

        print(f"tmp in_features={in_features}, out={out_features}, heads{num_heads}, concat={concat}")
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias
        self.const_attention = const_attention

        # Weight matrix from paper
        self.W = nn.Linear(in_features=self.in_features, out_features=self.num_heads*self.out_features, bias=False)
        # Attentional mechanism from paper
        # TODO trying to figure out shapes
        # self.a = nn.Parameter(torch.Tensor(1, self.num_heads, (2*self.out_features)))  # NH different matrices of size 2*F_OUT
        if not const_attention:
            self.a = nn.Linear(in_features=self.num_heads*(2*self.out_features), out_features=self.num_heads, bias=False)  # Attention coefficients
        
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        if self.bias:
            self.bias_param = nn.Parameter(torch.Tensor(self.num_heads * self.out_features))

        self.normalised_attention_coeffs = None
        self.reset_parameters()

    def forward(self, x, edge_index, return_attention_weights=False):
        """
        Compute attention-weighted representations of all nodes in x

        :param x: Feature matrix of size (N, in_features), where N is the number of nodes
        :param edge_index: Edge indices of size (2, E), where E is the number of edges.
        The edges point from the first row to second row, i.e. edge i = [231, 100] will be an edge that points from 231 to 100.
        :param return_attention_weights: Return a tuple (out, (edge_index, normalised_attention_coeffs))

        :return: New node representations of size (N, num_heads*out_features), optionally with attention coefficients
        """
        if self.add_self_loops:
            edge_index = add_remaining_self_loops(edge_index)

        N = x.size(0)
        E = edge_index.size(1)

        source_edges, target_edges = edge_index

        # Dropout (1) on input features is applied outside of the layer
        
        # Transform features
        nodes_transformed = self.W(x)  # (N, F_IN) -> (N, NH*F_OUT)
        nodes_transformed = nodes_transformed.view(N, self.num_heads, self.out_features)  # -> (N, NH, F_OUT)
        # # Dropout (2): on transformed features
        # if self.dropout > 0:
        #     nodes_transformed = nn.Dropout(p=self.dropout)(nodes_transformed)

        # Perform attention over neighbourhoods. Done in naive fashion (i.e. compute attention for all nodes)
        source_transformed = nodes_transformed[source_edges]   # shape: (E, NH, F_OUT)
        target_transformed = nodes_transformed[target_edges]   # shape: (E, NH, F_OUT)
        assert target_transformed.size() == (E, self.num_heads, self.out_features), f"{target_transformed.size()} != {(E, self.num_heads, self.out_features)}"

        if not self.const_attention:
            # Equation (1)
            attention_pairs = torch.cat([source_transformed, target_transformed], dim=-1)  # shape: (E, NH, 2*F_OUT)
            # Trying attention as a tensor
            # attention_weights = (self.a * attention_pairs).sum(dim=-1)  # Calculate dot product over last dimension (the output features) to get (E, NH)

            # (E, NH, 2*F_OUT) -> (E, NH*(2*F_OUT)): self.a expects an input of size (NH*(2*F_OUT))
            attention_pairs = attention_pairs.view(E, self.num_heads*(2*self.out_features))
            attention_weights = self.a(attention_pairs)  # shape: (E, NH*(2*F_OUT)) -> (E, NH)

            # We had to cap the range of logits because they were going to infinity on PPI
            attention_weights = attention_weights - attention_weights.max()
            attention_weights = nn.LeakyReLU()(attention_weights)
            print("att weights. Min: {}, Max: {}".format(attention_weights.min(), attention_weights.max()))
            print("Min of abs: {}".format(attention_weights.abs().min()))
            assert attention_weights.size() == (E, self.num_heads), f"{attention_weights.size()} != {(E, self.num_heads)}"
        else:
            # Setting to constant attention, see what happens
            # If attention_weights = 0, then e^0 = 1 so the exponentiated attention weights will = 1
            attention_weights = torch.zeros((E, self.num_heads))


        # TODO can probably multiply logits with representations and then denominator afterwards?
        # Softmax over neighbourhoods: Equation (2)/(3)
        attention_exp = attention_weights.exp()
        # Calculate the softmax denominator for each neighbourhood (target): sum attention exponents for each neighbourhood
        # output shape: (N, NH)
        attention_softmax_denom = self.sum_over_neighbourhood(
            values=attention_exp,
            neighbourhood_indices=target_edges,
            aggregated_shape=(N, self.num_heads),
        )

        # Broadcast back up to (E,NH) so that we can calculate softmax by dividing each edge by denominator
        attention_softmax_denom = torch.index_select(attention_softmax_denom, dim=0, index=target_edges)
        # normalise attention coeffs using a softmax operator.
        # Add an extra small number (epsilon) to prevent underflow / division by zero
        normalised_attention_coeffs = attention_exp / (attention_softmax_denom + 1e-8)  # shape: (E, NH)
        self.normalised_attention_coeffs = normalised_attention_coeffs  # Save attention weights

        # Dropout (3): on normalized attention coefficients
        if self.dropout > 0:
            normalised_attention_coeffs = self.dropout_layer(normalised_attention_coeffs)

        # Inside parenthesis of Equation (4):
        # Multiply all nodes in neighbourhood (with incoming edges) by attention coefficients
        weighted_neighbourhood_features = normalised_attention_coeffs.view(E, self.num_heads, 1) * source_transformed # target_transformed   # shape: (E, NH, F_OUT) * (E, NH, 1) -> (E, NH, F_OUT)
        assert weighted_neighbourhood_features.size() == (E, self.num_heads, self.out_features)

        # Equation (4):
        # Get the attention-weighted sum of neighbours. Aggregate again according to target edge.
        output_features = self.sum_over_neighbourhood(
            values=weighted_neighbourhood_features,
            neighbourhood_indices=target_edges,
            aggregated_shape=(N, self.num_heads, self.out_features),
        )

        print("Out features Min: {}, Max: {}".format(output_features.min(), output_features.max()))
        print("Min of abs: {}".format(output_features.abs().min()))
        # Equation (5)/(6)
        if self.concat:
            output_features = output_features.view(-1, self.num_heads*self.out_features)  # self.num_heads*self.out_features
        else:
            output_features = torch.mean(output_features, dim=1)  # Aggregate over the different heads

        if self.bias:
            output_features += self.bias_param

        if return_attention_weights:
            return output_features, (edge_index, self.normalised_attention_coeffs.detach().cpu())

        return output_features
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        # nn.init.xavier_uniform_(self.a)
        if not self.const_attention:
            nn.init.xavier_uniform_(self.a.weight)
        if self.bias:
            nn.init.zeros_(self.bias_param)
        # Can also init bias=0 if on

    def sum_over_neighbourhood(self, values: torch.tensor, neighbourhood_indices: torch.tensor, aggregated_shape, return_original_size: bool = False) -> torch.tensor:
        """
        Aggregate values over N neighbourhoods using torch.scatter_add, with some extra size checking.
        Optionally return the values broadcasted back up to original size after summing.

        TODO params
        :param values:
        :param neighbourhood_indices:
        :param aggregated_shape:
        :param return_original_size:
        :return:
        """

        # Create a new tensor in which to store the aggregated values. Created using the values tensor, so that the dtype and device match
        aggregated = values.new_zeros(aggregated_shape)

        # scatter_add requires target to match src's shape, e.g. needs to be of size (E, NH), not (E,)
        target_idx = explicit_broadcast(neighbourhood_indices, values)

        # Sum all elements according to the neighbourhood index. e.g. index=[0, 0, 0, 1, 1, 2], src=[1, 2, 3, 4, 5, 6] -> [1+2+3, 4+5, 6]
        aggregated.scatter_add_(dim=0, index=target_idx, src=values)  # shape: (E,NH) -> (N,NH)

        assert aggregated.size() == aggregated_shape, f"Aggregated size incorrect. Is {aggregated.size()}, should be {aggregated_shape}"

        return aggregated


# Copied helper function from Aleksa Gordic's pytorch-GAT repo:
#     https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L340
def explicit_broadcast(this, other):
    # Append singleton dimensions until this.dim() == other.dim()
    for _ in range(this.dim(), other.dim()):
        this = this.unsqueeze(-1)

    # Explicitly expand so that shapes are the same
    expanded = this.expand_as(other)
    assert expanded.size() == other.size(), f"Error: Broadcasting didn't work. Have size {expanded.size()}, expected {other.size()}"

    return expanded


if __name__ == "__main__":
    print("Debugging: Playing with Cora dataset")
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root='/tmp/cora', name='Cora')

    print(dataset[0])  # The entire graph is stored in dataset[0]
    model = GATLayer(in_features=1433, out_features=7, num_heads=3, concat=False)  # just playing around with 3 heads and 2 output features
    out = model.forward(dataset[0].x, dataset[0].edge_index)
    print(out.size())
    print(out)


# For reference:
#     def forward_i(self, x, edge_index, i=0):
#         """
#         Forward pass for node i. Useful for understanding.
#         forward() should be equivalent to running forward_i for all i?
#         """
#         N = x.size(0)
#         E = edge_index.size(1)
#
#         # Transform features
#         node_features = self.W(x)  # (N, F_IN) -> (N, F_OUT)
#         assert node_features.size() == (N, self.out_features)
#
#         # Perform attention on all incoming nodes with i TODO extend later
#         i_idx = (edge_index[1] == i)
#         in_neighbours_idx = edge_index[0][i_idx]
#         print("in_neighbours = ", in_neighbours_idx)
#
#         # Repeat node i's representation so that we can concat with all the neighbours
#         node_i_stacked = node_features[i].expand_as(node_features[in_neighbours_idx])
#         print(node_i_stacked.size(), node_features[in_neighbours_idx].size())
#         assert node_i_stacked.size() == node_features[in_neighbours_idx].size()
#
#         # Compute attention weights (scalars)
#         attention_weights_i = self.a(torch.cat([node_i_stacked, node_features[in_neighbours_idx]], dim=-1))
#         print("att weights: ", attention_weights_i.size(), in_neighbours_idx.size(0))
#         assert attention_weights_i.size() == (in_neighbours_idx.size(0), 1)
#
#         attention_weights_i = nn.LeakyReLU()(attention_weights_i)
#
#         softmax_attention = nn.Softmax(dim=0)(attention_weights_i)  # Softmax over j, the different neighbours (dim=0)
#
#         # Use attention weights for this neighbourhood to weight features
#         h_i = torch.sum(softmax_attention * node_features[in_neighbours_idx], dim=0)  # (num_neighbourhood, 1) * (num_neighbourhood, F_OUT) -> (F_OUT)
#         print('h_i', h_i.size())
#         assert h_i.size() == (self.out_features,)
#
#         return h_i
