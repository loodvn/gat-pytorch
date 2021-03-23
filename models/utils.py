from typing import Optional

import torch


def sum_over_neighbourhood(values: torch.tensor, neighbourhood_indices: torch.tensor, aggregated_shape,
                           broadcast_back: bool = False) -> torch.tensor:
    """
    Aggregate values over N neighbourhoods using torch.scatter_add, with some extra size checking.
    Optionally return the values broadcasted back up to original size after summing.

    TODO params
    :param values:
    :param neighbourhood_indices:
    :param aggregated_shape:
    :param broadcast_back:
    :return:
    """

    # Create a new tensor in which to store the aggregated values. Created using the values tensor, so that the dtype and device match
    aggregated = values.new_zeros(aggregated_shape)

    # scatter_add requires target to match src's shape, e.g. needs to be of size (E, NH), not (E,)
    target_idx = explicit_broadcast(neighbourhood_indices, values)

    # Sum all elements according to the neighbourhood index. e.g. index=[0, 0, 0, 1, 1, 2], src=[1, 2, 3, 4, 5, 6] -> [1+2+3, 4+5, 6]
    aggregated.scatter_add_(dim=0, index=target_idx, src=values)  # shape: (E,NH) -> (N,NH)

    if broadcast_back:
        aggregated = torch.index_select(aggregated, dim=0, index=neighbourhood_indices)

    assert aggregated.size() == aggregated_shape, f"Aggregated size incorrect. Is {aggregated.size()}, should be {aggregated_shape}"

    return aggregated


# Copied helper function from Aleksa Gordic's pytorch-GAT repo:
#     https://github.com/gordicaleksa/pytorch-GAT/blob/39c8f0ee634477033e8b1a6e9a6da3c7ed71bbd1/models/definitions/GAT.py#L340
# Added some error handling
def explicit_broadcast(this, other):
    # Append singleton dimensions until this.dim() == other.dim()
    for _ in range(this.dim(), other.dim()):
        this = this.unsqueeze(-1)

    # Explicitly expand so that shapes are the same
    expanded = this.expand_as(other)
    assert expanded.size() == other.size(), f"Error: Broadcasting didn't work. Have size {expanded.size()}, expected {other.size()}"

    return expanded


# Util functions taken from pytorch-geometric/utils/loop.py
# Modified slightly to remove edge weight handling
def add_remaining_self_loops(edge_index,
                             num_nodes: Optional[int] = None):
    r"""Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    mask = row != col

    loop_index = torch.arange(0, N, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index


def maybe_num_nodes(index: torch.Tensor,
                    num_nodes: Optional[int] = None) -> int:
    return int(index.max()) + 1 if num_nodes is None else num_nodes
