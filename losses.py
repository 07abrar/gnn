"""Utility losses for the heterogeneous link prediction experiments."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def bce_with_ground_truth(
    decoder,
    src_node_embeddings: Tensor,
    dst_node_embeddings: Tensor,
    all_edge_index: Tensor,
    pos_edge_index: Tensor,
    num_dst_nodes: int,
) -> Tensor:
    """Binary cross entropy over *all* candidate edges for a relation.

    Parameters
    ----------
    decoder:
        Edge decoder used to score pairs of nodes.
    src_node_embeddings / dst_node_embeddings:
        Node embeddings for the source and destination node types.
    all_edge_index:
        Edge index describing every candidate pair considered during training
        (e.g. the full Cartesian product between the node sets).
    pos_edge_index:
        Ground-truth positive edges for the relation.
    num_dst_nodes:
        Number of destination nodes in ``dst_node_embeddings``.  This is used to
        build a unique identifier for each edge so that positives can be matched
        against ``all_edge_index`` efficiently.

    Notes
    -----
    ``all_edge_index`` is expected to contain every positive edge listed in
    ``pos_edge_index``.  The function does not perform negative sampling; instead
    it assigns the target label ``1`` to the provided positive edges and ``0`` to
    every other candidate edge.
    """

    if all_edge_index.numel() == 0:
        return torch.tensor(0.0, device=src_node_embeddings.device)

    device = src_node_embeddings.device
    logits = decoder(src_node_embeddings, dst_node_embeddings, all_edge_index)
    targets = torch.zeros_like(logits, device=device)

    if pos_edge_index.numel() > 0:
        # Build identifiers ``src * num_dst_nodes + dst`` for all candidates and
        # positives.  Sorting + searchsorted keeps the implementation vectorised
        # and differentiable (targets are constructed without touching ``logits``).
        edge_ids = all_edge_index[0] * num_dst_nodes + all_edge_index[1]
        sorted_edge_ids, ordering = torch.sort(edge_ids)

        pos_ids = pos_edge_index[0] * num_dst_nodes + pos_edge_index[1]
        insertion_points = torch.searchsorted(sorted_edge_ids, pos_ids)

        valid = (insertion_points < sorted_edge_ids.numel()) & (
            sorted_edge_ids[insertion_points] == pos_ids
        )
        if valid.any():
            matched_positions = ordering[insertion_points[valid]]
            targets[matched_positions] = 1.0

    return F.binary_cross_entropy_with_logits(logits, targets)


def chain_regularizer(pos_edge_scores, pos_edge_index, num_src_nodes: int):
    """
    Regularizes the sum of predicted probabilities for each source node.
    - Converts logits to probabilities.
    - Sums probabilities for each source node using scatter_add_.
    - Penalizes cases where the sum exceeds 1 (using relu), and averages the penalty.
    """

    # Check for no edges
    if pos_edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos_edge_scores.device)

    # Convert logits of positive edges into probabilities (sigmoid).
    # .detach() ensures it is not backpropagated through here (only used as weights).
    probs = torch.sigmoid(pos_edge_scores).detach()
    sums = torch.zeros(num_src_nodes, device=probs.device).scatter_add_(
        0, pos_edge_index[0], probs
    )

    # Penalize sums > 1 (using relu) and average the penalty.
    return torch.relu(sums - 1).mean()
