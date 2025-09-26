"""
defines loss functions for training link prediction models in heterogeneous graphs
"""

import torch
from torch import Tensor
import torch.nn.functional as F


def bce_with_ground_truth(
    decoder,
    src_node_embeddings: Tensor,
    dst_node_embeddings: Tensor,
    all_edge_index: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Computes binary cross-entropy loss for edge prediction.
    """

    # Check for no edges
    if all_edge_index.numel() == 0:
        return torch.tensor(0.0, device=src_node_embeddings.device)

    logits = decoder(src_node_embeddings, dst_node_embeddings, all_edge_index)
    targets = targets.to(device=logits.device, dtype=logits.dtype)

    return F.binary_cross_entropy_with_logits(logits, targets)


def chain_regularizer(
    pos_relation_logits: Tensor, pos_edge_index: Tensor, num_src_nodes: int
) -> Tensor:
    """
    Computes a regularization loss if a source node is linked to more than 4 destination nodes.
    Source nodes, which is a "building" in our case, can only be linked to up to 4 "dline" nodes at most.
    """

    # Check for no edges
    if pos_edge_index.numel() == 0:
        return torch.tensor(0.0, device=pos_relation_logits.device)

    # Convert logits of positive edges into probabilities (sigmoid).
    # .detach() ensures it is not backpropagated through here (only used as weights).
    probs = torch.sigmoid(pos_relation_logits).detach()
    sums = torch.zeros(num_src_nodes, device=probs.device).scatter_add_(
        0, pos_edge_index[0], probs
    )

    # Penalize sums > 4 (using relu) and average the penalty.
    return torch.relu(sums - 4).mean()
