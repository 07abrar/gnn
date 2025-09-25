"""
defines loss functions for training link prediction models in heterogeneous graphs
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


def bce_pairwise(
    decoder,
    src_node_embeddings,
    dst_node_embeddings,
    pos_edge_index,
    num_src_nodes: int,
    num_dst_nodes: int,
) -> Tensor:
    """
    Computes binary cross-entropy loss for edge prediction.
    - Uses positive edge indices and generates negative samples (non-existent edges) using negative_sampling.
    - Scores both positive and negative edges using the provided decoder.
    - Concatenates scores and creates target labels (1 for positive, 0 for negative).
    - Returns the binary cross-entropy loss between predicted logits and targets.
    """

    # Check for no edges
    if pos_edge_index.numel() == 0:
        z = torch.tensor(0.0, device=src_node_embeddings.device, requires_grad=True)
        return z

    # Generate negative samples
    neg_edge_index = negative_sampling(
        pos_edge_index,
        num_nodes=(num_src_nodes, num_dst_nodes),
        num_neg_samples=pos_edge_index.size(1),
        method="sparse",
    )

    # Score positive and negative edges
    pos_edge_scores = decoder(src_node_embeddings, dst_node_embeddings, pos_edge_index)
    neg_edge_scores = decoder(src_node_embeddings, dst_node_embeddings, neg_edge_index)

    # Concatenate scores into logits and create target labels (1 for pos, 0 for neg)
    logits = torch.cat([pos_edge_scores, neg_edge_scores])
    targets = torch.cat(
        [torch.ones_like(pos_edge_scores), torch.zeros_like(neg_edge_scores)]
    )

    # Binary cross-entropy with logits (BCEWithLogitsLoss) compares predicted logits vs. true labels.
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
