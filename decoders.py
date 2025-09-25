"""
Edge-scoring decoders for link prediction tasks in heterogeneous graphs.
Each decoder computes a score for edges based on the embeddings of source and destination nodes.
"""

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    Small multilayer perceptron (MLP) to score edges. It concatenates source and
    destination node embeddings, passes them through the MLP, and outputs a score.
    """

    def __init__(self, hid: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(
        self, src_node_embeddings, dst_node_embeddings, edge_index
    ) -> torch.Tensor:
        src, dst = (
            src_node_embeddings[edge_index[0]],
            dst_node_embeddings[edge_index[1]],
        )
        return self.mlp(torch.cat([src, dst], dim=-1)).squeeze(-1)


class DotDecoder(nn.Module):
    """
    Scores edges using the dot product of source and destination node embeddings.
    """

    def forward(self, src_node_embeddings, dst_node_embeddings, edge_index):
        src, dst = (
            src_node_embeddings[edge_index[0]],
            dst_node_embeddings[edge_index[1]],
        )
        return (src * dst).sum(-1)


class BilinearDecoder(nn.Module):
    """
    Scores edges using a bilinear transformation (learned weight matrix) between source and destination node embeddings.
    """

    def __init__(self, hid: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(hid, hid))
        nn.init.xavier_uniform_(self.W)

    def forward(
        self, src_node_embeddings, dst_node_embeddings, edge_index
    ) -> torch.Tensor:
        src, dst = (
            src_node_embeddings[edge_index[0]],
            dst_node_embeddings[edge_index[1]],
        )
        return (src @ self.W * dst).sum(-1)


def build_decoder(kind: str, hidden: int) -> nn.Module:
    """
    Factory function that returns the appropriate decoder based on the specified kind.
    """
    if kind == "mlp":
        return MLPDecoder(hidden)
    if kind == "dot":
        return DotDecoder()
    if kind == "bilinear":
        return BilinearDecoder(hidden)
    raise ValueError(f"Unknown decoder kind: {kind}")
