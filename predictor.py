"""Utilities for running inference with a trained heterogeneous GNN model."""

from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import HeteroData

from hetero_gine import HeteroGINE

# Define a type alias for edge types as tuples of (src_type, edge_type, dst_type)
edge_type = tuple[str, str, str]


class Predictor:
    """Encapsulates inference utilities for a trained encoder and its decoders."""

    def __init__(
        self,
        model: HeteroGINE,
        decoders: torch.nn.ModuleDict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.decoders = decoders
        self.device = device

    def to(self, device: torch.device) -> None:
        """Move the encoder/decoders to ``device`` for inference."""

        self.device = device
        self.model.to(device)
        self.decoders.to(device)

    def eval(self) -> None:
        """Put the encoder/decoder modules into evaluation mode."""

        self.model.eval()
        self.decoders.eval()

    def __dec(self, relation: edge_type) -> torch.nn.Module:
        """Helper function to get the decoder for a given relation."""

        return self.decoders[str(relation)]

    def encode_batch(
        self, batch: HeteroData
    ) -> tuple[dict[str, torch.Tensor], HeteroData]:
        """Run the encoder on a mini-batch and return embeddings and device-aligned batch."""

        batch = batch.to(self.device)
        self.eval()
        with torch.no_grad():
            edge_attr_dict = {
                relation: batch[relation].edge_attr for relation in batch.edge_types
            }
            embeddings = self.model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)
        return embeddings, batch

    @torch.no_grad()
    def all_pairs_scores_batched(
        self,
        src_node_embeddings: torch.Tensor,
        dst_node_embeddings: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        relation: edge_type,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """
        Per-blueprint all-pairs scores inside a hetero mini-batch.
        Returns a dict[g] with local edge_index, probs, and local→global maps.
        """

        device = src_node_embeddings.device
        out: dict[int, dict[str, torch.Tensor]] = {}

        edge_decoder = self.__dec(relation)

        # iterate only over graphs present in both endpoints
        g_src = torch.unique(src_batch)
        g_dst = torch.unique(dst_batch)
        g_common = [int(g) for g in g_src.tolist() if (dst_batch == g).any()]

        for g in g_common:
            src_keep = (src_batch == g).nonzero(as_tuple=False).view(-1)
            dst_keep = (dst_batch == g).nonzero(as_tuple=False).view(-1)
            if src_keep.numel() == 0 or dst_keep.numel() == 0:
                continue

            s_emb = src_node_embeddings[src_keep]
            d_emb = dst_node_embeddings[dst_keep]

            s_ids = torch.arange(s_emb.size(0), device=device)
            d_ids = torch.arange(d_emb.size(0), device=device)
            edge_index = torch.stack(
                torch.meshgrid(s_ids, d_ids, indexing="ij"), dim=0
            ).view(2, -1)

            logits = edge_decoder(s_emb, d_emb, edge_index)
            probs = torch.sigmoid(logits)

            out[g] = {
                "edge_index_local": edge_index,  # [2, S*D] local ids
                "probs": probs,  # [S*D]
                "src_local_to_global": src_keep,  # map back to batch-global
                "dst_local_to_global": dst_keep,
            }
        return out

    @torch.no_grad()
    def best_pairs_batched(
        self,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        relation: edge_type,
    ) -> dict[int, dict[int, tuple[int, float]]]:
        """
        Argmax per source within each blueprint.
        Returns: { graph_id: {src_global: (dst_global, prob)} }
        """

        ap = self.all_pairs_scores_batched(src_h, dst_h, src_batch, dst_batch, relation)
        result: dict[int, dict[int, tuple[int, float]]] = {}

        for g, payload in ap.items():
            probs = payload["probs"]
            S = payload["src_local_to_global"].numel()
            D = payload["dst_local_to_global"].numel()
            P = probs.view(S, D)

            best_prob, best_dst_local = P.max(dim=1)
            src_gl = payload["src_local_to_global"]
            dst_gl = payload["dst_local_to_global"][best_dst_local]

            result[g] = {
                int(src_gl[i]): (int(dst_gl[i]), float(best_prob[i])) for i in range(S)
            }
        return result

    @torch.no_grad()
    def pairs_above_threshold_batched(
        self,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        relation: edge_type,
        threshold: float = 0.9,
    ) -> dict[int, list[tuple[int, int, float]]]:
        """Return all source/destination pairs whose probability exceeds ``threshold``."""

        all_pairs = self.all_pairs_scores_batched(
            src_h, dst_h, src_batch, dst_batch, relation
        )
        filtered: dict[int, list[tuple[int, int, float]]] = {}

        for graph_id, payload in all_pairs.items():
            probs = payload["probs"]
            keep = probs >= threshold
            if keep.any():
                local_edge_index = payload["edge_index_local"][:, keep]
                src_global = payload["src_local_to_global"][local_edge_index[0]]
                dst_global = payload["dst_local_to_global"][local_edge_index[1]]
                filtered[graph_id] = [
                    (int(src.item()), int(dst.item()), float(prob.item()))
                    for src, dst, prob in zip(src_global, dst_global, probs[keep])
                ]

        return filtered


def build_predictor(
    trainer: "Trainer", device: Optional[torch.device] = None
) -> Predictor:
    """Convenience helper to create a predictor from a trained ``Trainer`` instance."""

    from trainer import Trainer  # Local import to avoid circular dependency

    if not isinstance(trainer, Trainer):
        raise TypeError("trainer must be an instance of Trainer")

    predictor = Predictor(trainer.model, trainer.decoders, trainer.device)
    if device is not None:
        predictor.to(device)
    return predictor
