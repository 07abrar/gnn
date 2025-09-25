"""End-to-end smoke test for the heterogeneous GNN on dummy data."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch_geometric.data import HeteroData

from configs import (
    DecoderConfig,
    ModelConfig,
    OptimConfig,
    SchedulerConfig,
    TrainConfig,
)
from data import build_loader
from decoders import build_decoder
from hetero_gine import HeteroGINE
from optimizers import make_optimizer, make_scheduler
from trainer import Trainer


Relation = Tuple[str, str, str]


RELATIONS: List[Relation] = [
    ("building", "x_link", "dline"),
    ("building", "y_link", "dline"),
    ("dline", "name_link", "name"),
]


def full_edge_index(n_src: int, n_dst: int) -> torch.Tensor:
    """Create a full bipartite edge index between ``n_src`` and ``n_dst`` nodes."""

    src_indices = torch.arange(n_src)
    dst_indices = torch.arange(n_dst)
    src_grid, dst_grid = torch.meshgrid(src_indices, dst_indices, indexing="ij")
    return torch.stack((src_grid.flatten(), dst_grid.flatten()), dim=0)


def create_dummy_graph() -> HeteroData:
    """Return a synthetic ``HeteroData`` graph with known positive pairs."""

    torch.manual_seed(0)
    data = HeteroData()

    # Random node features
    data["building"].x = torch.randn(6, 6)
    data["dline"].x = torch.randn(10, 6)
    data["name"].x = torch.randn(6, 6)

    # Enumerate every possible edge for the relations of interest
    data[("building", "x_link", "dline")].edge_index = full_edge_index(6, 10)
    data[("building", "y_link", "dline")].edge_index = full_edge_index(6, 10)
    data[("dline", "name_link", "name")].edge_index = full_edge_index(10, 6)

    # Random edge attributes (2-dim) for every candidate edge
    for relation in RELATIONS:
        num_src = data[relation[0]].x.size(0)
        num_dst = data[relation[2]].x.size(0)
        data[relation].edge_attr = torch.randn(num_src * num_dst, 2)

    # Ground-truth positive edges that we expect the model to recover
    data[("building", "x_link", "dline")].pos_edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0], [5, 8, 6, 9, 7, 8]],
        dtype=torch.long,
    )
    data[("building", "y_link", "dline")].pos_edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 3, 1, 4, 2]],
        dtype=torch.long,
    )
    data[("dline", "name_link", "name")].pos_edge_index = torch.tensor(
        [[0, 1, 3, 5, 8, 9], [1, 4, 3, 0, 2, 5]],
        dtype=torch.long,
    )

    return data


def build_trainer(graph: HeteroData) -> Tuple[Trainer, torch.utils.data.DataLoader]:
    """Configure the model, decoders, and training components."""

    model_cfg = ModelConfig(
        node_feature_dimensions={"building": 6, "dline": 6, "name": 6}
    )
    dec_cfg = DecoderConfig(
        edge_dimensions={
            ("building", "x_link", "dline"): "mlp",
            ("building", "y_link", "dline"): "mlp",
            ("dline", "name_link", "name"): "mlp",
        },
        hidden=64,
    )
    optim_cfg = OptimConfig(name="adamw", lr=1e-3, weight_decay=1e-4)
    sched_cfg = SchedulerConfig(name=None)
    train_cfg = TrainConfig(
        target_loss=1e-3,
        max_epochs=1000,
        device="cpu",
        relations=RELATIONS,
    )

    edge_dimensions = {rel: graph[rel].edge_attr.size(1) for rel in RELATIONS}
    model = HeteroGINE(
        node_types=list(graph.x_dict.keys()),
        relations=RELATIONS,
        node_feature_dimensions=model_cfg.node_feature_dimensions,
        edge_dimensions=edge_dimensions,
        hidden_size=model_cfg.hidden_size,
        n_layers=model_cfg.n_layers,
        aggregates=model_cfg.aggregates,
        activation_func=model_cfg.activation_func,
        use_residual=model_cfg.use_residual,
    )

    decoders = {
        rel: build_decoder(dec_cfg.edge_dimensions[rel], dec_cfg.hidden)
        for rel in RELATIONS
    }

    trainer = Trainer(
        model,
        decoders,
        train_cfg,
        optim_cfg,
        sched_cfg,
        make_optimizer,
        make_scheduler,
    )

    loader = build_loader([graph], batch_size=1, shuffle=True)
    return trainer, loader


def run_training(trainer: Trainer, loader: torch.utils.data.DataLoader) -> None:
    """Train the model and report the best loss achieved."""

    best_loss = trainer.train(loader)
    print(f"Training finished with best loss {best_loss:.4e}")


def describe_predictions(
    title: str,
    predictions: Dict[int, List[Tuple[int, int, float]]],
    src_label: str,
    dst_label: str,
) -> None:
    """Pretty-print filtered predictions for readability."""

    print(f"\n{title}")
    if not predictions:
        print("  (no pairs above the requested threshold)")
        return

    for graph_id, edges in predictions.items():
        print(f"  Graph {graph_id}:")
        for src, dst, prob in edges:
            print(f"    {src_label} {src} -> {dst_label} {dst}: {prob:.2%}")


def run_inference(trainer: Trainer, graph: HeteroData, threshold: float = 0.8) -> None:
    """Score every candidate edge and list predictions above ``threshold``."""

    eval_loader = build_loader([graph], batch_size=1, shuffle=False)
    batch = next(iter(eval_loader)).to(trainer.train_cfg.device)

    trainer.model.eval()
    with torch.no_grad():
        edge_attr_dict = {
            relation: batch[relation].edge_attr
            for relation in trainer.train_cfg.relations
        }
        embeddings = trainer.model(batch.x_dict, batch.edge_index_dict, edge_attr_dict)

    relation_descriptions = {
        ("building", "x_link", "dline"): ("building", "dline", "Predicted X-links"),
        ("building", "y_link", "dline"): ("building", "dline", "Predicted Y-links"),
        ("dline", "name_link", "name"): ("dline", "name", "Predicted name-links"),
    }

    for relation in trainer.train_cfg.relations:
        src_type, _, dst_type = relation
        src_batch = batch[src_type].batch
        dst_batch = batch[dst_type].batch
        decoder = trainer.decoders[str(relation)]

        predictions = trainer.pairs_above_threshold_batched(
            embeddings[src_type],
            embeddings[dst_type],
            src_batch,
            dst_batch,
            decoder,
            threshold,
        )

        src_label, dst_label, title = relation_descriptions[relation]
        describe_predictions(title, predictions, src_label, dst_label)


def main() -> None:
    graph = create_dummy_graph()
    trainer, loader = build_trainer(graph)
    run_training(trainer, loader)
    run_inference(trainer, graph, threshold=0.8)


if __name__ == "__main__":
    main()
