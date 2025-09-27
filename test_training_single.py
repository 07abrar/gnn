import sys

sys.path.insert(0, "/work/hosaka/src")

import torch
from torch_geometric.data import HeteroData

from configs import (
    ModelConfig,
    DecoderConfig,
    OptimConfig,
    SchedulerConfig,
    TrainConfig,
)
from data import build_loader
from hetero_gine import HeteroGINE
from decoders import build_decoder
from optimizers import make_optimizer, make_scheduler
from trainer import Trainer
from predictor import Predictor, build_predictor

edge_type = tuple[str, str, str]


RELATIONS: list[edge_type] = [
    ("building", "x_link", "dline"),
    ("building", "y_link", "dline"),
    ("dline", "name_link", "name"),
]
NUM_EDGE_ATTRS = 2
NODE_TYPES = ["building", "dline", "name"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def full_edge_index(n_src: int, n_dst: int) -> torch.Tensor:
    """Create a full bipartite edge index between n_src and n_dst nodes."""
    src_indices = torch.arange(n_src)
    dst_indices = torch.arange(n_dst)
    src_grid, dst_grid = torch.meshgrid(src_indices, dst_indices, indexing="ij")
    return torch.stack((src_grid.flatten(), dst_grid.flatten()), dim=0)


def create_dummy_graph() -> HeteroData:
    torch.manual_seed(0)
    d = HeteroData()

    # Random node features for 6 buildings, 10 dim-lines, 6 names
    d[NODE_TYPES[0]].x = torch.randn(6, 6)
    d[NODE_TYPES[1]].x = torch.randn(10, 6)
    d[NODE_TYPES[2]].x = torch.randn(6, 6)

    # All possible edges for the relations of interest
    d[(NODE_TYPES[0], "x_link", NODE_TYPES[1])].edge_index = full_edge_index(6, 10)
    d[(NODE_TYPES[0], "y_link", NODE_TYPES[1])].edge_index = full_edge_index(6, 10)
    d[(NODE_TYPES[1], "name_link", NODE_TYPES[2])].edge_index = full_edge_index(10, 6)

    # Positive pairs (ground truth links to learn)
    positive_x = torch.tensor(
        [[0, 1, 2, 3, 4, 0], [5, 8, 6, 9, 7, 8]],
    )
    positive_y = torch.tensor(
        [[0, 1, 2, 3, 4], [0, 3, 1, 4, 2]],
    )
    positive_n = torch.tensor(
        [[0, 1, 3, 5, 8, 9], [1, 4, 3, 0, 2, 5]],
    )
    d["x_link"].y = positive_x
    d["y_link"].y = positive_y
    d["name_link"].y = positive_n

    for relation in RELATIONS:
        # Random edge attributes
        d[relation].edge_attr = torch.randn(
            (d[relation[0]].x.size(0) * d[relation[2]].x.size(0)),
            NUM_EDGE_ATTRS,
        )

        # Create target tensors for bce_with_ground_truth loss
        num_all_edge = d[relation].edge_index[1].shape
        target = torch.zeros(num_all_edge, dtype=torch.int)
        d[relation].bce_loss_target = target
        for i, ei in enumerate(d[relation].edge_index.T):
            for ei2 in d[relation[1]].y.T:
                if torch.all(ei == ei2):
                    d[relation].bce_loss_target[i] = 1.0

    return d


def build_trainer() -> Trainer:

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
        device=DEVICE,
        relations=RELATIONS,
    )

    edge_dimensions = {rel: NUM_EDGE_ATTRS for rel in RELATIONS}
    model = HeteroGINE(
        node_types=NODE_TYPES,
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

    return Trainer(
        model,
        decoders,
        train_cfg,
        optim_cfg,
        sched_cfg,
        make_optimizer,
        make_scheduler,
    )


def inference(
    predictor: Predictor,
    loader: torch.utils.data.DataLoader,
    relations: list[edge_type],
    threshold: float = 0.9,
) -> None:

    batch = next(iter(loader))
    embeddings, batch = predictor.encode_batch(batch)

    relation_descriptions = {
        ("building", "x_link", "dline"): (
            "building",
            "dline",
            "Predicted X-links",
        ),
        ("building", "y_link", "dline"): (
            "building",
            "dline",
            "Predicted Y-links",
        ),
        ("dline", "name_link", "name"): (
            "dline",
            "name",
            "Predicted name-links",
        ),
    }

    for relation in relations:
        src_type, _, dst_type = relation
        src_batch = batch[src_type].batch
        dst_batch = batch[dst_type].batch
        decoder = predictor.decoders[str(relation)]

        predictions = predictor.pairs_above_threshold_batched(
            embeddings[src_type],
            embeddings[dst_type],
            src_batch,
            dst_batch,
            relation,
            threshold,
        )

        src_label, dst_label, title = relation_descriptions[relation]
        print(f"\n{title}")
        if not predictions:
            print("  (no pairs above the requested threshold)")
            return

        for graph_id, edges in predictions.items():
            print(f"  Graph {graph_id}:")
            for src, dst, prob in edges:
                print(f"    {src_label} {src} -> {dst_label} {dst}: {prob:.2%}")


def main() -> None:
    graph = create_dummy_graph()
    trainer = build_trainer()
    loader = build_loader([graph], batch_size=1, shuffle=True)
    loss = trainer.train(loader)
    print(f"Training completed loss: {loss:.4f}")
    predictor = build_predictor(trainer)
    inference(predictor, loader, RELATIONS, threshold=0.9)

    # Save the model and decoder state_dicts
    trainer.save_checkpoint("model_and_decoders.pth")
    print("\nModel and decoders saved to 'model_and_decoders.pth'.")

    # Try loading the model and decoder state_dicts into a new Trainer instance
    new_trainer = build_trainer()
    new_trainer.load_checkpoint("model_and_decoders.pth", map_location=DEVICE)
    new_predictor = build_predictor(new_trainer, device=DEVICE)
    print("\nModel and decoders successfully loaded into a new Trainer instance.")
    inference(new_predictor, loader, RELATIONS, threshold=0.9)


if __name__ == "__main__":
    main()
