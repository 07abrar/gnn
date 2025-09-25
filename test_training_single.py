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
from optimizers import (
    make_optimizer,
    make_scheduler,
)
from trainer import Trainer


def full_edge_index(n_src: int, n_dst: int) -> torch.Tensor:
    """Create a full bipartite edge index between n_src and n_dst nodes."""
    src_indices = torch.arange(n_src)
    dst_indices = torch.arange(n_dst)
    src_grid, dst_grid = torch.meshgrid(src_indices, dst_indices, indexing="ij")
    return torch.stack((src_grid.flatten(), dst_grid.flatten()), dim=0)


# 1) Test data
torch.manual_seed(0)
d = HeteroData()

# 6 buildings, 10 dim-lines, 6 names
d["building"].x = torch.randn(6, 6)  # 6 buildings, 6-dim features
d["dline"].x = torch.randn(10, 6)  # 10 dim-lines, 6-dim features
d["name"].x = torch.randn(6, 6)  # 6 names, 6-dim features

# All possible node
all_x_link = full_edge_index(6, 10)
all_y_link = full_edge_index(6, 10)
all_name_link = full_edge_index(10, 6)
d[("building", "x_link", "dline")].edge_index = all_x_link
d[("building", "y_link", "dline")].edge_index = all_y_link
d[("dline", "name_link", "name")].edge_index = all_name_link

# Edge attributes
for relation in [
    ("building", "x_link", "dline"),
    ("building", "y_link", "dline"),
    ("dline", "name_link", "name"),
]:
    d[relation].edge_attr = torch.randn(
        (d[relation[0]].x.size(0) * d[relation[2]].x.size(0)), 2
    )  # 2-dim edge features

# Positive pairs (ground truth links to learn)
positive_x = torch.tensor([[0, 1, 2, 3, 4, 0], [5, 8, 6, 9, 7, 8]], dtype=torch.long)
positive_y = torch.tensor([[0, 1, 2, 3, 4], [0, 3, 1, 4, 2]], dtype=torch.long)
positive_n = torch.tensor([[0, 1, 3, 5, 8, 9], [1, 4, 3, 0, 2, 5]], dtype=torch.long)
d["x_link"].y = positive_x
d["y_link"].y = positive_y
d["name_link"].y = positive_n

# 2) configs
model_cfg = ModelConfig(node_feature_dimensions={"building": 6, "dline": 6, "name": 6})
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
    relations=[
        ("building", "x_link", "dline"),
        ("building", "y_link", "dline"),
        ("dline", "name_link", "name"),
    ],
)

# 3) build model
relations = train_cfg.relations
node_types = list(d.x_dict.keys())
edge_dimensions = {rel: d[rel].edge_attr.size(1) for rel in relations}
model = HeteroGINE(
    node_types=node_types,
    relations=relations,
    node_feature_dimensions=model_cfg.node_feature_dimensions,
    edge_dimensions=edge_dimensions,
    hidden_size=model_cfg.hidden_size,
    n_layers=model_cfg.n_layers,
    aggregates=model_cfg.aggregates,
    activation_func=model_cfg.activation_func,
    use_residual=model_cfg.use_residual,
)

# 4) decoders
decoders = {
    rel: build_decoder(dec_cfg.edge_dimensions[rel], dec_cfg.hidden)
    for rel in relations
}

# 5) train
trainer = Trainer(
    model,
    decoders,
    train_cfg,
    optim_cfg,
    sched_cfg,
    make_optimizer,
    make_scheduler,
)
loader = build_loader([d], batch_size=1, shuffle=True)

# trainer.train(d)
trainer.train(loader)


# 6) inference: best pair per src for x/y/name links
@torch.no_grad()
def best_pairs(src_h: torch.Tensor, dst_h: torch.Tensor, decoder):
    # uses trainer.all_pairs_scores() which enumerates Cartesian product in ij order
    ei, p = trainer.all_pairs_scores(src_h, dst_h, decoder)
    n_src, n_dst = src_h.size(0), dst_h.size(0)
    P = p.view(n_src, n_dst)  # [num_src, num_dst]
    best_prob, best_dst = P.max(dim=1)  # argmax per source
    return {int(s): (int(best_dst[s]), float(best_prob[s])) for s in range(n_src)}


# with torch.no_grad():
#     edge_attr_dict = {k: d[k].edge_attr for k in d.edge_types}
#     h = model(d.x_dict, d.edge_index_dict, edge_attr_dict)

#     rel_x = ("building", "x_link", "dline")
#     rel_y = ("building", "y_link", "dline")
#     rel_n = ("dline", "name_link", "name")

#     best_x = best_pairs(h["building"], h["dline"], decoders[rel_x])
#     best_y = best_pairs(h["building"], h["dline"], decoders[rel_y])
#     best_n = best_pairs(h["dline"], h["name"], decoders[rel_n])

#     print("Best X-links per building {building -> (dline, prob)}:", best_x)
#     print("Best Y-links per building {building -> (dline, prob)}:", best_y)
#     print("Best name-links per dline {dline -> (name, prob)}:", best_n)

with torch.no_grad():
    batch = next(iter(loader)).to("cpu")
    edge_attr = (
        {k: batch[k].edge_attr for k in batch.edge_types}
        if hasattr(batch[next(iter(batch.edge_types))], "edge_attr")
        else {}
    )
    h = model(batch.x_dict, batch.edge_index_dict, edge_attr)

    rel_x = ("building", "x_link", "dline")
    rel_y = ("building", "y_link", "dline")
    rel_n = ("dline", "name_link", "name")

    best_x = trainer.best_pairs_batched(
        h["building"],
        h["dline"],
        batch["building"].batch,
        batch["dline"].batch,
        decoders[rel_x],
    )
    best_y = trainer.best_pairs_batched(
        h["building"],
        h["dline"],
        batch["building"].batch,
        batch["dline"].batch,
        decoders[rel_y],
    )
    best_n = trainer.best_pairs_batched(
        h["dline"],
        h["name"],
        batch["dline"].batch,
        batch["name"].batch,
        decoders[rel_n],
    )
    print("best_x:", best_x)
    print("best_y:", best_y)
    print("best_n:", best_n)
