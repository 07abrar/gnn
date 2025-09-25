import sys

sys.path.insert(0, "/work/hosaka/src")

import numpy as np
import torch
from torch.optim import Optimizer
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from configs import (
    OptimConfig,
    SchedulerConfig,
    TrainConfig,
)
from hetero_gine import HeteroGINE
from losses import (
    bce_pairwise,
    chain_regularizer,
)

# Define a type alias for edge types as tuples of (src_type, edge_type, dst_type)
edge_type = tuple[str, str, str]


class Trainer:
    """
    implements the training engine for the heterogeneous GNN model.
    Initializes model, decoders, optimizer, and scheduler based on provided configurations.
    """

    def __init__(
        self,
        # From configs.py
        model: HeteroGINE,
        decoders: dict[edge_type, torch.nn.Module],
        train_cfg: TrainConfig,
        optim_cfg: OptimConfig,
        sched_cfg: SchedulerConfig,
        # From optimizers.py
        optimizer: Optimizer,
        scheduler,
    ) -> None:
        """
        Wraps model, decoders (as a ModuleDict), optimizer/scheduler into one object.
        """

        # Store the GNN model and decoders inside ModuleDict
        self.model = model
        self.decoders = torch.nn.ModuleDict({str(k): v for k, v in decoders.items()})

        # Collect all parameters from model and decoders
        params = list(model.parameters()) + [
            p
            for decoder_module in decoders.values()
            for p in decoder_module.parameters()
        ]

        # Build optimizer and scheduler using config
        self.opt = optimizer(
            params,
            optim_cfg.name,
            optim_cfg.lr,
            optim_cfg.weight_decay,
            optim_cfg.betas,
            optim_cfg.momentum,
            optim_cfg.eps,
        )
        self.sched = scheduler(
            self.opt,
            sched_cfg.name,
            sched_cfg.step_size,
            sched_cfg.gamma,
            sched_cfg.T_max,
        )
        self.train_cfg = train_cfg

    def __dec(self, relations: edge_type):
        """Helper function to get the decoder for a given relation."""
        return self.decoders[str(relations)]

    def __smooth(self, loss_history: list[float], k: int) -> float:
        """Helper function to smooth the loss history using a moving average."""
        if len(loss_history) < k:
            return sum(loss_history) / len(loss_history)
        else:
            return sum(loss_history[-k:]) / k

    def __per_graph_losses(self, batch, node_embeddings):
        """
        Compute loss per relation and per blueprint inside a hetero mini-batch.
        Negatives are sampled only within the same blueprint.
        """
        device = torch.device(self.train_cfg.device)
        total_loss = torch.tensor(0.0, device=device)

        # Get edge indices for each relation
        # and call bce_pairwise for compute loss on positive and sampled negative
        for relation in self.train_cfg.relations:
            src_node_type, _, dst_node_type = relation

            # Gets the batched global edge index for this relation
            edge_index = batch[relation].edge_index.to(device)

            # Skip if no edges for this relation in the batch
            if edge_index.numel() == 0:
                continue

            # Fetches per-endpoint blueprint ids for all edges
            src_graph_id = batch[src_node_type].batch[edge_index[0]]
            dst_graph_id = batch[dst_node_type].batch[edge_index[1]]

            # Checks all endpoints of each edge belong to the same blueprint
            assert torch.equal(src_graph_id, dst_graph_id)

            # Iterates over each blueprint present in this mini-batch for this relation
            for graph_id in src_graph_id.unique():
                mask = src_graph_id == graph_id
                edge_index_graph = edge_index[:, mask]

                # Collects the global node indices of this blueprint for both node types
                src_local_nodes = torch.where(batch[src_node_type].batch == graph_id)[0]
                dst_local_nodes = torch.where(batch[dst_node_type].batch == graph_id)[0]

                # Creates global→local maps initialized to −1
                src_global_to_local = -torch.ones(
                    batch[src_node_type].num_nodes,
                    dtype=torch.long,
                    device=device,
                )
                dst_global_to_local = -torch.ones(
                    batch[dst_node_type].num_nodes,
                    dtype=torch.long,
                    device=device,
                )

                # Fills maps so nodes of this blueprint get local ids [0..n−1]
                src_global_to_local[src_local_nodes] = torch.arange(
                    src_local_nodes.numel(), device=device
                )
                dst_global_to_local[dst_local_nodes] = torch.arange(
                    dst_local_nodes.numel(), device=device
                )

                # Remaps the blueprint’s edges from global ids to local ids
                local_edge_index = torch.stack(
                    [
                        src_global_to_local[edge_index_graph[0]],
                        dst_global_to_local[edge_index_graph[1]],
                    ],
                    dim=0,
                )

                # Fetches the node embeddings for this blueprint
                src_node_emb = node_embeddings[src_node_type][src_local_nodes]
                dst_node_emb = node_embeddings[dst_node_type][dst_local_nodes]

                # Computes link loss with negatives sampled inside this blueprint only
                relation_loss = bce_pairwise(
                    self.__dec(relation),
                    src_node_emb,
                    dst_node_emb,
                    local_edge_index,
                    num_src_nodes=src_local_nodes.numel(),
                    num_dst_nodes=dst_local_nodes.numel(),
                )

                # Checks if chain penalty applies and there are edges
                if relation in self.train_cfg.chain_on and local_edge_index.numel() > 0:
                    relation_logits = self.__dec(relation)(
                        src_node_emb, dst_node_emb, local_edge_index
                    )

                    # Adds weighted chain regularization
                    relation_loss = (
                        relation_loss
                        + self.train_cfg.chain_lambda
                        * chain_regularizer(
                            relation_logits,
                            local_edge_index,
                            num_src_nodes=src_local_nodes.numel(),
                        )
                    )

                # Returns the total loss over all relations and blueprints
                total_loss = total_loss + relation_loss
        return total_loss

    def train(self, loader: DataLoader):
        # Set reproducibility and device
        torch.manual_seed(self.train_cfg.seed)
        device = torch.device(self.train_cfg.device)
        self.model.to(device)
        for decoder_module in self.decoders.values():
            decoder_module.to(device)

        lost_history = []
        best_loss = float("inf")
        epoch = 0

        # Record starting loss for percentage calculation and tqdm display
        initial_loss = 1.0
        target_loss = self.train_cfg.target_loss
        log_initial = np.log10(initial_loss)
        log_target = np.log10(target_loss)
        pbar = tqdm(total=100, desc="Training Progress (%)", unit="%")

        # Training loop
        while epoch < self.train_cfg.max_epochs and best_loss > target_loss:
            for batch in loader:
                batch = batch.to(device)
                self.model.train()
                self.opt.zero_grad()

                #  Collect edge attributes and node features from DataLoader batch
                edge_attr_dict = {
                    k: batch[k].edge_attr.to(device) for k in batch.edge_types
                }

                # Forward pass through the GNN -> node embeddings
                x_dict = {k: x.to(device) for k, x in batch.x_dict.items()}
                edge_index_dict = {
                    k: ei.to(device) for k, ei in batch.edge_index_dict.items()
                }
                node_embeddings = self.model(x_dict, edge_index_dict, edge_attr_dict)

                total_loss = self.__per_graph_losses(
                    batch,
                    node_embeddings,
                )

                # Backprop and optimization step
                total_loss.backward()
                self.opt.step()

                # Track and smooth the loss
                lost_history.append(float(total_loss.detach().item()))
                loss_avg = self.__smooth(lost_history, self.train_cfg.eval_smoothing)
                best_loss = min(best_loss, loss_avg)

                # Calculate percentage progress and update tqdm bar
                log_current = np.log10(loss_avg)
                p = (log_initial - log_current) / (log_initial - log_target) * 100
                pbar.n = int(p)
                pbar.postfix = {"loss": f"{loss_avg:.4e}", "epoch": epoch}
                pbar.refresh()

                # Step the scheduler if applicable
                if self.sched and self.sched.__class__.__name__ != "ReduceLROnPlateau":
                    self.sched.step()
            if self.sched and self.sched.__class__.__name__ == "ReduceLROnPlateau":
                self.sched.step(total_loss.detach())

            epoch += 1

        return total_loss.item()

    @torch.no_grad()
    def all_pairs_scores_batched(
        self,
        src_node_embeddings: torch.Tensor,
        dst_node_embeddings: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        edge_decoder,
    ):
        """
        Per-blueprint all-pairs scores inside a hetero mini-batch.
        Returns a dict[g] with local edge_index, probs, and local→global maps.
        """
        device = src_node_embeddings.device
        out = {}

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
        decoder,
    ):
        """
        Argmax per source within each blueprint. Returns:
        { graph_id: {src_global: (dst_global, prob)} }
        """
        ap = self.all_pairs_scores_batched(src_h, dst_h, src_batch, dst_batch, decoder)
        result = {}

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

    """
    The following train() method is works for a single HeteroData graph
    """
    # def train(self, data: HeteroData):
    #     # Set reproducibility and device
    #     torch.manual_seed(self.train_cfg.seed)
    #     device = torch.device(self.train_cfg.device)
    #     self.model.to(device)
    #     for decoder_module in self.decoders.values():
    #         decoder_module.to(device)

    #     lost_history = []
    #     best_loss = float("inf")
    #     epoch = 0

    #     # Record starting loss for percentage calculation and tqdm display
    #     initial_loss = 1.0
    #     target_loss = self.train_cfg.target_loss
    #     pbar = tqdm(total=100, desc="Training Progress (%)", unit="%")

    #     # Training loop
    #     while epoch < self.train_cfg.max_epochs and best_loss > target_loss:
    #         self.model.train()
    #         self.opt.zero_grad()

    #         #  Collect edge attributes and node features from HeteroData
    #         edge_attr_dict = {
    #             k: data[k].edge_attr.to(device) for k in data.edge_types
    #         }

    #         # Forward pass through the GNN -> node embeddings
    #         x_dict = {k: x.to(device) for k, x in data.x_dict.items()}
    #         edge_index_dict = {
    #             k: ei.to(device) for k, ei in data.edge_index_dict.items()
    #         }
    #         node_embeddings = self.model(
    #             x_dict,
    #             edge_index_dict,
    #             edge_attr_dict,
    #         )

    #         # Get edge indices for each relation
    #         # and call bce_pairwise for compute loss on positive and sampled negative edges
    #         total_loss = torch.tensor(0.0, device=device)
    #         for relation in self.train_cfg.relations:
    #             edge_index = data[relation].edge_index.to(device)
    #             src_type, dst_type = relation[0], relation[2]
    #             relation_loss = bce_pairwise(
    #                 self.__dec(relation),
    #                 node_embeddings[src_type],
    #                 node_embeddings[dst_type],
    #                 edge_index,
    #                 data[src_type].x.size(0),
    #                 data[dst_type].x.size(0),
    #             )

    #             # Optionally add chain regularization loss for specified relations
    #             if relation in self.train_cfg.chain_on:
    #                 relation_logits = self.__dec(relation)(
    #                     node_embeddings[src_type],
    #                     node_embeddings[dst_type],
    #                     edge_index,
    #                 )
    #                 relation_loss = (
    #                     relation_loss
    #                     + self.train_cfg.chain_lambda
    #                     * chain_regularizer(
    #                         relation_logits,
    #                         edge_index,
    #                         data[src_type].x.size(0),
    #                     )
    #                 )

    #             # Accumulate loss over all relations
    #             total_loss = total_loss + relation_loss

    #         # Backprop and optimization step
    #         total_loss.backward()
    #         self.opt.step()

    #         # Track and smooth the loss
    #         lost_history.append(float(total_loss.detach().item()))
    #         loss_avg = self.__smooth(
    #             lost_history, self.train_cfg.eval_smoothing
    #         )
    #         best_loss = min(best_loss, loss_avg)

    #         # Calculate percentage progress and update tqdm bar
    #         log_initial = np.log10(initial_loss)
    #         log_target = np.log10(target_loss)
    #         log_current = np.log10(loss_avg)
    #         p = (log_initial - log_current) / (log_initial - log_target) * 100
    #         pbar.n = p
    #         pbar.postfix = {"loss": f"{loss_avg:.1e}"}
    #         pbar.refresh()

    #         # Step the scheduler if applicable
    #         if (
    #             self.sched
    #             and hasattr(self.sched, "step")
    #             and self.sched.__class__.__name__ != "ReduceLROnPlateau"
    #         ):
    #             self.sched.step()
    #         if (
    #             self.sched
    #             and self.sched.__class__.__name__ == "ReduceLROnPlateau"
    #         ):
    #             self.sched.step(total_loss.detach())

    #         epoch += 1

    #     return total_loss.item()

    # @torch.no_grad()
    # def all_pairs_scores(
    #     self, src_node_embeddings, dst_node_embeddings, edge_decoder
    # ):
    #     """
    #     Returns all-pairs edge scores with sigmoid probabilities
    #     between every src and dst node using the provided edge decoder.
    #     """
    #     src_indices = torch.arange(
    #         src_node_embeddings.size(0), device=src_node_embeddings.device
    #     )
    #     dst_indices = torch.arange(
    #         dst_node_embeddings.size(0), device=src_node_embeddings.device
    #     )
    #     edge_index = torch.stack(
    #         torch.meshgrid(src_indices, dst_indices, indexing="ij"), dim=0
    #     ).view(2, -1)
    #     logits = edge_decoder(
    #         src_node_embeddings, dst_node_embeddings, edge_index
    #     )
    #     edge_probs = torch.sigmoid(logits)
    #     return edge_index, edge_probs
