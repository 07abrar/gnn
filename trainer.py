import sys
from typing import Callable, Optional, Union

sys.path.insert(0, "/work/hosaka/src")

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from configs import (
    OptimConfig,
    SchedulerConfig,
    TrainConfig,
)
from hetero_gine import HeteroGINE
from losses import bce_with_ground_truth, chain_regularizer

# Define a type alias for edge types as tuples of (src_type, edge_type, dst_type)
edge_type = tuple[str, str, str]

# Type alias for scheduler factory functions
SchedulerFactory = Callable[
    [
        Optimizer,
        Optional[str],
        int,
        float,
        int,
    ],
    Optional[Union[_LRScheduler, ReduceLROnPlateau]],
]


class Trainer:
    """
    Implements the training engine for the heterogeneous GNN model.
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
        optimizer: Callable[
            [
                list[torch.nn.Parameter],
                str,
                float,
                float,
                tuple[float, float],
                float,
                float,
            ],
            Optimizer,
        ],
        scheduler: SchedulerFactory,
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
        self.device = torch.device(train_cfg.device)

    def to(self, device: torch.device) -> None:
        """Move the model and all decoder modules to the requested device."""

        self.device = device
        self.model.to(device)
        self.decoders.to(device)
        self.model.train()
        self.decoders.train()

    def __dec(self, relations: edge_type):
        """Helper function to get the decoder for a given relation."""
        return self.decoders[str(relations)]

    def __smooth(self, loss_history: list[float], k: int) -> float:
        """Helper function to smooth the loss history using a moving average."""
        if len(loss_history) < k:
            return sum(loss_history) / len(loss_history)
        else:
            return sum(loss_history[-k:]) / k

    def train(self, loader: DataLoader) -> float:
        # Set reproducibility and device
        torch.manual_seed(self.train_cfg.seed)
        self.to(self.device)

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
                batch = batch.to(self.device)
                self.model.train()
                self.opt.zero_grad()

                #  Collect edge attributes and node features from DataLoader batch
                edge_attr_dict = {
                    relation: batch[relation].edge_attr for relation in batch.edge_types
                }

                x_dict = {node_type: x for node_type, x in batch.x_dict.items()}
                edge_index_dict = {
                    relation: edge_index
                    for relation, edge_index in batch.edge_index_dict.items()
                }

                # Forward pass through the GNN to update the node embeddings
                node_embeddings = self.model(x_dict, edge_index_dict, edge_attr_dict)

                total_loss = torch.tensor(0.0, device=self.device)
                for relation in self.train_cfg.relations:
                    src_node_type, _, dst_node_type = relation
                    all_edge_index = batch[relation].edge_index
                    pos_edge_index = batch[relation].y.to(self.device)

                    relation_loss = bce_with_ground_truth(
                        self.__dec(relation),
                        node_embeddings[src_node_type],
                        node_embeddings[dst_node_type],
                        all_edge_index,
                        targets=batch[relation].bce_loss_target.to(self.device),
                    )

                    # Regularization to limit number of links per source node
                    if (
                        relation in self.train_cfg.chain_on
                        and pos_edge_index.numel() > 0
                    ):
                        pos_relation_logits = self.__dec(relation)(
                            node_embeddings[src_node_type],
                            node_embeddings[dst_node_type],
                            pos_edge_index,
                        )
                        relation_loss = (
                            relation_loss
                            + self.train_cfg.chain_lambda
                            * chain_regularizer(
                                pos_relation_logits,
                                pos_edge_index,
                                node_embeddings[src_node_type].size(0),
                            )
                        )

                    total_loss = total_loss + relation_loss

                # Backprop and optimization step
                total_loss.backward()
                self.opt.step()

            epoch += 1

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
            if self.sched and not isinstance(self.sched, ReduceLROnPlateau):
                self.sched.step()
        if self.sched and isinstance(self.sched, ReduceLROnPlateau):
            self.sched.step(float(total_loss.detach().item()))

        epoch += 1

        return total_loss.item()

    def save_checkpoint(self, path: str) -> None:
        """Persist the encoder and decoder weights to disk."""

        checkpoint = {
            "encoder": self.model.state_dict(),
            "decoder": {
                relation: self.decoders[str(relation)].state_dict()
                for relation in self.train_cfg.relations
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Load encoder and decoder weights from a checkpoint file."""

        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint["encoder"])
        for relation in self.train_cfg.relations:
            self.decoders[str(relation)].load_state_dict(
                checkpoint["decoder"][relation]
            )
