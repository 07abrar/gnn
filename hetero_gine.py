import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GINEConv

# Define a type alias for edge types as tuples of (src_type, edge_type, dst_type)
edge_type = tuple[str, str, str]


def _act(name: str) -> nn.Module:
    """Return the activation module specified by ``name``."""

    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation function: {name}")


def _mlp(hid: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))


class HeteroGINE(nn.Module):
    """
    Custom heterogeneous GNN model using PyTorch and PyTorch Geometric.
    This model implements a multi-layer GINE-based architecture for graphs with multiple node and edge types.
    """

    def __init__(
        self,
        node_types: list[str],
        relations: list[edge_type],
        node_feature_dimensions: dict[str, int],
        edge_dimensions: dict[edge_type, int],
        hidden_size: int = 64,
        n_layers: int = 2,
        aggregates: str = "sum",
        activation_func: str = "relu",
        use_residual: bool = True,
    ):
        super().__init__()
        self.relations = relations
        self.use_residual = use_residual
        self.activation_func = _act(activation_func)

        # One network per node type. Maps node features → hidden layer.
        self.input_projection_layers = nn.ModuleDict(
            {
                node_type: nn.Linear(node_feature_dimensions[node_type], hidden_size)
                for node_type in node_types
            }
        )

        # per-layer hetero convs (forward + reverse for each relation)
        layers = []  # accumulator for hidden layers
        for _ in range(n_layers):
            convs = {}
            for fwd in relations:
                # Define reverse relation
                rev = (
                    fwd[2],
                    f"rev_{fwd[1]}",
                    fwd[0],
                )

                # Forward GINEConv
                convs[fwd] = GINEConv(
                    _mlp(hidden_size),
                    train_eps=True,
                    edge_dim=edge_dimensions[fwd],
                )

                # Symmetric reverse GINEConv
                convs[rev] = GINEConv(
                    _mlp(hidden_size),
                    train_eps=True,
                    edge_dim=edge_dimensions[fwd],
                )
            layers.append(HeteroConv(convs, aggr=aggregates))
        self.layers = nn.ModuleList(layers)

    def __add_reverses(
        self,
        edge_index_dict: dict[edge_type, torch.Tensor],
        edge_attr_dict: dict[edge_type, torch.Tensor],
    ):
        """
        Takes dictionaries of edge indices and edge attributes for each edge type,
        and adds reverse edges for every relation in self.relations.
        """
        full_edge_index_dict, full_edge_attr_dict = dict(edge_index_dict), dict(
            edge_attr_dict
        )
        for fwd in self.relations:
            reverse_relation = (fwd[2], f"rev_{fwd[1]}", fwd[0])
            if fwd in edge_index_dict:
                full_edge_index_dict[reverse_relation] = edge_index_dict[fwd].flip(0)
                full_edge_attr_dict[reverse_relation] = edge_attr_dict[fwd]
        return full_edge_index_dict, full_edge_attr_dict

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Each node (x_dict) are projected to hidden space using the corresponding linear layer and activation function.
        Reverse edges and attributes are added to the edge_index_dict for bidirectional message passing.
        Then, for each GNN layer:
        - The layer processes the node embeddings and edge information.
        - An activation function is applied to the output of each layer.
        - If residual connections are enabled and it's not the first layer, the output is added to the previous embeddings.
        """

        # Project each node type to first hidden layer and apply activation.
        node_embeddings = {
            node_type: self.activation_func(
                self.input_projection_layers[node_type](features)
            )
            for node_type, features in x_dict.items()
        }

        # TODO Comment out. Maybe not need to make reverse edges/attrs.
        # Augment edges/attrs with reverse relations for message passing.
        full_edge_index_dict, full_edge_attr_dict = self.__add_reverses(
            edge_index_dict, edge_attr_dict
        )

        for layer_idx, conv in enumerate(self.layers):

            # TODO: GINEConv input all edge_attr and edge_index
            # Update node embeddings by passing through the hidden layer.
            updated_node_embeddings = conv(
                node_embeddings, full_edge_index_dict, full_edge_attr_dict
            )

            # Nonlinearity after the layer for all node types.
            updated_node_embeddings = {
                node_type: self.activation_func(v)
                for node_type, v in updated_node_embeddings.items()
            }

            # Residual connection from previous layer outputs, starting at layer 1.
            if self.use_residual and layer_idx > 0:
                node_embeddings = {
                    node_type: node_embeddings[node_type]
                    + updated_node_embeddings[node_type]
                    for node_type in node_embeddings
                }
            else:
                node_embeddings = updated_node_embeddings
        return node_embeddings
