"""
Centralized configuration for heterogeneous GNN model, decoder, optimizer, scheduler, and training settings.
"""

from dataclasses import dataclass, field
from typing import Optional

# Define a type alias for edge types as tuples of (src_type, edge_type, dst_type)
edge_type = tuple[str, str, str]


@dataclass
class ModelConfig:
    """
    Configures the GNN model, including input dimensions per node type (the number of features),
    hidden layer size, number of layers, aggregation method, activation function,
    and whether to use residual connections.
    """

    # The number of features per node type
    node_feature_dimensions: dict[str, int] = field(
        default_factory=lambda: {"building": 6, "dline": 6, "name": 6}
    )
    hidden_size: int = 64  # hidden size
    n_layers: int = 2
    aggregates: str = "sum"  # sum|mean|max
    activation_func: str = "relu"  # relu|gelu
    use_residual: bool = True


@dataclass
class DecoderConfig:
    """
    Specifies how to decode edge relations (e.g., between "building" and "dline")
    using different methods (MLP, dot, bilinear), and sets the hidden size for the decoder.
    """

    edge_dimensions: dict[edge_type, str] = field(
        default_factory=lambda: {
            ("building", "x_link", "dline"): "mlp",  # mlp|dot|bilinear
            ("building", "y_link", "dline"): "mlp",
            ("dline", "name_link", "name"): "mlp",
        }
    )
    hidden: int = 64


@dataclass
class OptimConfig:
    """
    Sets optimizer parameters like type (adam, sgd, etc.), learning rate, weight decay, betas, momentum, and epsilon.
    """

    name: str = "adam"  # adam|adamw|sgd|rmsprop
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)  # used by Adam/AdamW
    momentum: float = 0.9  # used by SGD/RMSprop
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """
    Configures learning rate scheduler options, such as type (step, cosine, etc.), step size, gamma, and maximum steps.
    """

    name: Optional[str] = None  # step|cosine|plateau|None
    step_size: int = 50
    gamma: float = 0.5
    T_max: int = 200


@dataclass
class TrainConfig:
    """
    Model training settings, including number of epochs, regularization lambda,
    device, random seed, and which relations to use for training and chaining.
    """

    target_loss: float = 1e-4
    max_epochs: int = 1000
    eval_smoothing: int = 20  # smoothing window for eval loss
    chain_lambda: float = 0.1
    device: str = "cpu"
    seed: int = 0
    relations: list[edge_type] = field(
        default_factory=lambda: [
            ("building", "x_link", "dline"),
            ("building", "y_link", "dline"),
            ("dline", "name_link", "name"),
        ]
    )
    chain_on: list[edge_type] = field(
        default_factory=lambda: [
            ("building", "x_link", "dline"),
            ("building", "y_link", "dline"),
        ]
    )


@dataclass
class DataConfig:
    """Data loading parameters, including batch size and whether to shuffle the data."""

    batch_size: int = 4
    shuffle: bool = True