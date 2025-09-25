"""
Entry point for datasets and loaders.
"""

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


class BlueprintDataset(torch.utils.data.Dataset):
    """
    Wrapper for a list of HeteroData objects. This class derived from Dataset abstract class.
    """

    def __init__(self, graphs: list[HeteroData]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return self.graphs[i]


def build_loader(graphs: list[HeteroData], batch_size: int = 4, shuffle: bool = True):
    """
    A helper function that creates a DataLoader for batching and shuffling graphs from a BlueprintDataset.
    """
    return DataLoader(BlueprintDataset(graphs), batch_size=batch_size, shuffle=shuffle)
