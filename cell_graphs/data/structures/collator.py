import torch
from torch_geometric.data import Batch

from cell_graphs.data.structures.graph_bag import GraphBag, GraphBagBatch


class GraphCollator:
    """Converts a list of `GraphBag`s into a batch during data loading."""

    def __call__(self, batch: list[GraphBag]) -> tuple[Batch, torch.Tensor]:
        """Collates a list of `GraphBag`s into a batch and corresponding targets.

        Args:
        ----
            batch: List of `GraphBag`s to collate.

        Returns:
        -------
        A tuple of the form (X, y) where X is a batch of graph bags and y are the targets.
        """
        collated = GraphBagBatch.from_graph_bags(batch)
        return collated, collated.y
