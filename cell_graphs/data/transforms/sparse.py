import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, ToSparseTensor


class ToSparse(BaseTransform):
    """Transform a graph edge index into a sparse adjacency matrix format."""

    def __init__(self) -> None:
        self.sparse_transform = ToSparseTensor(attr="edge_attr")

    def __call__(self, data: Data) -> Data:
        """Transform the given graph into a sparse adjacency matrix format.

        Args:
        ----
            data: Graph to transform into a sparse adjacency matrix format.

        Returns:
        -------
        Transformed graph with sparse adjacency matrix in the `adj_t` attribute.
        """
        data.edge_attr = torch.ones(data.edge_index.shape[1], device=data.x.device)
        return self.sparse_transform(data)
