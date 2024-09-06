from typing import Any, Self

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from cell_graphs.common.utils.tensors import tensor_to_list


class GraphBag(Data):
    """Represents a bag of graphs for MIL."""

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> int:
        """Returns the incremental count when creating mini-batches with the `DataLoader`.

        Args:
        ----
            key: Key in the internal storage of the graph `Data` object.
            value: Corresponding value stored in the graph attribute.
            args: See `Data.__inc__`.
            kwargs: See `Data.__inc__`.

        Returns:
        -------
        Integer by which to increment the batch index.
        """
        if "edge_index_" in key:
            return getattr(self, f"x_{key[-1]}").size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def to_data_list(self) -> list[Data]:
        """Transforms a bag of graphs to a list of `Data` objects.

        Returns
        -------
        List of graphs in the bag.
        """
        self.remove_padding()
        attributes = self.to_dict()

        data_list = []
        for i in range(self.num_graphs):
            suffix = f"_{i}"
            data_kwargs = {
                key.removesuffix(suffix): value
                for key, value in attributes.items()
                if key.endswith(suffix)
            }
            data_list.append(Data(**data_kwargs))

        return data_list

    @classmethod
    def from_data_list(cls, data_list: list[Data]) -> Self:
        """Converts a list of `Data` objects into a `GraphBag`.

        Args:
        ----
            data_list: List of graphs to combine into a bag of graphs.

        Returns:
        -------
        A `GraphBag` containing the given graphs.
        """
        data_kwargs = {}
        for i, data in enumerate(data_list):
            data_kwargs.update({f"{key}_{i}": value for key, value in data.to_dict().items()})
        return cls(**data_kwargs, num_graphs=torch.tensor([len(data_list)]))

    def add_padding(self, max_num_graphs: int) -> None:
        """Pad the attributes for a maximum number of graphs per bag.

        Args:
        ----
            max_num_graphs: Maximum number of graphs present in a bag.
        """
        attrs = self.to_dict()
        keys_dtypes = [
            (key.removesuffix("_0"), attrs[key].dtype, isinstance(attrs[key], torch.Tensor))
            for key in attrs
            if "_0" in key
        ]

        for key, dtype, is_tensor in keys_dtypes:
            pad_dict = {
                f"{key}_{i}": torch.tensor([], dtype=dtype)
                if is_tensor
                else np.array([], dtype=dtype)
                for i in range(self.num_graphs, max_num_graphs)
            }
            self.update(pad_dict)

    def remove_padding(self) -> None:
        """Remove the padded attributes."""
        self.update(
            {
                k: None
                for k, v in self.to_dict().items()
                if isinstance(v, torch.Tensor) and v.numel() == 0
            }
        )


class GraphBagBatch(Batch):
    """Represents a batch of bags of graphs."""

    @classmethod
    def from_graph_bags(cls, graph_bags: list[GraphBag]) -> Self:
        """Constructs a batch of `GraphBag` instances.

        Args:
        ----
            graph_bags: Bags of graphs to combine into a batch of graphs.

        Returns:
        -------
        A batch of bags of graphs.

        Raises:
        ------
        RuntimeError if the sum of bag lengths does not match the maximum batch index.
        """
        data_list = []
        bag_lengths = []
        for graph_bag in graph_bags:
            graph_bag_list = []
            added_target = False
            for data in graph_bag.to_data_list():
                if data.y is not None:
                    # Reduce targets to one target per bag
                    if not added_target:
                        added_target = True
                    else:
                        data.y = torch.tensor([], dtype=data.y.dtype)

                graph_bag_list.append(data)

            bag_lengths.append(len(graph_bag_list))
            data_list.extend(graph_bag_list)

        batch = cls.from_data_list(data_list)
        batch.bag_lengths = torch.tensor(bag_lengths)

        if not torch.all(torch.sum(batch.bag_lengths) == batch.batch.max().item() + 1):
            raise RuntimeError("Sum of bag lenghts have to match highest batch index.")

        return batch

    def from_tensor(self, batch_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Unpacks a batch tensor based on the saved indices.

        Args:
        ----
            batch_tensor: Tensor representation of the batch.

        Returns:
        -------
        List of graph representations per graph bag in the batch.
        """
        return tensor_to_list(batch_tensor, self.bag_lengths)
