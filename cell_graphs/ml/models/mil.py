from abc import ABC, abstractmethod
from copy import copy

import torch
from torch import nn

from cell_graphs.data.structures.graph_bag import GraphBagBatch
from cell_graphs.ml.models.fuser import AdditionFuser


class MILModel(nn.Module, ABC):
    """Base module to implement multiple-instance learning (MIL)."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        output_dim: int,
        fused_input_dim: int | None = None,
    ) -> None:
        """Args:
        ----
            backbone: Backbone encoder for individual instances.
            embedding_dim: Dimension of instance representations.
            output_dim: output dimension.
            fused_input_dim: Dimension of fused inputs. None by default.
        """
        super().__init__()

        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.fused_input_dim = fused_input_dim

        if fused_input_dim is not None:
            self.fuser = AdditionFuser(fused_input_dim, embedding_dim)

        self.head = nn.Linear(embedding_dim, output_dim)

    @abstractmethod
    def mil_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Pool a bag of instances into a single representation.

        Args:
        ----
            x: Bag of instances.

        Returns:
        -------
        The pooled representation of the bag of instances.
        """
        ...

    def forward(self, batch: GraphBagBatch) -> torch.Tensor:
        """Run a forward pass of the MIL head.

        Optionally fuses an input to the instance representations before pooling.

        Args:
        ----
            batch: Input graph bag batch.

        Returns:
        -------
        MIL-pooled graph bag batch representation.
        """
        batch = copy(batch)
        x = self.backbone(batch)

        if self.fused_input_dim is not None:
            x = self.fuser(x, batch.fuse)

        return torch.stack([self.mil_pooling(bag) for bag in batch.from_tensor(x)])


class AttentionMILModel(MILModel):
    """Implements attention-based deep MIL (https://arxiv.org/abs/1802.04712)."""

    def __init__(self, *args, mil_attention_dim: int = 128, **kwargs) -> None:
        """Args:
        ----
            args: See `MILModel`.
            mil_attention_dim: Dimension of the attention weights for MIL pooling.
            kwargs: See `MILModel`.
        """
        super().__init__(*args, **kwargs)
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, mil_attention_dim),
            nn.Tanh(),
            nn.Linear(mil_attention_dim, 1),
        )

    def mil_pooling(self, x: torch.Tensor) -> torch.Tensor:
        """Perform attention-based MIL pooling on a bag of instance representations.

        Args:
        ----
            x: Bag of instance representations.

        Returns:
        -------
        MIL-pooled graph bag representation.
        """
        a = self.attention(x)
        a = torch.softmax(a, dim=0)
        x = torch.sum(x * a, dim=0)
        x = self.head(x)
        return x
