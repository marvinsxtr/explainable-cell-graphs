import torch
from torch import nn


class AdditionFuser(nn.Module):
    """Fuses an input feature to the embeddings via addition."""

    def __init__(self, fused_input_dim: int, embedding_dim: int) -> None:
        """Args:
        ----
            fused_input_dim: Dimension of the inputs to fuse.
            embedding_dim: Dimension of the embedding to fuse the input to.
        """
        super().__init__()
        self.fuse_lin = nn.Linear(fused_input_dim, embedding_dim)

    def forward(self, batch: torch.Tensor, fused_inputs: torch.Tensor) -> torch.Tensor:
        """Pass the fused inputs through a linear layer and add them to the batch.

        Args:
        ----
            batch: Batch to fuse the inputs to.
            fused_inputs: Inputs to fuse to the batch.

        Returns:
        -------
        The fused batch representation.
        """
        fused_inputs = fused_inputs.float()
        batch = batch + self.fuse_lin(fused_inputs)
        return batch
