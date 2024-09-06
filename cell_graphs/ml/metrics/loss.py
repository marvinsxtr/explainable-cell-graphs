import torch
from torchmetrics import Metric


class AverageLossMetric(Metric):
    """Metric for tracking a weighted average loss over batches."""

    def __init__(self, **kwargs) -> None:
        """Args:
        ----
            kwargs: See `Metric`.
        """
        super().__init__(**kwargs)
        self.loss: torch.Tensor
        self.add_state(
            "loss",
            default=torch.tensor(0, device=self.device),
            dist_reduce_fx="sum",
            persistent=True,
        )

        self.n: torch.Tensor
        self.add_state(
            "n",
            default=torch.tensor(0, device=self.device),
            dist_reduce_fx="sum",
            persistent=True,
        )

    def update(self, loss: torch.Tensor, n: int) -> None:
        """Update the average loss for a given number of samples in a batch.

        Args:
        ----
            loss: Loss value.
            n: Number of samples to weigh the loss by.
        """
        self.loss = self.loss + (loss * n)
        self.n = self.n + n

    def compute(self) -> torch.Tensor:
        """Compute the weighted average loss over all samples.

        Returns
        -------
        Weighted average loss.
        """
        return self.loss / self.n
