import torch
from torch import nn


class BCEWithLogitsLoss(nn.Module):
    """Binary cross-entropy loss with logits and optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0, **kwargs) -> None:
        """Args:
        ----
            label_smoothing: Label smoothing to apply to the ground truth labels. Can be in [0, 1].
            kwargs: See `Bnn.CEWithLogitsLoss`.

        Raises
        ------
        ValueError if `label_smoothing` is not inside the interval [0, 1].
        """
        super().__init__()

        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("Label smoothing must be in [0, 1].")

        self.label_smoothing = label_smoothing
        self.bce_with_logits = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the binary cross-entropy loss with label smoothing.

        Args:
        ----
            x: Prediction.
            y: Ground truth.

        Returns:
        -------
        The binary cross-entropy loss.
        """
        if self.label_smoothing > 0:
            pos_smoothed_labels = 1.0 - self.label_smoothing
            neg_smoothed_labels = self.label_smoothing
            y = y * pos_smoothed_labels + (1 - y) * neg_smoothed_labels

        return self.bce_with_logits(x, y)
