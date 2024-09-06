import warnings

import torch
from torchsurv.loss.cox import neg_partial_log_likelihood

warnings.filterwarnings(action="ignore", category=UserWarning, module="torchsurv")


class CoxPHLoss(torch.nn.Module):
    """Cox Proportional Hazards (CoxPH) negative partial log-likelihood loss."""

    def forward(self, log_h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the negative partial log-likelihood loss.

        Args:
        ----
            log_h: The log hazard predictions.
            y: Survival time and event indicator.

        Returns:
        -------
        Negative partial log-likelihood loss.
        """
        durations, events = y.T

        events = events.bool()

        return neg_partial_log_likelihood(log_h, events, durations)
