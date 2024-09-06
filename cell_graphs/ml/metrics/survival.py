from abc import ABC, abstractmethod

import torch
from torchmetrics import Metric
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.cindex import ConcordanceIndex as CIndex


class SurvivalMetric(Metric, ABC):
    """Metric for keeping track of the concordance index and CoxPH loss."""

    def __init__(self, **kwargs) -> None:
        """Args:
        ----
            kwargs: See `Metric`.
        """
        super().__init__(**kwargs)
        self.events: torch.Tensor
        self.add_state(
            "events",
            default=torch.tensor([], device=self.device),
            dist_reduce_fx="cat",
            persistent=True,
        )

        self.times: torch.Tensor
        self.add_state(
            "times",
            default=torch.tensor([], device=self.device),
            dist_reduce_fx="cat",
            persistent=True,
        )

        self.preds: torch.Tensor
        self.add_state(
            "preds",
            default=torch.tensor([], device=self.device),
            dist_reduce_fx="cat",
            persistent=True,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the survival metric with predictions and targets of a single batch.

        Args:
        ----
            preds: Predicted risks.
            target: Ground truth survival times and event indicators.
        """
        times, events = torch.split(target, 1, dim=1)
        self.events = torch.cat((self.events, events))
        self.times = torch.cat((self.times, times))
        self.preds = torch.cat((self.preds, preds))

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Compute the survival metric over all batches.

        Returns
        -------
        The computed survival metric.
        """
        ...


class ConcordanceIndex(SurvivalMetric):
    """Concordance index metric."""

    def compute(self) -> torch.Tensor:
        """Compute the concordance index.

        Returns
        -------
        The concordance index.
        """
        c_index = CIndex()
        return c_index(
            self.preds.detach().cpu().squeeze(),
            self.events.detach().cpu().squeeze().bool(),
            self.times.detach().cpu().squeeze(),
        )


class CoxPHLossMetric(SurvivalMetric):
    """CoxPH negative partial log-likelihood loss."""

    def compute(self) -> torch.Tensor:
        """Compute the negative partial log-likelihood loss.

        Returns
        -------
        The negative partial log-likelihood loss.
        """
        return neg_partial_log_likelihood(
            self.preds.detach().cpu().squeeze(),
            self.events.detach().cpu().squeeze().bool(),
            self.times.detach().cpu().squeeze(),
        )
