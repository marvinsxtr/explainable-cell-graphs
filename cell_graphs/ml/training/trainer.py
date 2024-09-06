from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from hydra_zen.typing import Partial
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torchmetrics import MetricCollection

import wandb
from cell_graphs.common.logging.logger import get_hydra_output_dir, log, log_df
from cell_graphs.common.utils.constants import Columns, Splits
from cell_graphs.common.utils.helpers import get_device
from cell_graphs.data.transforms.sparse import ToSparse

type Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Trainer:
    """Trainer keeping track of gradient updates, metrics, predictions and checkpoints."""

    MODULE = "module"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"
    METRICS = "metrics"

    def __init__(
        self,
        module: nn.Module,
        loss: Loss,
        optimizer: Partial[Optimizer],
        lr_scheduler: Partial[LRScheduler],
        metrics: MetricCollection,
        num_epochs: int,
    ) -> None:
        """Args:
        ----
           module: Torch module to train.
           loss: Loss function.
           optimizer: Optimizer.
           lr_scheduler: Learning rate scheduler.
           metrics: Collection of metrics to keep track of for each dataset split.
           num_epochs: Number of epochs to train for.
        """
        self.module = module
        self.loss = loss
        self.optimizer = optimizer(module.parameters())
        self.lr_scheduler = lr_scheduler(self.optimizer, num_epochs)
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.sparse_transform = ToSparse()
        self.device = get_device()
        self.checkpoint_path = get_hydra_output_dir() / "checkpoint.pt"

        self.module.to(self.device)
        self.metrics.to(self.device)

        self.metrics_dict = {
            Splits.TRAIN: self.metrics.clone(prefix=f"{Splits.TRAIN}_"),
            Splits.VAL: self.metrics.clone(prefix=f"{Splits.VAL}_"),
            Splits.TEST: self.metrics.clone(prefix=f"{Splits.TEST}_"),
        }

    def train_loop(self, train_dl: DataLoader) -> dict[str, torch.Tensor]:
        """Runs a single training loop over the whole training dataset.

        Args:
        ----
            train_dl: Dataloader containing the training data.

        Returns:
        -------
        A dictionary of computed training metrics.
        """
        self.module.train()

        for batch in train_dl:
            x, y = batch
            pred, target, loss = self.forward(x, y)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.metrics_dict[Splits.TRAIN].update(
                preds=pred, target=target, loss=loss, n=x.num_graphs
            )

        self.lr_scheduler.step()

        return self.compute_metrics(Splits.TRAIN)

    def eval_loop(self, eval_dl: DataLoader, split: str) -> dict[str, torch.Tensor]:
        """Runs a single evaluation loop over the whole validation or test dataset.

        Saves the model predictions for later ensembling.

        Args:
        ----
            eval_dl: `DataLoader` containing the validation or test data.
            split: Whether to run a `val` or `test` loop.

        Returns:
        -------
        A dictionary of computed evaluation metrics.
        """
        self.module.eval()

        with torch.no_grad():
            predictions = []
            for batch in eval_dl:
                x, y = batch
                pred, target, loss = self.forward(x, y)

                self.metrics_dict[split].update(
                    preds=pred, target=target, loss=loss, n=x.num_graphs
                )
                predictions.append((x.case_uuid, pred))
            self.save_predictions(predictions)

        return self.compute_metrics(split)

    def forward(
        self, x: Batch, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass of the trained model.

        Args:
        ----
            x: Input batch.
            y: Ground truth targets.

        Returns:
        -------
        A tuple containing the model prediction, target for metric computation and loss.
        """
        x, y = x.to(self.device), y.to(self.device)

        pred = self.module(x)
        loss = self.loss(pred, y)

        return pred, y, loss

    def run_train(
        self, train_dl: DataLoader | None, val_dl: DataLoader | None, test_dl: DataLoader | None
    ) -> dict[str, torch.Tensor | None]:
        """Run a full training job over the given number of epochs.

        Optionally saves a model checkpoint to WandB after training.

        Args:
        ----
            train_dl: `DataLoader` containing the training samples.
            val_dl: `DataLoader` containing validation samples.
            test_dl: `DataLoader` containing test samples.

        Returns:
        -------
        A dictionary of final train, validation and test metrics.

        Raises:
        ------
        RunTimeError if both validation and test `DataLoader`s are passed.
        """
        if val_dl is not None and test_dl is not None:
            raise RuntimeError("Can only run either validation or testing.")

        for _ in range(self.num_epochs):
            metrics: dict[str, torch.Tensor | None] = {}

            if train_dl is not None:
                metrics.update(self.train_loop(train_dl))

            if val_dl is not None:
                metrics.update(self.eval_loop(val_dl, Splits.VAL))

            if test_dl is not None:
                metrics.update(self.eval_loop(test_dl, Splits.TEST))

            log(metrics)

        if test_dl is not None:
            self.save_checkpoint(self.checkpoint_path)

            if (run := wandb.run) is not None:
                run.log_model(path=self.checkpoint_path, name=self.checkpoint_path.stem)

        return metrics

    def compute_metrics(self, split: str) -> None:
        """Compute a set of metrics for a specific dataset split.

        Args:
        ----
            split: Dataset split to compute metrics for. Can be `train`, `val` or `test`.
        """
        metrics = self.metrics_dict[split].compute()
        self.metrics_dict[split].reset()
        return {k: v.item() for k, v in metrics.items()}

    def save_predictions(self, predictions: list[tuple[np.ndarray, torch.Tensor]]) -> None:
        """Save model predictions to disk and WandB.

        Args:
        ----
            predictions: Model predictions to save to disk.
        """
        case_uuids, preds = zip(*predictions, strict=True)

        case_uuids = np.concatenate(case_uuids).squeeze()
        preds: np.ndarray = torch.concatenate(preds).detach().cpu().numpy()

        indexes = np.unique(case_uuids, return_index=True)[1]
        case_uuids = [case_uuids[index] for index in sorted(indexes)]

        pred_df = pd.DataFrame(
            {
                Columns.CASE_UUID: case_uuids,
                **{f"{Columns.PRED}_{i}": preds[:, i] for i in range(preds.shape[1])},
            }
        )
        log_df(pred_df, "predictions", log=False)

    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Get the full state of the trainer as a dictionary.

        Args:
        ----
            args: See `Module.state_dict`.
            kwargs: See `Module.state_dict`.

        Returns:
        -------
        A dictionary containing the full state of the trainer, including optimizer and scheduler.
        """
        return {
            self.MODULE: self.module.state_dict(*args, **kwargs),
            self.OPTIMIZER: self.optimizer.state_dict(),
            self.LR_SCHEDULER: self.lr_scheduler.state_dict(),
            self.METRICS: self.metrics.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any], **kwargs) -> None:
        """Load a given trainer state dictionary.

        Args:
        ----
            state_dict: The trainer state dictionary to load.
            kwargs: See `Module.load_state_dict`.
        """
        self.module.load_state_dict(state_dict[self.MODULE], **kwargs)
        self.optimizer.load_state_dict(state_dict[self.OPTIMIZER])
        self.lr_scheduler.load_state_dict(state_dict[self.LR_SCHEDULER])
        self.metrics.load_state_dict(state_dict[self.METRICS])

    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save a model checkpoint to a local path.

        Args:
        ----
            checkpoint_path: Path to save the checkpoint to.
        """
        torch.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a model checkpoint from a local file.

        Args:
        ----
            checkpoint_path: Path to the model checkpoint to load.
        """
        self.load_state_dict(torch.load(checkpoint_path))
