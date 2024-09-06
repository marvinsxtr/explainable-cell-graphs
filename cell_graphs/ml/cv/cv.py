from ast import literal_eval
from collections.abc import Sequence

import pandas as pd
from hydra.core.utils import JobReturn, JobStatus
from hydra_zen import launch, multirun

from cell_graphs.common.logging.logger import log_df
from cell_graphs.common.utils.columns import ResultsColumns
from cell_graphs.common.utils.config import wrap
from cell_graphs.common.utils.constants import ConfigKeys, ConfigPaths
from cell_graphs.configs.training.train_config import TrainConfig
from cell_graphs.train import train
from wandb.util import generate_id


class CV:
    """Perform k-fold cross-validation with multiple seeds."""

    def __init__(
        self,
        seeds: list[int],
        selection_metric: str = "val_c_index",
        final_metric: str = "test_c_index",
        num_folds: int = 5,
        log_to_wandb: bool = True,
    ) -> None:
        """Args:
        ----
            seeds: Seeds to repeat the k-fold cross-validation on.
            selection_metric: Metric used to select the best hyperparameters in the inner folds.
              Note: Only needed for nested cross-validation, see `NestedCV`.
            final_metric: Metric to report in the outer test folds.
            num_folds: Number of folds (k).
            log_to_wandb: Whether to enable WandB logging for the individual folds.
        """
        self.seeds = seeds
        self.selection_metric = selection_metric
        self.final_metric = final_metric
        self.num_folds = num_folds
        self.log_to_wandb = log_to_wandb
        self.group_id = generate_id()

    def run(self, train_config: TrainConfig) -> None:
        """Run cross-validation for a given training configuration.

        Args:
        ----
            train_config: Training configuration to run for each fold.
        """
        overrides = {
            ConfigPaths.TEST_FOLD: multirun([0, 1, 2, 3, 4]),
            ConfigPaths.VAL_FOLD: None,
            ConfigPaths.SEED: multirun(self.seeds),
        }

        outer_results_df = self.run_train_jobs(train_config, overrides)
        log_df(outer_results_df, "test_results")

    def run_train_jobs(self, train_config: TrainConfig, overrides: dict) -> pd.DataFrame:
        """Run training jobs for a given config and overrides.

        Args:
        ----
            train_config: Training configuration to run.
            overrides: Config overrides to run in a grid.

        Returns:
        -------
        A `DataFrame` containing fold indices, overrides, resulting metrics.
        """
        if self.log_to_wandb:
            overrides["+config/wandb"] = "base"
            overrides[ConfigPaths.WANDB_GROUP] = (
                f"{self.group_id}_outer_folds"
                if overrides[ConfigPaths.VAL_FOLD] is None
                else f"{self.group_id}_inner_folds"
            )

        job_returns: list[JobReturn] = launch(
            {ConfigKeys.CONFIG: train_config},
            wrap(train),
            overrides=overrides,
            multirun=True,
            version_base=None,
        )[0]

        return pd.DataFrame(
            [
                self.validate_job_return(job_return)
                for job_return in job_returns
                if job_return.return_value is not None
            ]
        )

    def parse_overrides(self, overrides: Sequence[str]) -> tuple[dict, str]:
        """Parse a list of overrides into a dictionary and string without fold overrides.

        Args:
        ----
            overrides: List of overrides as strings.

        Returns:
        -------
        A tuple of the overrides as a dictionary and string without fold overrides.
        """
        parsed_overrides = {}
        for override in overrides:
            key, value = override.split("=")

            try:
                parsed = literal_eval(value)
            except Exception:
                parsed = None if value == "null" else value

            parsed_overrides[key] = parsed

        filtered_overrides = list(
            filter(
                lambda o: ConfigPaths.VAL_FOLD not in o and ConfigPaths.TEST_FOLD not in o,
                overrides,
            )
        )
        override_str = " ".join(filtered_overrides)

        return parsed_overrides, override_str

    def validate_job_return(self, job_return: JobReturn) -> dict:
        """Validates and parses the results of a training job.

        Args:
        ----
            job_return: Training job result.

        Returns:
        -------
        Dictionary of parsed and validated training job results.

        Raises:
        ------
        TypeError if the result is not a dict or overrides are empty.
        RunTimeError if the job did not finish successfully.
        """
        if job_return.status != JobStatus.COMPLETED:
            raise RuntimeError("Job did not finish successfully.")

        ret = job_return.return_value

        if not isinstance(ret, dict):
            raise TypeError("Job must return a dictionary.")

        if job_return.overrides is None:
            raise ValueError("Overrides are empty.")

        overrides, overrides_str = self.parse_overrides(job_return.overrides)

        result = {}
        result.update(overrides)
        result[ResultsColumns.OVERRIDES] = overrides_str

        if self.selection_metric in ret:
            result[self.selection_metric] = ret[self.selection_metric]

        if self.final_metric in ret:
            result[self.final_metric] = ret[self.final_metric]

        return result
