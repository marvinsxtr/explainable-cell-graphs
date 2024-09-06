from typing import Any

import pandas as pd
from hydra_zen import multirun

from cell_graphs.common.logging.logger import log_df
from cell_graphs.common.utils.columns import ResultsColumns
from cell_graphs.common.utils.constants import ConfigKeys, ConfigPaths
from cell_graphs.configs.training.train_config import TrainConfig
from cell_graphs.ml.cv.cv import CV


class NestedCV(CV):
    """Perform nested k-fold cross-validation."""

    def __init__(
        self,
        *args,
        overrides: list[tuple[str, Any]],
        **kwargs,
    ) -> None:
        """Args:
        ----
            args: See `CV`.
            overrides: Overrides for hyperparameter grid-search.
            kwargs: See `CV`.
        """
        super().__init__(*args, **kwargs)
        self.overrides = dict(overrides)

    def run(self, train_config: TrainConfig) -> None:
        """Run nested cross-validation for a given training configuration.

        Args:
        ----
            train_config: Training configuration to run for each fold.
        """
        overrides = {
            ConfigPaths.TEST_FOLD: multirun([0, 1, 2, 3, 4]),
            ConfigPaths.VAL_FOLD: multirun([0, 1, 2, 3, 4]),
        }
        overrides.update(self.overrides)

        inner_results_df = self.run_train_jobs(train_config, overrides)
        log_df(inner_results_df, "val_results")

        for test_fold in range(self.num_folds):
            overrides = self.get_best_overrides(inner_results_df, test_fold)

            outer_results_df = self.run_train_jobs(train_config, overrides)
            log_df(outer_results_df, f"test_results_{test_fold}")

    def get_best_overrides(self, results_df: pd.DataFrame, test_fold: int) -> dict:
        """Get the best hyperparameters from the inner folds by max. average validation metric.

        Args:
        ----
            results_df: `DataFrame` containing the overrides and corresponding validation metrics.
            test_fold: Test fold to find the best hyperparameters for.

        Returns:
        -------
        Best hyperparameters for the given test fold by max. average validation metric.
        """
        fold_results = results_df[results_df[ConfigPaths.TEST_FOLD] == test_fold]
        fold_results = fold_results.drop(columns=[ConfigPaths.VAL_FOLD, ConfigPaths.TEST_FOLD])

        agg_dict = {
            col: "mean" if col == self.selection_metric else "first"
            for col in fold_results.columns
        }

        avg_val_results = fold_results.groupby([ResultsColumns.OVERRIDES], as_index=False).agg(
            agg_dict
        )

        best_row: pd.Series = avg_val_results.iloc[avg_val_results[self.selection_metric].idxmax()]
        non_config_labels = [l for l in best_row.keys() if ConfigKeys.CONFIG not in l]  # noqa: SIM118

        best_overrides = best_row.drop(labels=non_config_labels).to_dict()
        best_overrides.update(
            {
                ConfigPaths.TEST_FOLD: test_fold,
                ConfigPaths.VAL_FOLD: None,
                ConfigPaths.SEED: multirun(self.seeds),
            }
        )
        return best_overrides
