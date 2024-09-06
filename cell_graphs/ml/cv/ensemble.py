from typing import Literal

import pandas as pd
import torch
from torchsurv.metrics.cindex import ConcordanceIndex

from cell_graphs.common.logging.logger import log
from cell_graphs.common.utils.columns import MetaColumns
from cell_graphs.common.utils.constants import Columns, ConfigKeys, MetricNames
from cell_graphs.common.utils.wandb import get_wandb_runs, load_df, reload_config
from wandb.apis.public.runs import Run, Runs


class Ensemble:
    """Ensemble multiple survival regression models.

    NOTE: This class expects a same-commit outer k-fold cross-validation run saved on WandB.
    """

    def __init__(
        self,
        wandb_group: str,
        num_folds: int = 5,
        aggregation: Literal["mean", "median"] = "mean",
        predictions_file_name: str = "predictions.parquet",
        stage_column: Literal["STAGE_3+"] = "STAGE_3+",
    ) -> None:
        """Args:
        ----
           wandb_group: WandB group of the cross-validation runs to load predictions/configs from.
           num_folds: Number of test folds.
           aggregation: Aggregation used for risk prediction ensembling. Defaults to mean.
           predictions_file_name: Prediction `DataFrame` file name.
           stage_column: Metadata column containing the cancer stage.
        """
        self.wandb_group = wandb_group
        self.num_folds = num_folds
        self.aggregation = aggregation
        self.predictions_file_name = predictions_file_name
        self.stage_column = stage_column

    def create(self) -> None:
        """Create a risk ensemble from the given WandB group and log metrics.

        Logs the average c-index and standard deviation over test folds of
        * the cancer stage as risk score baseline
        * the GNN-based survival regression model averaged over seeds
        * the ensemble of GNN-based survival regression models
        """
        runs = get_wandb_runs(self.wandb_group)

        model_c_index_list, ensemble_c_index_list, stage_c_index_list = [], [], []
        for test_fold in range(self.num_folds):
            test_fold_runs = self.get_test_fold_runs(runs, test_fold)

            targets = self.load_targets(test_fold_runs[0])
            seed_preds = [self.load_preds(seed_run) for seed_run in test_fold_runs]

            seed_c_index = []
            for seed_pred in seed_preds:
                joined_pred_df = seed_pred.join(targets, on=Columns.CASE_UUID, how="left")
                seed_c_index.append(self.get_c_index(joined_pred_df, f"{Columns.PRED}_0"))
            avg_model_c_index = torch.stack(seed_c_index).mean()

            aggregated_preds = pd.concat(seed_preds, axis=1).aggregate([self.aggregation], axis=1)
            joined_df = aggregated_preds.join(targets, on=Columns.CASE_UUID, how="left")

            model_c_index_list.append(avg_model_c_index)
            ensemble_c_index_list.append(self.get_c_index(joined_df, self.aggregation))
            stage_c_index_list.append(self.get_c_index(joined_df, self.stage_column))

        self.log_c_index(model_c_index_list, "model")
        self.log_c_index(ensemble_c_index_list, "ensemble")
        self.log_c_index(stage_c_index_list, "uicc8")

    def get_c_index(self, df: pd.DataFrame, pred_column: str) -> torch.Tensor:
        """Compute the c-index from a `DataFrame` of predictions and ground truth survival.

        Args:
        ----
            df: `DataFrame` containing the risk prediction and ground truth survival.
            pred_column: Column in `df` containing the risk predictions.

        Returns:
        -------
        The c-index of the given survival predictions.
        """
        return ConcordanceIndex()(
            torch.from_numpy(df[pred_column].to_numpy()),
            torch.from_numpy(df[MetaColumns.OS_EVENT].to_numpy()),
            torch.from_numpy(df[MetaColumns.OS_M].to_numpy()),
        )

    def log_c_index(self, c_index_list: list[torch.Tensor], prefix: str) -> None:
        """Log the average and standard deviation of a list of c-indices.

        Args:
        ----
            c_index_list: List of c-indices.
            prefix: Prefix to use for logging of the c-index.
        """
        stacked_c_index = torch.stack(c_index_list)
        metrics = {
            f"avg_{prefix}_{MetricNames.C_INDEX}": stacked_c_index.mean().item(),
            f"std_{prefix}_{MetricNames.C_INDEX}": stacked_c_index.std().item(),
        }

        log(metrics)

    def load_targets(self, run: Run) -> pd.DataFrame:
        """Load the prediction targets and cancer stage from a previous run.

        NOTE: Reloading the config only works if there were no changes to the config structure.

        Args:
        ----
            run: WandB run to reinstantiate the config from.

        Returns:
        -------
        `DataFrame` containing survival regression targets and the cancer stage.
        """
        run_config = reload_config(run)

        return run_config.dataset.patient_df[
            [MetaColumns.CASE_UUID, MetaColumns.OS_M, MetaColumns.OS_EVENT, self.stage_column]
        ].set_index(MetaColumns.CASE_UUID)

    def load_preds(self, run: Run) -> pd.DataFrame:
        """Load risk predictions from a WandB run.

        Args:
        ----
            run: WandB run to load predictions from.

        Returns:
        -------
        `DataFrame` containing risk predictions.
        """
        return load_df(run, self.predictions_file_name).set_index(Columns.CASE_UUID)

    def get_test_fold_runs(self, runs: Runs, test_fold: int) -> list[Run]:
        """Filter runs by a given test fold.

        Args:
        ----
            runs: WandB runs to filter.
            test_fold: Test fold to filter the runs by.

        Returns:
        -------
        A list of WandB runs with the matching test fold.
        """
        return [
            run for run in runs if run.config[ConfigKeys.CONFIG][ConfigKeys.TEST_FOLD] == test_fold
        ]
