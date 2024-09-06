from hydra_zen import store

from cell_graphs.configs.cv.cv_config import BaseCVMainConfig
from cell_graphs.configs.cv.nested_cv_config import BaseNestedCVMainConfig
from cell_graphs.configs.data.datasets import ExternalDatasetConfig
from cell_graphs.configs.training.train_config import RegressionHPConfig
from cell_graphs.configs.training.trainer import (
    RegressionTrainerConfig,
)

cv_config_store = store(
    group="config",
    hydra_defaults=[
        "_self_",
        {"train_config/wandb": None},
        {"train_config/dataset": "external_luad"},
        {"train_config/trainer": "regression"},
        {"train_config/hp": "regression"},
    ],
)
cv_config_store(BaseNestedCVMainConfig, name="nested_cv")
cv_config_store(BaseCVMainConfig, name="cv")

dataset_store = store(group="config/train_config/dataset")
dataset_store(ExternalDatasetConfig, name="external_luad")

trainer_store = store(group="config/train_config/trainer")
trainer_store(RegressionTrainerConfig, name="regression")

hp_store = store(group="config/train_config/hp")
hp_store(RegressionHPConfig, name="regression")
