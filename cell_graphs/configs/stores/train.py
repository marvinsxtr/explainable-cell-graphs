from hydra_zen import store

from cell_graphs.configs.data.datasets import ExternalDatasetConfig
from cell_graphs.configs.training.train_config import (
    BaseTrainConfig,
    RegressionHPConfig,
)
from cell_graphs.configs.training.trainer import (
    RegressionTrainerConfig,
)

train_config_store = store(
    group="config",
    hydra_defaults=[
        "_self_",
        {"wandb": None},
        {"dataset": "external_luad"},
        {"trainer": "regression"},
        {"hp": "regression"},
    ],
)
train_config_store(BaseTrainConfig, name="base")

dataset_store = store(group="config/dataset")
dataset_store(ExternalDatasetConfig, name="external_luad")

trainer_store = store(group="config/trainer")
trainer_store(RegressionTrainerConfig, name="regression")

hp_store = store(group="config/hp")
hp_store(RegressionHPConfig, name="regression")
