from dataclasses import dataclass

from hydra_zen import builds, multirun

from cell_graphs.common.utils.constants import ConfigPaths
from cell_graphs.configs.base import BaseConfig
from cell_graphs.configs.training.train_config import BaseTrainConfig, TrainConfig
from cell_graphs.ml.cv.nested_cv import NestedCV


@dataclass
class NestedCVMainConfig(BaseConfig):
    """Configures a nested cross-validation run."""

    nested_cv: NestedCV
    train_config: TrainConfig


NestedCVConfig = builds(
    NestedCV,
    log_to_wandb=True,
    overrides=[
        (ConfigPaths.LR, builds(multirun, [5e-5, 1e-5, 5e-6])),
    ],
    seeds=[42, 225, 319, 89, 445],
)

BaseNestedCVMainConfig = builds(
    NestedCVMainConfig,
    seed=42,
    wandb=None,
    train_config=BaseTrainConfig,
    nested_cv=NestedCVConfig,
)
