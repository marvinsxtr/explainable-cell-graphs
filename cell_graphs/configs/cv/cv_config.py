from dataclasses import dataclass

from hydra_zen import builds

from cell_graphs.configs.base import BaseConfig
from cell_graphs.configs.training.train_config import BaseTrainConfig, TrainConfig
from cell_graphs.ml.cv.cv import CV


@dataclass
class CVMainConfig(BaseConfig):
    """Configures a cross-validation run."""

    cv: CV
    train_config: TrainConfig


BaseCVConfig = builds(
    CV,
    seeds=[176, 225, 319, 89, 445],
    log_to_wandb=True,
)

BaseCVMainConfig = builds(
    CVMainConfig,
    seed=42,
    wandb=None,
    train_config=BaseTrainConfig,
    cv=BaseCVConfig,
)
