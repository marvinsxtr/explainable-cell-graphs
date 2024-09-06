from dataclasses import dataclass

from hydra_zen import builds

from cell_graphs.common.logging.wandb import WandBRun


@dataclass
class BaseConfig:
    """Configures a basic config."""

    seed: int
    wandb: WandBRun | None


DefaultBaseConfig = builds(BaseConfig, seed=42, wandb=None)
