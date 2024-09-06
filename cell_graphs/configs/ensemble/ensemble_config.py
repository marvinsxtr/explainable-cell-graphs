from dataclasses import dataclass

from hydra_zen import MISSING, builds

from cell_graphs.common.logging.wandb import WandBRun
from cell_graphs.configs.base import BaseConfig, DefaultBaseConfig
from cell_graphs.configs.logging.wandb import BaseWandBConfig
from cell_graphs.ml.cv.ensemble import Ensemble


@dataclass
class EnsembleConfig(BaseConfig):
    """Configures an ensemble."""

    ensemble: Ensemble


OfflineWandBConfig = builds(WandBRun, mode="offline", builds_bases=(BaseWandBConfig,))

BaseEnsemble = builds(Ensemble, wandb_group=MISSING)

BaseEnsembleConfig = builds(
    EnsembleConfig,
    ensemble=BaseEnsemble,
    wandb=OfflineWandBConfig,
    builds_bases=(DefaultBaseConfig,),
)
