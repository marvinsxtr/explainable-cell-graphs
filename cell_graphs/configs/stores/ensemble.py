from hydra_zen import store

from cell_graphs.configs.ensemble.ensemble_config import BaseEnsembleConfig

train_config_store = store(group="config")
train_config_store(BaseEnsembleConfig, name="base")
