from hydra_zen import store

from cell_graphs.configs.logging.wandb import BaseWandBConfig

config_store = store(group="config", hydra_defaults=["_self_", {"wandb": None}])

wandb_config_store = store(group="config/wandb")
wandb_config_store(BaseWandBConfig, name="base")
