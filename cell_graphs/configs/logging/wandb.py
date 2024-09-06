from hydra_zen import builds

from cell_graphs.common.logging.wandb import WandBRun

BaseWandBConfig = builds(WandBRun, group=None, mode="online")
