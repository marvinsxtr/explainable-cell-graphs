from dataclasses import dataclass
from typing import Any

from hydra_zen import MISSING, builds
from hydra_zen.typing import Partial
from torch_geometric.loader import DataLoader

from cell_graphs.common.utils.columns import MetaColumns
from cell_graphs.configs.base import BaseConfig, DefaultBaseConfig
from cell_graphs.configs.data.data_loaders import EvalDataLoaderConfig, TrainDataLoaderConfig
from cell_graphs.configs.data.splitter import StratifiedSplitterConfig
from cell_graphs.data.datasets.lung_dataset import LungDataset
from cell_graphs.ml.data.splitter import StratifiedSplitter
from cell_graphs.ml.training.trainer import Trainer


@dataclass
class TrainConfig(BaseConfig):
    """Configures a training run."""

    val_fold: int | None
    test_fold: int
    hp: dict[str, Any]
    dataset: LungDataset
    train_dl: Partial[DataLoader]
    eval_dl: Partial[DataLoader]
    splitter: StratifiedSplitter
    trainer: Trainer


BaseHPConfig = builds(dict, input_dim=17, num_epochs=50, lr=5e-5)

RegressionHPConfig = builds(
    dict,
    batch_size=16,
    target_columns=[MetaColumns.OS_M, MetaColumns.OS_EVENT],
    builds_bases=(BaseHPConfig,),
)

BaseTrainConfig = builds(
    TrainConfig,
    val_fold=1,
    test_fold=0,
    train_dl=TrainDataLoaderConfig,
    eval_dl=EvalDataLoaderConfig,
    splitter=StratifiedSplitterConfig,
    hp=BaseHPConfig,
    trainer=MISSING,
    dataset=MISSING,
    builds_bases=(DefaultBaseConfig,),
)
