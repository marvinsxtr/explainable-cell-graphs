from hydra_zen import builds
from torch_geometric.transforms import KNNGraph

from cell_graphs.common.utils.columns import MetaColumns
from cell_graphs.common.utils.constants import Indications
from cell_graphs.data.datasets.external_dataset import ExternalLungDataset

KNN3PreTransformConfig = builds(KNNGraph, k=3, force_undirected=True)

ExternalDatasetConfig = builds(
    ExternalLungDataset,
    root="./data/external",
    indication=Indications.AC,
    target_columns="${hp: target_columns}",
    fuse_columns=[MetaColumns.STAGE_3_PLUS],
    pre_transform=KNN3PreTransformConfig,
)
