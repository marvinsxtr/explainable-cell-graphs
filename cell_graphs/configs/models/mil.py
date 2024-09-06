from hydra_zen import builds

from cell_graphs.configs.models.gnn import SurvivalGNNConfig
from cell_graphs.ml.models.mil import AttentionMILModel

SurvivalGNNMILModelConfig = builds(
    AttentionMILModel,
    backbone=SurvivalGNNConfig,
    embedding_dim=64,
    output_dim=1,
    fused_input_dim=1,
)
