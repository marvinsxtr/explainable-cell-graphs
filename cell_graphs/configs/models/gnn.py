from hydra_zen import builds

from cell_graphs.ml.models.backbones import SurvivalGNN

SurvivalGNNConfig = builds(
    SurvivalGNN,
    input_dim="${hp: input_dim}",
    embedding_dim=64,
    hidden_dim=64,
    num_layers=3,
    conv="GINConv",
    pooling="TopKPooling",
    dropout=0.1,
)
