from hydra_zen import MISSING, builds
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from cell_graphs.configs.metrics.metrics import SurvivalMetricsConfig
from cell_graphs.configs.models.mil import SurvivalGNNMILModelConfig
from cell_graphs.ml.loss.bce import BCEWithLogitsLoss
from cell_graphs.ml.loss.coxph import CoxPHLoss
from cell_graphs.ml.training.trainer import Trainer

CosineSchedulerConfig = builds(CosineAnnealingLR, zen_partial=True)

CoxPHLossConfig = builds(CoxPHLoss)
BCEWithLogitsLossConfig = builds(BCEWithLogitsLoss, label_smoothing=0.0)

AdamWConfig = builds(AdamW, lr="${hp: lr}", zen_partial=True)

BaseTrainerConfig = builds(
    Trainer,
    module=MISSING,
    loss=MISSING,
    metrics=MISSING,
    optimizer=AdamWConfig,
    lr_scheduler=CosineSchedulerConfig,
    num_epochs="${hp: num_epochs}",
)

RegressionTrainerConfig = builds(
    Trainer,
    module=SurvivalGNNMILModelConfig,
    loss=CoxPHLossConfig,
    metrics=SurvivalMetricsConfig,
    builds_bases=(BaseTrainerConfig,),
)
