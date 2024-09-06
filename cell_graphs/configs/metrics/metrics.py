from hydra_zen import builds
from torchmetrics import MetricCollection

from cell_graphs.common.utils.constants import MetricNames
from cell_graphs.ml.metrics.survival import ConcordanceIndex, CoxPHLossMetric

SurvivalMetricsConfig = builds(
    MetricCollection,  # type:ignore[arg-type]
    metrics=builds(  # type: ignore[call-overload]
        dict,
        **{
            MetricNames.LOSS: builds(CoxPHLossMetric),
            MetricNames.C_INDEX: builds(ConcordanceIndex),
        },
    ),
)
