from hydra_zen import builds
from torch.utils.data import DataLoader

from cell_graphs.data.structures.collator import GraphCollator

GraphCollatorConfig = builds(GraphCollator)

TrainDataLoaderConfig = builds(
    DataLoader,
    batch_size="${hp: batch_size}",
    shuffle=True,
    drop_last=False,
    collate_fn=GraphCollatorConfig,
    zen_partial=True,
)

EvalDataLoaderConfig = builds(
    DataLoader,
    batch_size="${hp: batch_size}",
    shuffle=False,
    drop_last=False,
    collate_fn=GraphCollatorConfig,
    zen_partial=True,
)
