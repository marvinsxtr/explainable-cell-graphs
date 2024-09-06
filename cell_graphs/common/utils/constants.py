from typing import Final


class ConfigKeys:
    """Keys present in configs."""

    CONFIG: Final[str] = "config"
    SEED: Final[str] = "seed"
    WANDB: Final[str] = "wandb"
    TEST_FOLD: Final[str] = "test_fold"


class ConfigPaths:
    """Collection of paths to specific config keys."""

    WANDB_GROUP: Final[str] = "config.wandb.group"
    VAL_FOLD: Final[str] = "config.val_fold"
    TEST_FOLD: Final[str] = "config.test_fold"
    LR: Final[str] = "config.hp.lr"
    NUM_EPOCHS: Final[str] = "config.hp.num_epochs"
    SEED: Final[str] = "config.seed"


class MetricNames:
    """Metric names."""

    LOSS: Final[str] = "loss"
    C_INDEX: Final[str] = "c_index"
    AUROC: Final[str] = "auroc"


class Indications:
    """Lung cancer indications."""

    AC: Final[str] = "AC"
    SCC: Final[str] = "SCC"


class Splits:
    """Dataset split names."""

    TRAIN: Final[str] = "train"
    VAL: Final[str] = "val"
    TEST: Final[str] = "test"


class Columns:
    """General `DataFrame` column names."""

    CASE_UUID: Final[str] = "case_uuid"
    PRED: Final[str] = "pred"
    RELEVANCE: Final[str] = "relevance"
