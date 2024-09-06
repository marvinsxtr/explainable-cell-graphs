from cell_graphs.common.utils.config import run
from cell_graphs.configs.training.train_config import TrainConfig


def train(config: TrainConfig) -> None:
    """Run a training job from a config with a given validation and test fold.

    Args:
    ----
        config: Train config to run.
    """
    if config.val_fold == config.test_fold:
        if config.wandb is not None:
            config.wandb.run.finish()
        return

    train_ds, val_ds, test_ds = config.splitter.split_dataset(
        config.dataset, val_fold=config.val_fold, test_fold=config.test_fold
    )

    train_dl = config.train_dl(train_ds) if train_ds is not None else None
    val_dl = config.eval_dl(val_ds) if val_ds is not None else None
    test_dl = config.eval_dl(test_ds) if test_ds is not None else None

    metrics = config.trainer.run_train(train_dl, val_dl, test_dl)

    if config.wandb is not None:
        config.wandb.run.finish()

    return metrics


if __name__ == "__main__":
    import cell_graphs.configs.stores.shared
    import cell_graphs.configs.stores.train  # noqa: F401

    run(train)
