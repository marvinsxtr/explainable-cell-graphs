import logging
from pathlib import Path

import pandas as pd
from hydra.core.hydra_config import HydraConfig

import wandb

logger = logging.getLogger()


def get_hydra_output_dir() -> Path:
    """Return the hydra output directory.

    Returns
    -------
    Path to the hydra output directory.
    """
    return Path(HydraConfig.get().runtime.output_dir)


def log(metrics: dict) -> None:
    """Log a dictionary of metrics to the terminal and WandB.

    The metrics will only be logged to WandB if there is an active run.

    Args:
    ----
        metrics: Dictionary of metrics to log.
    """
    if (wandb_run := wandb.run) is not None:
        wandb_run.log(metrics)

    log_str = ""
    for key, value in metrics.items():
        log_str += f"{key}: {value:<12.4f}"

    logger.info(log_str)


def log_df(df: pd.DataFrame, name: str, log: bool = True) -> None:
    """Save a `DataFrame` locally as .parquet and to WandB and optionally log it to the terminal.

    Args:
    ----
        df: `DataFrame` to save.
        name: File name to use without file extension.
        log: Whether to log the `DataFrame` to the terminal. Defaults to True.
    """
    output_dir = get_hydra_output_dir()
    df_path = Path(output_dir) / f"{name}.parquet"
    df.to_parquet(df_path)

    if (wandb_run := wandb.run) is not None:
        wandb_run.save(str(df_path), base_path=output_dir)

    if log:
        logger.info(f"\n{df}")
