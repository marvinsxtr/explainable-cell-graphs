from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch

import wandb
from cell_graphs.common.utils.config import wrap
from cell_graphs.common.utils.constants import ConfigKeys
from cell_graphs.common.utils.helpers import get_device
from cell_graphs.configs.training.train_config import TrainConfig
from wandb.apis.public.runs import Run, Runs


def load_df(wandb_run: Run, file_name: str) -> pd.DataFrame:
    """Load a `DataFrame` from a WandB run.

    Args:
    ----
        wandb_run: WandB run to load the `DataFrame` from.
        file_name: File name of the `DataFrame` to load.
    """
    config = wandb_run.file(file_name)

    with TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir)
        config.download(tmp_file)
        df = pd.read_parquet(tmp_file)

    return df


def get_wandb_runs(wandb_group: str) -> Runs:
    """Retrieve all WandB runs contained in a WandB run group.

    Args:
    ----
        wandb_group: WandB group to retrieve runs from.

    Returns:
    -------
    A list of WandB runs associated with the given WandB group.
    """
    if (wandb_run := wandb.run) is None:
        raise RuntimeError("No WandB run found.")

    return wandb.Api().runs(
        path=f"{wandb_run.entity}/{wandb_run.project}", filters={"group": wandb_group}
    )


def load_wandb_artifact(
    wandb_run: Run,
    model_name: str,
) -> dict:
    """Load an artifact from a WandB run.

    Automatically moves the model to GPU if CUDA is available.

    Args:
    ----
        wandb_run: WandB run to load the artifact from.
        model_name: Name of the artifact to load without version number.

    Returns:
    -------
    The state dict of the loaded artifact.
    """
    artifacts = [a for a in wandb_run.logged_artifacts() if model_name in a.name]

    if len(artifacts) == 1:
        artifact = artifacts[0]
    else:
        raise ValueError("Did not find exactly one matching artifact.")

    versioned_model_name = artifact.name
    device = get_device()

    wandb_run = wandb.run
    if wandb_run is None or wandb_run.offline:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            artifact.download(tmp_path)
            downloaded_model_path = tmp_path / (model_name + ".pt")
            return torch.load(downloaded_model_path, map_location=device)
    else:
        downloaded_model_path = wandb_run.use_model(
            f"{wandb_run.entity}/{wandb_run.project}/{versioned_model_name}"
        )
        return torch.load(downloaded_model_path, map_location=device)


def reload_config(wandb_run: Run) -> TrainConfig:
    """Reload a config from a WandB run and instantiate using hydra-zen.

    NOTE: Reloading the config only works if there were no changes to the config structure.

    Args:
    ----
        wandb_run: WandB run to reload the config from.

    Returns:
    -------
    An instantiated training config.
    """
    run_config = wandb_run.config.copy()
    run_config[ConfigKeys.CONFIG][ConfigKeys.WANDB] = None
    return wrap(lambda config: config)(run_config)
