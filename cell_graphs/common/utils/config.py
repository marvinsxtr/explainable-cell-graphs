import contextlib
import logging
from collections.abc import Callable
from functools import partial

from hydra.core.global_hydra import GlobalHydra
from hydra_zen import MISSING, builds, instantiate, store, to_yaml, zen
from hydra_zen.wrapper import Zen
from omegaconf import DictConfig, OmegaConf

import wandb
from cell_graphs.common.logging.logger import get_hydra_output_dir, logger
from cell_graphs.common.logging.wandb import WandBRun
from cell_graphs.common.utils.constants import ConfigKeys
from cell_graphs.common.utils.helpers import seed_everything


def pre_call(root_config: DictConfig, log_debug: bool = False) -> None:
    """Logs the config, sets the seed and initializes a WandB run before config instantiation.

    Args:
    ----
        root_config: Unresolved config.
        log_debug: Whether to log the config, seed and output path.
    """
    if log_debug:
        logger.setLevel(logging.DEBUG)

    config: DictConfig | None = root_config.get(ConfigKeys.CONFIG)

    if (seed := config.get(ConfigKeys.SEED)) is not None:
        seed_everything(seed)
        logger.debug(f"Set seed to {seed}.")
    else:
        logger.warn("No seed was configured! Run may not be reproducible.")

    if config is None:
        raise KeyError(f"Config must contain {ConfigKeys.CONFIG} at root-level.")
    else:
        logger.debug(f"Running config:\n{to_yaml(config)}")

    output_path = get_hydra_output_dir()
    logger.debug(f"Saving outputs in {output_path}")

    logger.setLevel(logging.INFO)

    if (wandb_config := config.get(ConfigKeys.WANDB)) is not None:
        wandb_run: WandBRun = instantiate(wandb_config)
        wandb_run.run.config.update(OmegaConf.to_container(root_config))
        wandb.save(output_path / ".hydra/*", base_path=output_path, policy="now")


def wrap(main_function: Callable, log_debug: bool = False) -> Zen:
    """Prepares the hydra environment and root config store of a main function.

    Args:
    ----
        main_function: Main function to wrap with hydra-zen.
        log_debug: Whether to log the config, seed and output path before instantiation.

    Returns:
    -------
    The zen-wrapped main function.
    """
    main_function_name = main_function.__name__

    GlobalHydra.instance().clear()
    with contextlib.suppress(KeyError):
        store.delete_entry(group=None, name=main_function_name)

    def _hp_resolver(ref: str) -> str:
        """Resolves references to hyperparameters.

        Args:
        ----
            ref: Reference to resolve. Must be a dot-separated string.
        """
        train_config = "train_config." if main_function_name in ("cv", "nested_cv") else ""
        return f"${{{ConfigKeys.CONFIG}.{train_config}hp.{ref}}}"

    OmegaConf.register_new_resolver("hp", _hp_resolver, replace=True)

    default = {ConfigKeys.CONFIG: MISSING}
    store(
        builds(dict, **default),
        name=main_function.__name__,
        hydra_defaults=["_self_", default],
    )
    store.add_to_hydra_store(overwrite_ok=True)

    return zen(
        main_function, pre_call=partial(pre_call, log_debug=log_debug), resolve_pre_call=False
    )


def run(main_function: Callable) -> None:
    """Configure and run a given function using hydra-zen.

    Args:
    ----
        main_function: Function to configure and run.
    """
    zen = wrap(main_function, log_debug=True)
    zen.hydra_main(config_name=zen.func.__name__, config_path=None, version_base=None)
