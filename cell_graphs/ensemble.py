from cell_graphs.common.utils.config import run
from cell_graphs.configs.ensemble.ensemble_config import EnsembleConfig


def ensemble(config: EnsembleConfig) -> None:
    """Run an ensembling with a given config.

    Args:
    ----
        config: The ensemble config to run.
    """
    config.ensemble.create()


if __name__ == "__main__":
    import cell_graphs.configs.stores.ensemble
    import cell_graphs.configs.stores.shared  # noqa: F401

    run(ensemble)
