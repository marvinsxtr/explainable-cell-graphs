from hydra_zen import builds

from cell_graphs.ml.data.splitter import StratifiedSplitter

StratifiedSplitterConfig = builds(StratifiedSplitter, n_splits=5, random_state=42, shuffle=True)
