from enum import Enum

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

from cell_graphs.data.datasets.lung_dataset import LungDataset


class Split(Enum):
    """Data split identifier."""

    NONE = 0
    TRAIN = 1
    VAL = 2
    TEST = 3


class StratifiedSplitter(StratifiedKFold):
    """Perform stratified k-fold dataset splitting compatible with nested cross-validation."""

    def fold_to_split(self, fold_idx: int, val_fold: int | None, test_fold: int) -> Split:
        """Map a fold index to a data split.

        For nested cross-validation, the test split will only be used when the `val_fold` is None.

        Args:
        ----
            fold_idx: Fold index to map to a data split.
            val_fold: Validation fold index. If None, the test split will be assigned.
            test_fold: Test fold index. Will not be assigned to a split unless `val_fold` is None.

        Returns:
        -------
        The split to assign to the given fold index.
        """
        num_splits = self.get_n_splits()

        if fold_idx not in range(num_splits):
            raise ValueError(f"Fold index must be in [0, {num_splits - 1}].")

        all_folds = set(range(num_splits))
        val_folds = {val_fold} if val_fold is not None else set()
        test_folds = {test_fold} if val_fold is None else set()
        train_folds = all_folds - val_folds - {test_fold}

        if fold_idx in train_folds:
            return Split.TRAIN
        elif fold_idx in val_folds:
            return Split.VAL
        elif fold_idx in test_folds:
            return Split.TEST
        else:
            return Split.NONE

    def split_dataset(
        self, dataset: LungDataset, val_fold: int | None, test_fold: int
    ) -> tuple[Subset | None, Subset | None, Subset | None]:
        """Split a dataset into train, val and test using stratified sampling.

        Depending on whether `val_fold` is None, returns a validation or test fold for nested CV.

        Args:
        ----
            dataset: Dataset to split.
            val_fold: Validation fold index. If None, the test split will be assigned.
            test_fold: Test fold index. Will not be assigned to a split unless `val_fold` is None.

        Returns:
        -------
        A tuple of train, validation and test `Subset`s of the given dataset.
        """
        splits = np.zeros_like(dataset.strata_df.index.to_numpy())
        for fold_idx, (_, test_idx) in enumerate(self.split(dataset, dataset.strata_df)):
            splits[test_idx] = self.fold_to_split(fold_idx, val_fold, test_fold).value

        subsets: list[Subset | None] = [None] * 3
        for idx, split in enumerate((Split.TRAIN, Split.VAL, Split.TEST)):
            if split.value in splits:
                subsets[idx] = Subset(dataset, np.where(splits == split.value)[0].tolist())

        return tuple(subsets)
