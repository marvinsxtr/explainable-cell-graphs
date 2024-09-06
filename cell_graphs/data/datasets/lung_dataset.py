from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import trange

from cell_graphs.data.structures.graph_bag import GraphBag


class LungDataset(InMemoryDataset, ABC):
    """Base class for graph datasets handling pre-processing, metadata, saving loading."""

    MAPPING_PATH = "metadata/case_spot_mapping.parquet"
    METADATA_PATH = "metadata/patient_metadata.parquet"
    SPOTS_PATH = "raw_points"

    def __init__(
        self,
        root: str,
        target_columns: list[str],
        indication: Literal["SCC", "AC"],
        fuse_columns: list[str] | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        use_tissue_segmentation: bool = True,
    ) -> None:
        """Download and preprocess a lung graph dataset.

        Args:
        ----
            root: Local path used for saving the raw and processed dataset.
            target_columns: Prediction target column names.
            indication: Lung cancer indication. Can be Squamous Cell (SCC) or Adenocarcinoma (AC).
            fuse_columns: Metadata columns to fuse to graph representations.
            transform: Graph transform to apply after preprocessing and loading from disk.
            pre_transform: Graph transform to apply as preprocessing and before saving to disk.
            pre_filter: Filter to apply to remove graphs from the dataset before saving to disk.
            use_tissue_segmentation: Whether to use tissue segmentation classes as cell features.

        Raises:
        ------
        RuntimeError when the preprocessed data on disk does not match the configurated dataset.
        """
        self.target_columns = target_columns
        self.indication = indication
        self.fuse_columns = fuse_columns
        self.use_tissue_segmentation = use_tissue_segmentation
        self.loaded_metadata = False

        super().__init__(
            root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter
        )
        self.load(self.processed_paths[0])

        if not self.loaded_metadata:
            self.load_metadata()

        if len(self._data.num_graphs) != len(self):
            raise RuntimeError("Saved preprocessed data differs from data preprocessed by config.")

    @property
    @abstractmethod
    def cell_classification_categories(self) -> list[str]:
        """Cell classification categories.

        Returns
        -------
        A list of cell classification categories.
        """
        ...

    @property
    @abstractmethod
    def tissue_segmentation_categories(self) -> list[str]:
        """Tissue segmentation categories.

        Returns
        -------
        List of tissue segmentation classes.
        """
        ...

    @property
    @abstractmethod
    def max_spots_per_case(self) -> int:
        """Maximum number of spots per case.

        Returns
        -------
        Maximum number of spots per case.
        """
        ...

    @abstractmethod
    def filter_cases(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Filter the cases by a set of filters.

        Args:
        ----
            patient_df: `DataFrame` containing patient metadata to filter by.

        Returns:
        -------
        Filtered `DataFrame` with patient metadata.
        """
        ...

    @abstractmethod
    def filter_spots(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Filter individual spots by QC criteria.

        Args:
        ----
            mapping_df: `DataFrame` mapping case IDs to spot IDs with QC information.

        Returns:
        -------
        The filtered mapping.
        """
        ...

    @abstractmethod
    def strata_from_metadata(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Return a `DataFrame` containing the aggregated strata for each patient in a string.

        Args:
        ----
            patient_df: `DataFrame` containing patient metadata.

        Returns:
        -------
        A `DataFrame` containing the aggregated strata in a single column.
        """
        ...

    def load_metadata(self) -> None:
        """Load and pre-process metadata.

        Raises
        ------
        RuntimeError if the indices of metadata and strata do not match.
        """
        mapping_df = pd.read_parquet(Path(self.raw_dir) / self.MAPPING_PATH)
        patient_df = pd.read_parquet(Path(self.raw_dir) / self.METADATA_PATH)
        self.spots_path = Path(self.raw_dir) / self.SPOTS_PATH

        patient_df = self.filter_cases(patient_df)
        self.mapping_df = self.filter_spots(mapping_df)

        # Remove cases without any remaining spots after filtering
        self.patient_df = patient_df[patient_df["caseUuid"].isin(self.mapping_df["caseUuid"])]

        self.strata_df = self.strata_from_metadata(self.patient_df)

        if not self.patient_df.index.equals(self.strata_df.index):
            raise RuntimeError("Indices of patient and strata dataframes must be equal.")

        self.loaded_metadata = True

    def idx_to_case_spot_uuids(self, idx: int) -> tuple[str, list[str]]:
        """Return a case UUID and a list of spot UUIDs for an index.

        Args:
        ----
            idx: Integer index of the case to retrieve spot IDs for.

        Returns:
        -------
        A tuple containing the case ID and a list of corresponding spot IDs.
        """
        case_uuid = self.patient_df.iloc[idx]["caseUuid"]
        return case_uuid, self.mapping_df[self.mapping_df["caseUuid"] == case_uuid][
            "spotUuid"
        ].to_list()

    def __len__(self) -> int:
        """Return the number of patients in the dataset.

        Returns
        -------
        The number of patients in the dataset.
        """
        return len(self.patient_df)

    @property
    def raw_file_names(self) -> list[str]:
        """File or folder names which are checked for existence before re-downloading the data.

        Returns
        -------
        List of folders expected to exist in the `raw` subdirectory of the dataset.
        """
        return ["raw_points", "metadata"]

    @property
    def processed_file_names(self) -> list[str]:
        """File or folder names which are checked for existence before re-preprocessing the data.

        Returns
        -------
        List of folders expected to exist in the `processed` subdirectory of the dataset.
        """
        return ["data.pt", "pre_filter.pt", "pre_transform.pt"]

    @property
    def cell_features(self) -> list[str]:
        """Return the full cell feature names.

        Returns
        -------
        A list of cell feature names.
        """
        cell_features = self.cell_classification_categories

        if self.use_tissue_segmentation:
            cell_features += self.tissue_segmentation_categories

        return cell_features

    @property
    def input_dim(self) -> int:
        """Return the input dimension of the dataset.

        Returns
        -------
        The number of input dimensions of the dataset.
        """
        input_dim = len(self.cell_features)

        if self.use_tissue_segmentation:
            input_dim += len(self.tissue_segmentation_categories)

        return input_dim

    def spot_df_to_graph(self, spot_df: gpd.GeoDataFrame) -> Data:
        """Converts a geo dataframe with points into a torch geometric graph.

        Args:
        ----
            spot_df: GeoPandas `DataFrame` containing one-hot encoded cell features and positions.

        Returns:
        -------
        Torch geometric graph object.
        """
        spot_df["geom"] = spot_df["geom"][~spot_df["geom"].is_empty]
        spot_df = spot_df.dropna()

        features = spot_df.reindex(columns=self.cell_features, fill_value=False)
        features = np.array(features, dtype=np.float32)

        points = spot_df["geom"].apply(lambda p: (p.x, p.y)).to_list()
        points = np.array(points, dtype=np.float32)

        graph = Data(x=torch.from_numpy(features), pos=torch.from_numpy(points))

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        return graph

    def graph_to_spot_df(self, data: Data) -> gpd.GeoDataFrame:
        """Converts a torch geometric graph object into a GeoPandas `DataFrame`.

        Args:
        ----
            data: Torch geometric graph object.

        Returns:
        -------
        GeoPandas `DataFrame` for the given torch geometric graph.
        """
        data = data.detach().cpu()
        return gpd.GeoDataFrame(
            data=data.x,
            columns=self.cell_features,
            geometry=gpd.points_from_xy(x=data.pos[:, 0], y=data.pos[:, 1]),
        )

    def process(self) -> None:
        """Create graphs from spot dataframes and write them to disk."""
        self.load_metadata()

        data_list = []
        for idx in trange(len(self)):
            case_uuid, spot_uuids = self.idx_to_case_spot_uuids(idx)
            patient_record = self.patient_df.loc[self.patient_df["caseUuid"] == case_uuid]

            y = torch.from_numpy(patient_record[self.target_columns].to_numpy(dtype=np.float32))
            fuse = torch.from_numpy(patient_record[self.fuse_columns].to_numpy(dtype=np.float32))
            case_uuid = patient_record["caseUuid"].to_numpy()

            bag_data_list = []
            for spot_uuid in spot_uuids:
                spot_df = gpd.read_parquet(self.spots_path / (spot_uuid + ".parquet"))
                graph = self.spot_df_to_graph(spot_df)
                graph.y = y
                graph.fuse = fuse
                graph.case_uuid = case_uuid
                bag_data_list.append(graph)

            bag = GraphBag.from_data_list(bag_data_list)
            bag.add_padding(self.max_spots_per_case)
            data_list.append(bag)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        self.save(data_list, self.processed_paths[0])
