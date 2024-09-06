import os

import pandas as pd
from dotenv import load_dotenv
from gcsfs import GCSFileSystem

from cell_graphs.common.logging.logger import logger
from cell_graphs.common.utils.constants import Indications
from cell_graphs.data.datasets.lung_dataset import LungDataset


class ExternalLungDataset(LungDataset):
    """The preprocessed external lung dataset. Raw data available at
    https://zenodo.org/record/7760826.
    """

    @property
    def cell_classification_categories(self) -> list[str]:
        """Cell classification categories for the external dataset.

        Returns
        -------
        A list of cell classification categories.
        """
        return [
            "Alt MAC",
            "B cell",
            "Cancer",
            "Cl MAC",
            "Cl Mo",
            "DCs cell",
            "Endothelial cell",
            "Int Mo",
            "Mast cell",
            "NK cell",
            "NONE",
            "Neutrophils",
            "Non-Cl Mo",
            "T other",
            "Tc",
            "Th",
            "Treg",
        ]

    @property
    def tissue_segmentation_categories(self) -> list[str]:
        """Tissue segmentation categories for the external dataset.

        Returns
        -------
        An empty list since there is no tissue segmentation available for the external dataset.
        """
        return []

    @property
    def max_spots_per_case(self) -> int:
        """Maximum number of spots per case.

        Returns
        -------
        Maximum number of spots per case in the external dataset.
        """
        return 1

    def download(self) -> None:
        """Download the external dataset."""
        try:
            load_dotenv()
            logger.info("Downloading the dataset...")
            GCSFileSystem().get(os.environ["EXTERNAL_DS_PATH"], self.raw_dir, recursive=True)
        except Exception as e:
            raise RuntimeError(
                "Please download the external dataset manually by running "
                "`python data/download_external_data.py`."
            ) from e

    def filter_cases(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Filter the cases by a set of filters.

        Args:
        ----
            patient_df: `DataFrame` containing patient metadata to filter by.

        Returns:
        -------
        Filtered `DataFrame` with patient metadata.
        """
        if self.indication != Indications.AC:
            raise ValueError("The external dataset only contains adenocarcinoma.")

        if self.fuse_columns is not None:
            patient_df = patient_df.dropna(subset=self.fuse_columns)

        if len(self.target_columns) != 0:
            patient_df = patient_df[(patient_df["OS_M"] > 3)].copy()

        patient_df["RELAPSE"] = None
        patient_df.loc[(patient_df["PROGRESSION"] == True), "RELAPSE"] = True
        patient_df.loc[(patient_df["PROGRESSION"] == False), "RELAPSE"] = False

        patient_df["SURVIVAL"] = None
        patient_df.loc[
            (patient_df["OS_M"] <= 36) & (patient_df["OS_EVENT"] == True), "SURVIVAL"
        ] = False
        patient_df.loc[(patient_df["OS_M"] > 36), "SURVIVAL"] = True

        patient_df = patient_df.dropna(subset=self.target_columns)

        return patient_df.reset_index()

    def filter_spots(self, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Filter individual spots by QC criteria.

        This is a no-op for the external dataset since QC was done in advance.

        Args:
        ----
            mapping_df: `DataFrame` mapping case IDs to spot IDs with QC information.

        Returns:
        -------
        The filtered mapping.
        """
        return mapping_df

    def strata_from_metadata(self, patient_df: pd.DataFrame) -> pd.DataFrame:
        """Return a `DataFrame` containing the aggregated strata for each patient in a string.

        Args:
        ----
            patient_df: `DataFrame` containing patient metadata.

        Returns:
        -------
        A `DataFrame` containing the aggregated strata in a single column.
        """
        if "OS_M" in self.target_columns:
            strata = patient_df[["OS_M", "OS_EVENT", "STAGE_3+", "AGE_75+"]].copy()
            strata["OS_M"] = pd.qcut(strata["OS_M"], q=5).cat.codes
        elif "RELAPSE" in self.target_columns:
            strata = patient_df[["RELAPSE", "STAGE_3+", "AGE_75+"]].copy()
        elif "GRADE" in self.target_columns:
            strata = patient_df[["GRADE", "SEX"]].copy()
        elif "SURVIVAL" in self.target_columns:
            strata = patient_df[["SURVIVAL", "STAGE_3+", "AGE_75+"]].copy()
        else:
            raise ValueError(
                f"Stratification for the target {self.target_columns} is not supported yet."
            )
        return pd.DataFrame({"STRATA": strata.agg(lambda x: "-".join(map(str, x)), axis=1)})
