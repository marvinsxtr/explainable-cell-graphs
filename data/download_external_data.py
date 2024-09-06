import logging
import shutil
import uuid
import warnings
import zipfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlretrieve

import click
import geopandas as gpd
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath, CloudPath
from scipy.io import loadmat
from scipy.ndimage import center_of_mass
from shapely.geometry import Point
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@click.command()
@click.option(
    "--local_path",
    type=str,
    default="./data/external/raw",
    help="Path to save the external dataset to.",
)
@click.option(
    "--destination_path",
    type=str,
    help="Optional path to copy the pre-processed dataset to.",
    default=None,
    required=False,
)
def main(local_path: str, destination_path: str | None) -> None:
    """Downloads and pre-processes the external dataset."""
    download_path = Path(local_path) / "tmp"
    download_path.mkdir(exist_ok=True, parents=True)

    download_data_path = download_path / "LungData"
    segm_path = download_data_path / "LUAD_IMC_Segmentation"
    cell_type_path = download_data_path / "LUAD_IMC_CellType"
    patient_metadata_path = download_data_path / "LUAD Clinical Data.xlsx"

    download_external_dataset(download_data_path)

    local_metadata_path = Path(local_path) / "metadata"
    local_metadata_path.mkdir(exist_ok=False, parents=True)

    local_raw_points_path = Path(local_path) / "raw_points"
    local_raw_points_path.mkdir(exist_ok=False, parents=True)

    patient_df = pd.read_excel(patient_metadata_path)
    patient_df = pre_process_metadata(patient_df)
    patient_df.to_parquet(local_metadata_path / "patient_metadata.parquet")

    mapping_df = generate_mapping_df(patient_df)
    mapping_df.to_parquet(local_metadata_path / "case_spot_mapping.parquet")

    all_cell_labels = get_cell_types(cell_type_path)

    logger.info("Pre-processing spots...")

    for case_uuid in tqdm(patient_df["caseUuid"]):
        key = patient_df[patient_df["caseUuid"] == case_uuid]["KEY"].to_numpy()[0]
        spot_uuid = mapping_df[mapping_df["caseUuid"] == case_uuid]["spotUuid"].to_numpy()[0]
        geo_cell_df = process_single_spot(
            spot_uuid, key, cell_type_path, all_cell_labels, segm_path
        )
        geo_cell_df.to_parquet(local_raw_points_path / f"{spot_uuid}.parquet")

    if destination_path is not None:
        dst_path = AnyPath(destination_path)

        if isinstance(dst_path, CloudPath):
            dst_path.upload_from(local_path)
        elif isinstance(dst_path, Path):
            shutil.copytree(local_path, dst_path)
        else:
            raise TypeError("Could not extract destination path.")


def pre_process_metadata(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-process the patient metadata."""
    orig_cols = patient_df.columns.copy()

    patient_df["OS_M"] = patient_df["Survival or loss to follow-up (years)"] * 12
    patient_df["OS_EVENT"] = patient_df["Death (No: 0, Yes: 1)"].astype("bool")
    patient_df["AGE_75+"] = patient_df["Age (<75: 0, ≥75: 1)"]
    patient_df["BMI_30+"] = patient_df["BMI (<30: 0, ≥30: 1)"]
    patient_df["PACK_YEARS_30+"] = patient_df["Pack Years (1-30: 0, ≥30: 1)"]
    patient_df["SEX"] = patient_df["Sex (Male: 0, Female: 1)"].apply(
        lambda gender: "f" if gender == 1 else "m"
    )
    patient_df["STAGE_3+"] = patient_df["Stage (I-II: 0, III-IV:1)"]
    patient_df["SMOKING_STATUS"] = patient_df["Smoking Status (Smoker: 0, Non-smoker:1)"].map(
        {0: 1, 1: 0}
    )
    patient_df["PROGRESSION"] = patient_df["Progression (No: 0, Yes: 1) "]
    patient_df["PATTERN"] = patient_df[
        "Predominant histological pattern (Lepidic:1, Papillary: 2, Acinar: 3, Micropapillary: 4, Solid: 5)"
    ].map({1: "Lepidic", 2: "Papillary", 3: "Acinar", 4: "Micropapillary", 5: "Solid"})
    patient_df["KEY"] = patient_df["Key"]
    patient_df["caseUuid"] = [str(uuid.uuid4()) for _ in range(len(patient_df.index))]

    patient_df = patient_df.drop(orig_cols, axis=1)
    return patient_df


def get_cell_types(cell_type_path: Path) -> list[str]:
    """Extract all cell types."""
    cell_mat = loadmat(cell_type_path / "LUAD_D001.mat")
    all_cell_labels = sorted([x[0] for x in list(cell_mat["allLabels"].squeeze())])
    all_cell_labels.remove("NONE")
    return all_cell_labels


def generate_mapping_df(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a mapping from case UUIDs to spot UUIDs."""
    mapping_df = pd.DataFrame()
    mapping_df["caseUuid"] = patient_df["caseUuid"]
    mapping_df["spotUuid"] = [str(uuid.uuid4()) for _ in range(len(patient_df.index))]
    return mapping_df


def download_external_dataset(data_path: Path) -> None:
    """Download and extract the dataset."""
    if data_path.exists():
        logger.info(f"Skipping download since {data_path} already exists.")
        return

    def hook(t: tqdm) -> Callable:
        """Display file downloads nicely with tqdm."""
        last_b = [0]

        def update_to(b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return update_to

    logger.info("Downloading zip file...")

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="LungData.zip") as t:
        zip_path, _ = urlretrieve(
            "https://zenodo.org/record/7760826/files/LungData.zip", reporthook=hook(t), data=None
        )

    logger.info("Extracting zip file...")

    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(data_path.parent)


def process_single_spot(
    spot_uuid: str,
    key: str,
    cell_type_path: Path,
    all_cell_labels: list[str],
    segm_path: Path,
) -> gpd.GeoDataFrame:
    """Generate a `GeoDataFrame` from a downloaded segmentation mask."""
    # Get list of cell types for all cells in the spot
    cell_types_mat = loadmat(cell_type_path / f"{key}.mat")
    cell_types = cell_types_mat["cellTypes"]
    cell_types_list = [np.nan if len(x) == 0 else x[0] for x in list(cell_types.squeeze())]

    # Prepare one hot encoded cell types
    one_hot_cells = pd.get_dummies(pd.Series(cell_types_list))
    one_hot_cells = one_hot_cells.reindex(columns=all_cell_labels, fill_value=False)

    # Add indices
    one_hot_cells["spot_id"] = [spot_uuid] * len(one_hot_cells.index)
    one_hot_cells["polygon_id"] = [str(uuid.uuid4()) for _ in range(len(one_hot_cells.index))]
    one_hot_cells = one_hot_cells.set_index(["spot_id", "polygon_id"])

    # Get geometry from cell id mask
    cell_id_mask = loadmat((segm_path / key) / "nuclei_multiscale.mat")["nucleiOccupancyIndexed"]
    cell_coordinates = center_of_mass(
        cell_id_mask != 0, cell_id_mask, list(range(1, np.max(cell_id_mask) + 1))
    )
    cell_coordinates = np.array(cell_coordinates)

    # Scale coordinates to millimeters
    cell_coordinates /= 1000
    cell_points = [Point(xy[0], xy[1]) for xy in cell_coordinates]

    # Insert geometry
    geo_cell_df = gpd.GeoDataFrame(one_hot_cells)
    geo_cell_df["geom"] = cell_points
    geo_cell_df = geo_cell_df.set_geometry("geom")

    return geo_cell_df


if __name__ == "__main__":
    main()
