"""Crane data loading and management utilities."""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

__all__ = ["available_crane_files", "load_crane_file", "CraneData", "DATA_DIR"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


class DropdownOption(TypedDict):
    """Type for Dash dropdown options."""

    label: str
    value: str


class CraneData(TypedDict):
    """Type for crane data structure."""

    VMm: Optional[NDArray[np.floating[Any]]]
    VFm: Optional[NDArray[np.floating[Any]]]
    TP_y_m: Optional[NDArray[np.floating[Any]]]
    TP_z_m: Optional[NDArray[np.floating[Any]]]
    Pmax: Optional[NDArray[np.floating[Any]]]


class DataDirectoryError(Exception):
    """Raised when the data directory is not accessible."""

    pass


def available_crane_files() -> List[DropdownOption]:
    """
    Return list of Dash dropdown options for all .mat files in the data directory.

    Returns:
        List of dropdown options with 'label' and 'value' keys.
        Returns empty list if directory doesn't exist or has no .mat files.
    """
    if not os.path.isdir(DATA_DIR):
        return []

    files: List[DropdownOption] = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower().endswith(".mat"):
            label = os.path.splitext(fname)[0]
            files.append({"label": label, "value": fname})
    return files


def is_data_directory_available() -> bool:
    """Check if the data directory exists and contains .mat files."""
    return os.path.isdir(DATA_DIR) and len(available_crane_files()) > 0


@lru_cache(maxsize=32)
def load_crane_file(filename: str) -> CraneData:
    """
    Load crane data from a .mat file.

    Tries SPC.<field> first (e.g. SPC.VMm), then falls back to top-level variables.

    Args:
        filename: Name of the .mat file to load (relative to DATA_DIR)

    Returns:
        Dict with keys: VMm, VFm, TP_y_m, TP_z_m, Pmax

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be parsed
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Crane data file not found: {path}")

    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        raise ValueError(f"Failed to parse .mat file '{filename}': {exc}") from exc

    spc = mat.get("SPC", None)

    def from_spc(name: str) -> Optional[NDArray[np.floating[Any]]]:
        if spc is None:
            return None
        # Object-like access (scipy struct)
        if hasattr(spc, name):
            return getattr(spc, name)
        # Dict-like, just in case
        if isinstance(spc, dict) and name in spc:
            return spc[name]
        return None

    def get_field(name: str) -> Optional[NDArray[np.floating[Any]]]:
        value = from_spc(name)
        if value is not None:
            return value
        return mat.get(name, None)

    return CraneData(
        VMm=get_field("VMm"),
        VFm=get_field("VFm"),
        TP_y_m=get_field("TP_y_m"),
        TP_z_m=get_field("TP_z_m"),
        Pmax=get_field("Pmax"),
    )
