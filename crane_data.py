import os
from functools import lru_cache
from typing import Dict, Any, List

from scipy.io import loadmat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Now use ./data (not ./data/test)
DATA_DIR = os.path.join(BASE_DIR, "data")


def available_crane_files() -> List[dict]:
    """
    Scan the data directory for .mat files and return Dash dropdown options.
    """
    if not os.path.isdir(DATA_DIR):
        return []

    files = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith(".mat"):
            continue
        label = os.path.splitext(fname)[0]
        files.append({"label": label, "value": fname})
    return files


@lru_cache(maxsize=32)
def load_crane_file(filename: str) -> Dict[str, Any]:
    """
    Load a single .mat file and extract the key fields we care about.
    Result is cached in memory so repeated calls are cheap.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Crane data file not found: {path}")

    mat = loadmat(path)

    def _get(name: str):
        if name not in mat:
            return None
        return mat[name]

    return {
        "VMm": _get("VMm"),         # main jib angle (deg)
        "VFm": _get("VFm"),         # folding jib angle (deg)
        "TP_y_m": _get("TP_y_m"),   # crane tip outreach (m)
        "TP_z_m": _get("TP_z_m"),   # crane tip height (m)
        "Pmax": _get("Pmax"),       # allowable load matrix
    }
