import os
from functools import lru_cache
from typing import Dict, Any, List

from scipy.io import loadmat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Crane .mat files live in ./data
DATA_DIR = os.path.join(BASE_DIR, "data")


def available_crane_files() -> List[dict]:
    \"\"\"Scan the data directory for .mat files and return Dash dropdown options.\"\\"\"
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
    \"\\"\"Load a single .mat file and extract key fields. Cached in memory.

    Supports two layouts:
    - Top-level variables: VMm, VFm, TP_y_m, TP_z_m, Pmax
    - Nested under SPC struct: SPC.VMm, SPC.VFm, etc.
    \"\\"\"
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Crane data file not found: {path}")

    # struct_as_record=False + squeeze_me=True makes MATLAB structs behave like simple objects
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    spc = mat.get("SPC", None)

    def _from_spc(name: str):
        if spc is None:
            return None
        # attribute-style access (preferred)
        if hasattr(spc, name):
            return getattr(spc, name)
        # dict-style fallback
        if isinstance(spc, dict) and name in spc:
            return spc[name]
        return None

    def _get(name: str):
        # try SPC first
        v = _from_spc(name)
        if v is not None:
            return v
        # fall back to top-level
        return mat.get(name, None)

    return {
        "VMm": _get("VMm"),         # main jib angle (deg)
        "VFm": _get("VFm"),         # folding jib angle (deg)
        "TP_y_m": _get("TP_y_m"),   # crane tip outreach (m)
        "TP_z_m": _get("TP_z_m"),   # crane tip height (m)
        "Pmax": _get("Pmax"),       # allowable load matrix
    }
