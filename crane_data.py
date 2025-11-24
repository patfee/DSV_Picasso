import os
from functools import lru_cache

from scipy.io import loadmat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def available_crane_files():
    """Return list of Dash dropdown options for all .mat files in the data directory."""
    if not os.path.isdir(DATA_DIR):
        return []

    files = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower().endswith(".mat"):
            label = os.path.splitext(fname)[0]
            files.append({"label": label, "value": fname})
    return files


@lru_cache(maxsize=32)
def load_crane_file(filename):
    """Load crane data from a .mat file.

    Tries SPC.<field> first (e.g. SPC.VMm), then falls back to top-level variables.
    Returns a dict with keys: VMm, VFm, TP_y_m, TP_z_m, Pmax.
    """
    path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Crane data file not found: {path}")

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    spc = mat.get("SPC", None)

    def from_spc(name):
        if spc is None:
            return None
        # Object-like access (scipy struct)
        if hasattr(spc, name):
            return getattr(spc, name)
        # Dict-like, just in case
        if isinstance(spc, dict) and name in spc:
            return spc[name]
        return None

    def get_field(name):
        value = from_spc(name)
        if value is not None:
            return value
        return mat.get(name, None)

    return {
        "VMm": get_field("VMm"),
        "VFm": get_field("VFm"),
        "TP_y_m": get_field("TP_y_m"),
        "TP_z_m": get_field("TP_z_m"),
        "Pmax": get_field("Pmax"),
    }
