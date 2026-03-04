"""
import backend
backend.set_backend('cupy'/'numpy', 'single'/'double')
backend.show_backend_info()
"""

import importlib
from typing import Literal

# --- Global Dtype Definitions ---
# By default, start with single precision.
COMPLEX_DTYPE = None
FLOAT_DTYPE = None
# ---------------------------------

# by default we start with NumPy
xp = importlib.import_module("numpy")

try:
    import cupy as _cp
except ImportError:
    _cp = None

def set_backend(
    name: Literal["numpy", "cupy"],
    precision: Literal["single", "double"] = "single"
) -> None:
    """
    Switch all downstream code to use either NumPy or CuPy with the
    specified precision.
    """
    global xp, COMPLEX_DTYPE, FLOAT_DTYPE
    
    if name == "numpy":
        xp = importlib.import_module("numpy")
    elif name == "cupy":
        if _cp is None:
            raise ImportError("CuPy is not installed. Cannot set backend to 'cupy'.")
        xp = _cp
    else:
        raise ValueError("backend must be 'numpy' or 'cupy'")

    if precision == "single":
        COMPLEX_DTYPE = xp.complex64
        FLOAT_DTYPE = xp.float32
    elif precision == "double":
        COMPLEX_DTYPE = xp.complex128
        FLOAT_DTYPE = xp.float64
    else:
        raise ValueError("precision must be 'single' or 'double'")

    # Clear gate cache to rebuild matrices with the new dtypes
    try:
        from . import gates
        gates.clear_cache()
    except AttributeError: # ImportError:  # TODO: check this change
        pass

def show_backend_info() -> None:
    """
    Prints which xp we're using, and if it's a CuPy ndarray, what device.
    Prints the current precision settings as well.
    """
    print("Array backend:", xp.__name__)
    print(f"Precision: {COMPLEX_DTYPE.__name__}")
    a = xp.zeros(1)
    if _cp is not None and isinstance(a, _cp.ndarray):
        print(" → Running with CuPy on", a.device)
    else:
        print(" → Running with NumPy (CPU)")

# Initialize backend with default settings on import
set_backend('numpy', 'double')