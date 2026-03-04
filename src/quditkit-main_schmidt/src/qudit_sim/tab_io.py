import numpy as np
from .backend import xp
from .tableau import Tableau

SCHEMA = 1

def save_tableau(tab: Tableau, path: str) -> None:
    """Save a Tableau in .npz format.
    
    Parameters
    ----------
    tab : Tableau
        The tableau object to save.
    path : str
        File path (extension '.npz' is added automatically if missing).
    """
    if not path.endswith(".npz"):
        path += ".npz"
    tab.normalize()
    X, Z, T = tab.X, tab.Z, tab.tau_exp
    if xp.__name__ == "cupy":
        import cupy as cp
        X, Z, T = cp.asnumpy(X), cp.asnumpy(Z), cp.asnumpy(T)
    np.savez_compressed(
        path,
        schema=np.int64(SCHEMA),
        n=np.int64(tab.n),
        d=np.int64(tab.d),
        full=np.int8(1 if tab.full else 0),
        X=X,
        Z=Z,
        tau=T,
    )

def load_tableau(path: str) -> Tableau:
    """Load a Tableau from a .npz file.
    
    Parameters
    ----------
    path : str
        File path (extension '.npz' is added automatically if missing).
    
    Returns
    -------
    Tableau
        Reconstructed tableau object.
    """
    if not path.endswith(".npz"):
        path += ".npz"
    with np.load(path, allow_pickle=False) as f:
        n = int(f["n"])
        d = int(f["d"])
        full = bool(int(f["full"]))
        X, Z, T = f["X"], f["Z"], f["tau"]
    t = Tableau.from_rows(n, d, X, Z, T, full=full)
    t.normalize()
    return t