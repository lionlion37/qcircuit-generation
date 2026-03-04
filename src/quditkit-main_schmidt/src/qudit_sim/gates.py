import numpy as np
from .backend import xp
from . import backend

# matrix cache
MATRICES: dict[tuple[str,int], np.ndarray] = {}


def _cached(name: str, d: int, builder):
    """Return cached (name,d) matrix or build/store on first use."""
    key = (name, int(d))
    mat = MATRICES.get(key)
    if mat is None:
        mat = builder()
        MATRICES[key] = mat
    return mat


def clear_cache() -> None:
    """
    Clear all cached gate matrices.  Call this whenever xp (numpy/cupy)
    is swapped so that we rebuild with the correct backend.
    """
    MATRICES.clear()


# ================================================================================
# Phases =========================================================================
# ================================================================================

def omega(d: int):
    return xp.exp(2j * xp.pi / d)


def tau(d: int):
    return xp.exp(1j * xp.pi / d)


# ================================================================================
# 1-qudit Gates ==================================================================
# ================================================================================

def I(d: int):
    return _cached("I", d, lambda: xp.eye(d, dtype=xp.bool_))


def X(d: int):
    return _cached("X", d, lambda: xp.roll(xp.eye(d, dtype=xp.bool_), -1, axis=1))


def Z(d: int):
    return _cached("Z", d, lambda: xp.diag(xp.power(omega(d).astype(backend.COMPLEX_DTYPE), xp.arange(d, dtype=xp.int32))).astype(backend.COMPLEX_DTYPE))


def Y(d: int):
    return _cached(
        "Y", d,
        lambda: (tau(d) * X(d) @ Z(d) if d % 2 == 0 else X(d) @ Z(d)).astype(backend.COMPLEX_DTYPE)
    )


def H(d: int):
    def _build():
        j, k = xp.indices((d, d), dtype=xp.int32)
        return (xp.power(omega(d), j * k) / xp.sqrt(d)).astype(backend.COMPLEX_DTYPE)

    return _cached("H", d, _build)


def S(d: int):
    def _build():
        j = xp.arange(d, dtype=xp.int32)
        if d % 2:
            diag = xp.power(omega(d), (j * (j - 1) // 2) % d)
        else:
            diag = xp.power(tau(d), (j * j) % (2 * d))
        return xp.diag(diag).astype(backend.COMPLEX_DTYPE)

    return _cached("S", d, _build)


# ================================================================================
# 2-qudit Gates ==================================================================
# ================================================================================

def _perm_matrix(size, perm):
    P = xp.zeros((size, size), dtype=xp.bool_)
    P[perm, xp.arange(size, dtype=xp.int32)] = 1
    return P


def CNOT(d: int):
    def _build():
        n = d * d
        idx = xp.arange(n, dtype=xp.int32)
        c, t = divmod(idx, d)
        perm = c * d + (t + c) % d
        return _perm_matrix(n, perm)

    return _cached("CNOT_mat", d, _build)

def CZ(d: int):
    return _cached("CZ", d, lambda: xp.diag(
        xp.exp(2j * np.pi / d * (xp.arange(d)[:, None] * xp.arange(d)[None, :])).ravel()
    ))

def CNOTb(d: int):
    def _build():
        n = d * d
        idx = xp.arange(n, dtype=xp.int32)
        c, t = divmod(idx, d)
        perm = ((c + t) % d) * d + t
        return _perm_matrix(n, perm)

    return _cached("CNOTb_mat", d, _build)


def CNOT2(d: int):
    def _build():
        n = d ** 3
        idx = xp.arange(n, dtype=xp.int32)
        c = idx // (d * d)
        s = (idx // d) % d
        t = idx % d
        perm = c * d * d + s * d + (t + c) % d
        return _perm_matrix(n, perm)

    return _cached("CNOT2_mat", d, _build)

def SWAP(d: int):
    def _build():
        n = d * d
        idx = xp.arange(n, dtype=xp.int32)
        q1, q2 = divmod(idx, d)
        # |q1,q2> -> |q2,q1>
        perm = q2 * d + q1
        return _perm_matrix(n, perm)
    return _cached("SWAP", d, _build)

# ---------------------------------------------------------------------------------------
# Reshaping + indexing approach (without explicitly constructing cnot/cz/swap matrices) --
# ---------------------------------------------------------------------------------------

def _apply_permutation(vec, perm):
    """helper function"""
    out = xp.empty_like(vec)
    out[perm] = vec
    return out


def CNOT_reshape_idx(d: int, psi, dagger: bool = False):
    """2-qudit  CNOT  (control first, target second)
    |c,t⟩  →  |c, t±c⟩   (+ for gate, − for dagger)"""
    c = xp.arange(d, dtype=xp.int32)[:, None]  # shape (d,1)
    t = xp.arange(d, dtype=xp.int32)[None, :]  # shape (1,d)
    dt = (t - c) if dagger else (t + c)  # choose ±
    perm = (c * d + dt % d).reshape(-1)
    return _apply_permutation(psi, perm)


def CNOTb_reshape_idx(d: int, psi, dagger: bool = False):
    """2-qudit  CNOTb  (target controls control)
    |c,t⟩  →  |c±t, t⟩   (+ for gate, − for dagger)"""
    c = xp.arange(d, dtype=xp.int32)[:, None]
    t = xp.arange(d, dtype=xp.int32)[None, :]
    dc = (c - t) if dagger else (c + t)
    perm = ((dc % d) * d + t).reshape(-1)
    return _apply_permutation(psi, perm)


def apply_CNOT_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, int],
    dagger: bool = False,
) -> np.ndarray:
    """
    Generalized 2‑qudit CNOT on any pair of wires (control, target).
    First index is always control, second is target;
    for a “backwards” CNOT just swap them in the tuple.
    """
    i, j = qudits
    # reshape to (d,)*n
    psi_nd = psi.reshape([d] * n)
    # bring control, target to front
    axes = [i, j] + [k for k in range(n) if k not in (i, j)]
    tpsi = xp.transpose(psi_nd, axes)
    # flatten first two dims
    flat = tpsi.reshape(d * d, -1)
    # apply small CNOT permutation across rows
    permuted = CNOT_reshape_idx(d, flat, dagger)
    # restore shape
    tpsi2 = permuted.reshape([d, d] + [d] * (n - 2))
    # invert transpose
    inv_axes = xp.argsort(axes)
    out = xp.transpose(tpsi2, inv_axes)
    return out.reshape(d**n)


def apply_CNOTb_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, int],
    dagger: bool = False,
) -> np.ndarray:
    i, j = qudits
    psi_nd = psi.reshape([d] * n)
    axes = [i, j] + [k for k in range(n) if k not in (i, j)]
    tpsi = xp.transpose(psi_nd, axes)
    flat = tpsi.reshape(d * d, -1)
    permuted = CNOTb_reshape_idx(d, flat, dagger)
    tpsi2 = permuted.reshape([d, d] + [d] * (n - 2))
    inv_axes = xp.argsort(axes)
    out = xp.transpose(tpsi2, inv_axes)
    return out.reshape(d**n)


def CNOT2_reshape_idx(d: int, psi, dagger: bool = False):
    """3-qudit  CNOT2  (control – skip – target)
    |c,s,t⟩  →  |c,s, t±c⟩   (+ for gate, − for dagger)"""
    k = xp.arange(d ** 3)
    c = k // (d * d)
    s = (k // d) % d
    t = k % d
    dt = (t - c) if dagger else (t + c)
    perm = c * d * d + s * d + (dt % d)
    return _apply_permutation(psi, perm)

def apply_CZ_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, int],
    dagger: bool = False,
) -> np.ndarray:
    i, j = qudits
    psi_nd = psi.reshape([d] * n)

    # bring (i,j) to the front
    axes = [i, j] + [k for k in range(n) if k not in (i, j)]
    tpsi = xp.transpose(psi_nd, axes)

    # 2D phase grid  phase[c,t] = ω^(± c*t)
    idx = xp.arange(d)
    sign = -1 if dagger else 1
    phase2d = xp.exp(sign * 2j * np.pi * (idx[:, None] * idx[None, :]) / d)

    # broadcast multiply on first two axes only
    tpsi *= phase2d.reshape(d, d, *([1] * (n - 2)))

    # invert transpose and return flat
    inv_axes = xp.argsort(axes)
    return xp.transpose(tpsi, inv_axes).reshape(-1)

def apply_SWAP_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int, int],
    dagger: bool = False,
) -> np.ndarray:
    q1, q2 = qudits
    if q1 == q2:
        return psi
    psi_nd = psi.reshape([d] * n)
    axes = list(range(n))
    axes[q1], axes[q2] = axes[q2], axes[q1]
    out = xp.transpose(psi_nd, axes)
    return out.reshape(d**n)

# ================================================================================
# Weyl Operators =================================================================
# ================================================================================

def W(d: int, a: int, b: int):
    """
    Constructs the d x d Weyl operator W_{a,b} = τ^{-ab} X^a Z^b.
    """
    a %= d
    b %= d

    def _build():
        phase = xp.power(tau(d), -a * b)
        xa = xp.linalg.matrix_power(X(d), a)
        zb = xp.linalg.matrix_power(Z(d), b)
        return (phase * (xa @ zb)).astype(backend.COMPLEX_DTYPE)

    return _cached(f"W_{a}_{b}", d, _build)

def W_dag(d: int, a: int, b: int):
    """
    Constructs the dagger of the Weyl operator W_{a,b}, which is W_{-a,-b}.
    """
    a %= d
    b %= d

    return _cached(f"W_{a}_{b}†", d, lambda: W(d, -a, -b))

def apply_W_reshape_idx(
    psi: np.ndarray,
    n: int,
    d: int,
    qudits: tuple[int,],
    dagger: bool = False,
    a: int = 0,
    b: int = 0
) -> np.ndarray:
    """
    Applies W_{a,b} = τ^{-ab} X^a Z^b to a single qudit j.
    """
    j = qudits[0]
    a_orig, b_orig = a, b
    
    # For the dagger, W_{a,b}^† = W_{-a,-b}
    if dagger:
        a = -a_orig
        b = -b_orig

    a %= d
    b %= d
    
    # Reshape statevector for easy axis access
    psi_nd = psi.reshape([d] * n)
    
    # Apply the global phase τ^{-ab}
    phase_global = xp.power(tau(d), -a * b)
    psi_nd = psi_nd * phase_global
    
    # Apply Z^b (diagonal operator)
    # This adds a phase ω^{bk} to the k-th component along the target axis.
    if b != 0:
        phases = xp.power(omega(d), b * xp.arange(d, dtype=xp.int32))
        # Reshape phases to broadcast only along axis j
        phase_shape = [1] * n
        phase_shape[j] = d
        psi_nd = psi_nd * phases.reshape(phase_shape)

    # Apply X^a (permutation operator)
    # This is a cyclic shift along the target axis.
    if a != 0:
        psi_nd = xp.roll(psi_nd, shift=a, axis=j)

    return psi_nd.reshape(d**n)

# ================================================================================
# Daggers ========================================================================
# ================================================================================

def _dag(m):
    return m.conj().T


_dag = lambda m: xp.conjugate(m.T)
I_dag = I
X_dag = lambda d: _cached("X†", d, lambda: _dag(X(d)))
Y_dag = lambda d: _cached("Y†", d, lambda: _dag(Y(d)))
Z_dag = lambda d: _cached("Z†", d, lambda: _dag(Z(d)))
H_dag = lambda d: _cached("H†", d, lambda: _dag(H(d)))
S_dag = lambda d: _cached("S†", d, lambda: _dag(S(d)))
CNOT_dag = lambda d: _cached("CNOT†", d, lambda: _dag(CNOT(d)))
CNOTb_dag = lambda d: _cached("CNOTb†", d, lambda: _dag(CNOTb(d)))
CNOT2_dag = lambda d: _cached("CNOT2†", d, lambda: _dag(CNOT2(d)))
CZ_dag = lambda d: _cached("CZ†", d, lambda: _dag(CZ(d)))
SWAP_dag = SWAP
