from __future__ import annotations
import numpy as np
from numba import njit, set_num_threads
import os, multiprocessing as mp


GATE_I = 0
GATE_X = 1
GATE_Y = 2
GATE_Z = 3
GATE_H = 4
GATE_S = 5
GATE_CNOT = 6
GATE_CZ = 7
GATE_WEYL = 8
GATE_SWAP = 9

NAME_TO_ID = {
    "I": GATE_I,
    "X": GATE_X,
    "Y": GATE_Y,
    "Z": GATE_Z,
    "H": GATE_H,
    "S": GATE_S,
    "CNOT": GATE_CNOT,
    "CZ": GATE_CZ,
    "SWAP": GATE_SWAP
}

ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}
ID_TO_NAME[GATE_WEYL] = "W_ab"


def pack_circuit_from_qc(qc) -> np.ndarray:
    """
    Pack QuantumCircuit ops into an int64 array G of shape (L,4):
    G[k] = [gate_id, q1, q2, dagger_flag].
    For W_ab gates, q2 is repurposed as 'a' and dagger_flag as 'b'.
    Unknown gates are skipped.
    """
    ops = list(qc)
    packed_ops = []
    for gate, qudits, dagger in ops:
        # logic to handle realized Weyl gates
        if gate.a is not None and gate.b is not None:
            # This is a pre-realized W_ab gate,
            # pack it as [GATE_WEYL, qudit, a, b]
            final_a, final_b = gate.a, gate.b
            if dagger:
                # the dagger is encoded by negating the parameters
                final_a = -final_a
                final_b = -final_b
            packed_ops.append([GATE_WEYL, qudits[0], final_a, final_b])
        else:
            name = str(getattr(gate, "name", "")).upper()
            gid = NAME_TO_ID.get(name, -1)
            if gid < 0:
                continue
            
            q1 = int(qudits[0])
            q2 = int(qudits[1]) if len(qudits) > 1 else -1
            dag_flag = 1 if dagger else 0
            packed_ops.append([gid, q1, q2, dag_flag])

    if not packed_ops:
        return np.empty((0, 4), dtype=np.int64)
        
    return np.array(packed_ops, dtype=np.int64)

@njit(cache=True)
def nb_apply_gate(X, Z, tau_exp, gid, q1, q2, dag, mod_d, mod_2d):
    """
    Apply a single packed gate in-place to (X, Z, tau_exp).
    """
    dir_ = 1 if dag == 0 else -1
    even = (mod_d % 2) == 0

    if gid == GATE_I:
        return

    elif gid == GATE_X:
        tau_exp[:] = (tau_exp - dir_ * 2 * Z[:, q1]) % mod_2d

    elif gid == GATE_Z:
        tau_exp[:] = (tau_exp + dir_ * 2 * X[:, q1]) % mod_2d

    elif gid == GATE_Y:
        tau_exp[:] = (tau_exp + dir_ * 2 * (X[:, q1] - Z[:, q1])) % mod_2d

    elif gid == GATE_H:
        x_old = X[:, q1].copy()
        z_old = Z[:, q1].copy()
        X[:, q1] = (-dir_ * z_old) % mod_d
        Z[:, q1] = ( dir_ * x_old) % mod_d
        tau_exp[:] = (tau_exp + 2 * (X[:, q1] * Z[:, q1])) % mod_2d

    elif gid == GATE_S:
        xj = X[:, q1]
        Z[:, q1] = (Z[:, q1] + dir_ * xj) % mod_d
        if even:
            tau_exp[:] = (tau_exp + dir_ * (xj * xj)) % mod_2d
        else:
            tau_exp[:] = (tau_exp + dir_ * (xj * (xj - 1))) % mod_2d

    elif gid == GATE_CNOT:
        X[:, q2] = (X[:, q2] + dir_ * X[:, q1]) % mod_d
        Z[:, q1] = (Z[:, q1] - dir_ * Z[:, q2]) % mod_d
    
    elif gid == GATE_CZ:
        Z[:, q1] = (Z[:, q1] + dir_ * X[:, q2]) % mod_d
        Z[:, q2] = (Z[:, q2] + dir_ * X[:, q1]) % mod_d
        tau_exp[:] = (tau_exp + 2 * dir_ * (X[:, q1] * X[:, q2])) % mod_2d
        
    elif gid == GATE_WEYL:
        a = q2
        b = dag
        tau_exp[:] = (tau_exp + 2 * b * X[:, q1] - 2 * a * Z[:, q1]) % mod_2d

    elif gid == GATE_SWAP:
        if q1 != q2:
            tmp = X[:, q1].copy()
            X[:, q1] = X[:, q2]
            X[:, q2] = tmp

            tmp = Z[:, q1].copy()
            Z[:, q1] = Z[:, q2]
            Z[:, q2] = tmp

@njit(cache=True)
def nb_apply_circuit(X, Z, tau_exp, G, mod_d, mod_2d):
    """
    Fast path: apply all gates in G without printing.
    """
    L = G.shape[0]
    for k in range(L):
        nb_apply_gate(X, Z, tau_exp, G[k, 0], G[k, 1], G[k, 2], G[k, 3], mod_d, mod_2d)



# ----------------------------------------------------------------------------------------------
# Parallel vectorized kernel
# ----------------------------------------------------------------------------------------------
@njit(cache=True, parallel=True)
def nb_apply_circuit_parallel(X, Z, tau_exp, G, mod_d, mod_2d):
    """
    Vectorized, auto-parallelized version of nb_apply_circuit.
    Uses multiple cores for the slice-wise updates when possible.
    """
    even = (mod_d % 2) == 0
    L = G.shape[0]
    for k in range(L):
        gid = G[k, 0]
        q1  = G[k, 1]
        q2  = G[k, 2]
        dag = G[k, 3]
        dir_ = 1 if dag == 0 else -1

        if gid == GATE_I:
            continue

        elif gid == GATE_X:
            tmp = tau_exp.copy()
            tau_exp[:] = (tmp - dir_ * 2 * Z[:, q1]) % mod_2d

        elif gid == GATE_Z:
            tmp = tau_exp.copy()
            tau_exp[:] = (tmp + dir_ * 2 * X[:, q1]) % mod_2d

        elif gid == GATE_Y:
            tmp = tau_exp.copy()
            tau_exp[:] = (tmp + dir_ * 2 * (X[:, q1] - Z[:, q1])) % mod_2d

        elif gid == GATE_H:
            x_old = X[:, q1].copy()
            z_old = Z[:, q1].copy()
            X[:, q1] = (-dir_ * z_old) % mod_d
            Z[:, q1] = ( dir_ * x_old) % mod_d
            tmp = tau_exp.copy()
            tau_exp[:] = (tmp + 2 * (X[:, q1] * Z[:, q1])) % mod_2d

        elif gid == GATE_S:
            xj = X[:, q1]
            Z[:, q1] = (Z[:, q1] + dir_ * xj) % mod_d
            tmp = tau_exp.copy()
            if even:
                tau_exp[:] = (tmp + dir_ * (xj * xj)) % mod_2d
            else:
                tau_exp[:] = (tmp + dir_ * (xj * (xj - 1))) % mod_2d

        elif gid == GATE_CNOT:
            X[:, q2] = (X[:, q2] + dir_ * X[:, q1]) % mod_d
            Z[:, q1] = (Z[:, q1] - dir_ * Z[:, q2]) % mod_d
        
        elif gid == GATE_CZ:
            Z[:, q1] = (Z[:, q1] + dir_ * X[:, q2]) % mod_d
            Z[:, q2] = (Z[:, q2] + dir_ * X[:, q1]) % mod_d
            tau_exp[:] = (tau_exp + 2 * dir_ * (X[:, q1] * X[:, q2])) % mod_2d
        
        elif gid == GATE_WEYL:
            a = q2
            b = dag
            tau_exp[:] = (tau_exp + 2 * b * X[:, q1] - 2 * a * Z[:, q1]) % mod_2d

        elif gid == GATE_SWAP:
            if q1 != q2:
                tmp = X[:, q1].copy()
                X[:, q1] = X[:, q2]
                X[:, q2] = tmp

                tmp = Z[:, q1].copy()
                Z[:, q1] = Z[:, q2]
                Z[:, q2] = tmp
        
def setup_numba_threads(n_threads: int | None = None) -> int:
    """
    Set Numba's parallel thread count once, before first JIT compile.
    Returns the thread count used.
    """
    n = n_threads or (os.cpu_count() or mp.cpu_count() or 1)
    os.environ["NUMBA_NUM_THREADS"] = str(n)
    set_num_threads(n)
    return n
