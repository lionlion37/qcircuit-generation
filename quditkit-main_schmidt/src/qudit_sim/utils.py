from . import backend
from .backend import xp
import numpy as np
from .circuit import *
from typing import Tuple
from .tableau import Tableau
from .predefined_gates import *
from .tableau_numba_fast import setup_numba_threads

def warmup_numba(d: int) -> None:
    qc = QuantumCircuit(2, d)
    for g in (H_gate, S_gate, X_gate, Z_gate, CNOT_gate):
        if g.n_qudits == 1: qc.append(g, 0)
        else: qc.append(g, (0,1))
    tab = Tableau(2, d, full=False)
    tab.apply_circuit(qc, engine="numba")

def warmup_numba_parallel(d: int, n_threads: int | None = None) -> None:
    n = setup_numba_threads(n_threads)
    print(f"Using {n} threads.")
    qc = QuantumCircuit(2, d)
    for g in (H_gate, S_gate, X_gate, Z_gate, CNOT_gate):
        qc.append(g, 0 if g.n_qudits == 1 else (0, 1))
    Tableau(2, d, full=False).apply_circuit(qc, engine="numba", parallel=True)

def measureall(
    psi: np.ndarray,
    n: int,
    d: int,
) -> Tuple[np.ndarray, int]:
    """
    Perform a *global* computational-basis measurement on |psi⟩ ∈ C^{d^n}.

    Returns
    -------
    collapsed : ndarray
        Statevector after projection (|e_k⟩ with amplitude 1).
    k : int
        Linear index 0…d^n-1 of the outcome, i.e. the measured basis state.
    """
    size = d**n
    if psi.shape != (size,):
        raise ValueError(f"psi must be a flat vector of length {size}")

    psi = psi.astype(backend.COMPLEX_DTYPE)
    probs = (xp.abs(psi)**2).astype(backend.FLOAT_DTYPE)
    probs /= xp.sum(probs)

    k = int(xp.random.choice(xp.arange(size, dtype=xp.int64), p=probs))

    collapsed = xp.zeros_like(psi, dtype=backend.COMPLEX_DTYPE)
    collapsed[k] = 1

    return collapsed, k

def flip_endianness_statevector(psi: np.ndarray, n: int, d: int) -> np.ndarray:
    """
    Reverse the qudit axis order of a length d**n statevector.

    Interprets `psi` as shape (d,)*n with axes (q0,…,q{n-1}), reverses to
    (q{n-1},…,q0), then flattens back to 1D. This converts between big- and
    little-endian conventions; applying it twice returns the original vector.
    
    Args
    -------
    psi : np.ndarray
        1D statevector of length d**n.
    n : int
        Number of qudits.
    d : int
        Dimension of qudits.

    Returns
    -------
    numpy.ndarray
        1D array of length d**n (same dtype as `psi`).

    Raises
    ------
    ValueError
        If `psi.size != d**n`.
    """
    psi = xp.asarray(psi)
    if psi.size != d**n:
        raise ValueError(f"Expected length {d**n}, got {psi.size}")
    return psi.reshape([d]*n).transpose(tuple(reversed(range(n)))).reshape(-1)

def nonzero_dary_indices(
    vector: np.ndarray,
    d: int,
    tol: float = 1e-12,
) -> list[tuple[int, ...]]:
    """
    Return the non-zero positions of a statevector/probability vector in base-d (d-ary) form.

    Parameters
    ----------
    vector : np.ndarray
        Either a statevector (complex amplitudes) or a probability vector (real, >=0),
        of length d**n for some integer n.
    d : int
        Local dimension (>= 2).
    tol : float, optional
        Threshold below which entries are treated as zero.
        For statevectors this applies to |amp|**2; for probability vectors to p directly.

    Returns
    -------
    list[tuple[int, ...]]
        A list of n-length tuples with digits in base-d (most-significant digit first),
        corresponding to the indices with non-zero probability.
    """
    arr = xp.asarray(vector)
    if arr.ndim != 1:
        raise ValueError(f"`vector` must be 1D, got shape {arr.shape}.")

    if d < 2:
        raise ValueError(f"`d` must be >= 2, got {d}.")

    L = arr.size

    if L == 0:
        raise ValueError("`vector` must be non-empty.")

    n_float = xp.log(L) / xp.log(d)
    n = int(round(n_float))
    if d**n != L:
        raise ValueError(f"Length {L} is not a power of d={d}. Expected d**n for some integer n.")


    if xp.iscomplexobj(arr):
        mask = (arr.real**2 + arr.imag**2) > tol
    else:
        mask = arr > tol

    idxs = xp.nonzero(mask)[0]
    if idxs.size == 0:
        return []


    powers = d ** xp.arange(n - 1, -1, -1, dtype=xp.int64)
    results = []
    for idx in idxs:
        rem = int(idx)
        digits = [0] * n
        for k in range(n):
            q, rem = divmod(rem, powers[k])
            digits[k] = q
        results.append(tuple(digits))

    return results

def get_zero_vector(n: int, d: int) -> np.ndarray:
    """
    Creates the zero state vector.

    Args:
        n (int): The number of qudits.
        d (int): The dimension of qudits.

    Returns:
        np.ndarray: The state vector of shape ((d**n),).
    """
    size = d**n
    psi = xp.zeros(size, dtype=backend.COMPLEX_DTYPE)
    psi[0] = 1
    return psi


def entangle_input(n: int, d: int) -> np.ndarray:
    """
    Creates the state vector for a maximally entangled state of 2n qudits.
    This state consists of n entangled pairs, forming a universal input state.
    The state is: (1/sqrt(d^n)) * sum_{i=0}^{d^n-1} |i>|i>

    Args:
        n (int): The number of qudits in each of the two registers.
                 The total number of qudits will be 2n.
        d (int): The dimension of each qudit.

    Returns:
        np.ndarray: The state vector of shape (d**(2*n),).
    """
    total_qudits = 2 * n
    size = d**total_qudits
    num_basis_states = d**n

    val = backend.COMPLEX_DTYPE(1) / xp.sqrt(xp.array(num_basis_states, dtype=backend.FLOAT_DTYPE))
    psi = xp.zeros(size, dtype=backend.COMPLEX_DTYPE)
    
    for i in range(num_basis_states):
        index = i * num_basis_states + i
        psi[index] = val
        
    return psi

def collapse_universal_to_basis(vec_2n, n, d, basis_index=0, normalize=True):
    """
    Extract U|basis_index⟩ from (U ⊗ I)|Φ⟩ produced by a universal (2n-qudit) mat-vec run.

    vec_2n: length d**(2n) vector (output of (U ⊗ I)|Φ⟩)
    n, d : as usual
    basis_index: which |i⟩ on the RIGHT register to pick (default 0 -> |0…0⟩)
    Returns: length d**n vector equal to U|i⟩ (up to global phase)
    """
    D = d ** n
    M = vec_2n.reshape(D, D)              # rows = left (output), cols = right
    psi = xp.sqrt(D) * M[:, basis_index]  # column i = U|i⟩ / √D
    if normalize:
        nrm = xp.linalg.norm(psi)
        if nrm != 0:
            psi = psi / nrm
    return psi.astype(backend.COMPLEX_DTYPE, copy=False)

def compare_statevectors(psi1, psi2, atol=1e-8):
    """
    Check if two statevectors are equal up to a global phase.

    Parameters
    ----------
    psi1, psi2 : array_like
        Statevectors to compare. Can be 1D or broadcastable to 1D.
    atol : float, optional
        Absolute tolerance for the fidelity test. If inputs are complex64,
        a larger tolerance is used automatically.

    Returns
    -------
    bool
        True if the statevectors represent the same physical state,
        False otherwise.
    """
    a = xp.asarray(psi1).reshape(-1)
    b = xp.asarray(psi2).reshape(-1)

    if a.shape != b.shape:
        return False

    if xp.isnan(a).any() or xp.isnan(b).any() or xp.isinf(a).any() or xp.isinf(b).any():
        raise ValueError("Statevector contains NaN or Inf, cannot compare.")

    is_32 = a.dtype in (xp.float32, xp.complex64) or b.dtype in (xp.float32, xp.complex64)
    eps_floor = 1e-6 if is_32 else atol
    atol = max(atol, eps_floor)

    n1 = xp.linalg.norm(a)
    n2 = xp.linalg.norm(b)
    if n1 <= atol or n2 <= atol:
        raise ValueError("One or both statevectors have near-zero norm and cannot be compared.")

    u1 = a / n1
    u2 = b / n2

    # phase-invariant overlap
    fid = xp.abs(xp.vdot(u1, u2))

    return xp.isclose(fid, 1.0, atol=atol, rtol=0.0)

def norm_diff(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    Computes the norm of the difference between two state vectors.
    Args:
        psi1 (np.ndarray): First state vector.
        psi2 (np.ndarray): Second state vector.
    Returns:
        float: The norm of the difference ||ψ1 - ψ2||.
    """
    if psi1.shape != psi2.shape:
        raise ValueError(f"State vectors must have the same shape, but got {psi1.shape} and {psi2.shape}")
    atol = 1e-8
    psi1_nonzero = psi1[xp.abs(psi1) > atol]
    psi2_nonzero = psi2[xp.abs(psi2) > atol]
    phase_factor = psi1_nonzero[0] / psi2_nonzero[0]
    diff = xp.linalg.norm(psi1 - (psi2 * phase_factor))
    return diff

def apply_unitary(
    unitary: np.ndarray,
    n: int,
    d: int,
    psi0: np.ndarray | None = None
) -> np.ndarray:
    """
    Applies a given d^n x d^n unitary matrix to a state vector.

    Args:
        unitary (np.ndarray): The unitary matrix to apply.
        n (int): The number of qudits in the system.
        d (int): The dimension of each qudit.
        psi0 (np.ndarray, optional): The initial state vector. If None,
                                     starts in the |0...0> state.

    Returns:
        np.ndarray: The final state vector after applying the unitary.
    """
    size = d**n

    if unitary.shape != (size, size):
        raise ValueError(f"Unitary matrix must have shape ({size}, {size}), but got {unitary.shape}")

    psi = xp.zeros(size, dtype=backend.COMPLEX_DTYPE) if psi0 is None else psi0.copy().astype(backend.COMPLEX_DTYPE)
    if psi0 is None:
        psi[0] = 1

    unitary = unitary.astype(backend.COMPLEX_DTYPE)
    final_psi = unitary @ psi

    if xp.__name__ == 'cupy':
        return xp.asnumpy(final_psi)
    return final_psi



# ----------------------------------------------------------------------------------------------
# Random Circuit Generation
# ----------------------------------------------------------------------------------------------
def _parse_gate_key(key: str) -> tuple[str, bool]:
    """
    Parse names like 'X', 'Xdag', 'CNOT', 'CNOTdag' (case-insensitive).
    Returns (BASE_NAME, is_dagger).
    """
    k = key.strip()
    is_dagger = k.endswith("dag") or k.endswith("DAG")
    base = k[:-3] if is_dagger else k
    base = base.upper()
    if base == "CX":
        base = "CNOT"
    return base, is_dagger

def _prepare_weighted_gates(gateset: dict[str, float], num_qudits: int):
    """
    Validate, filter, and normalize the gateset.

    Rules:
      - Unknown gate names → error
      - Probabilities < 0 → error
      - All zero or empty → error
      - If num_qudits == 1: drop 2-qudit gates; error if none left
      - Normalize remaining probabilities to sum to 1

    Returns a list of (Gate, is_dagger, prob) with prob normalized.
    """
    if not gateset:
        raise ValueError("gateset is empty.")

    _BASE_GATE = {
    "I": I_gate,
    "X": X_gate,
    "Y": Y_gate,
    "Z": Z_gate,
    "H": H_gate,
    "S": S_gate,
    "CNOT": CNOT_gate,
    "CX": CNOT_gate,
    "CZ": CZ_gate,
    "SWAP": SWAP_gate,
    }
    
    unknown, negative = [], []
    items, total = [], 0.0
    for name, p in gateset.items():
        base, is_dag = _parse_gate_key(name)
        if base not in _BASE_GATE:
            unknown.append(name)
            continue
        p = float(p)
        if p < 0:
            negative.append((name, p))
            continue
        if p == 0:
            continue
        gate = _BASE_GATE[base]
        items.append((gate, is_dag, p))
        total += p

    if unknown:
        raise ValueError(f"Unknown gate names in gateset: {unknown}")
    if negative:
        raise ValueError(f"Negative probabilities in gateset: {negative}")
    if total == 0 or not items:
        raise ValueError("All gate probabilities are zero.")

    if num_qudits == 1:
        oneq = [(g, d, p) for (g, d, p) in items if g.n_qudits == 1]
        if not oneq:
            raise ValueError("num_qudits=1 but gateset contains only 2-qudit gates.")
        s = sum(p for (_, _, p) in oneq)
        return [(g, d, p / s) for (g, d, p) in oneq]

    s = sum(p for (_, _, p) in items)
    return [(g, d, p / s) for (g, d, p) in items]

def random_circuit(
    num_qudits: int,
    dim: int,
    max_size: int = 10,
    max_depth: int = 10,
    gateset: dict[str, float] | None = None,
    adjacent_only: bool = False,
    seed: int | None = None,
) -> QuantumCircuit:
    """
    Build a random QuantumCircuit with up to `max_size` total gates and up to
    `max_depth` *true* circuit layers (parallel time steps).

    The circuit will satisfy:
        qc.size  ≤ max_size
        qc.depth ≤ max_depth

    Supported gates: I, H, S, Hdag, Sdag, X, Y, Z, 
    Xdag, Ydag, Zdag, CNOT, CNOTdag, CZ, CZdag, SWAP.
    
    Gates are sampled from a probability dict like:
      {"X": p1, "Xdag": p2, "H": p3, "CNOT": p4, "CNOTdag": p5, ...}
      
      Example:
      gateset = {
        "I": 1.0,
        "H": 1.0, "S": 1.0, "Hdag": 1.0, "Sdag": 1.0,
        "X": 1.0, "Y": 1.0, "Z": 1.0,
        "Xdag": 1.0, "Ydag": 1.0, "Zdag": 1.0,
        "CNOT": 2.5, "CNOTdag": 2.5,
        "CZ": 2.5, "CZdag": 2.5, "SWAP": 2.5
      }
        
    Probabilities are normalized automatically (after filtering out invalid/2-qudit
    gates when num_qudits==1).

    _DEFAULT_GATESET = {
      "I": 1.0,
      "H": 1.0, "S": 1.0, "Hdag": 1.0, "Sdag": 1.0,
      "X": 1.0, "Y": 1.0, "Z": 1.0,
      "Xdag": 1.0, "Ydag": 1.0, "Zdag": 1.0,
      "CNOT": 1.0, "CNOTdag": 1.0,
      "CZ": 1.0, "CZdag": 1.0,
    }

    The circuit stores operations as (Gate, qudits_tuple, dagger_bool).
    `adjacent_only=True` means 2-qudit gates (e.g., CNOT/CZ) are applied only to
    neighboring qudits; the control/target orientation is randomized.

    Args:
        num_qudits: Number of qudits (wires).
        dim: Dimension of each qudit.
        max_size: Maximum number of gates to insert in total.
        max_depth: Maximum number of *parallel layers* to construct.
        gateset: Probability map of gate names to weights (see examples above).
        adjacent_only: If True, restrict 2-qudit gates to adjacent pairs.
        seed: RNG seed for reproducibility.

    Returns:
        QuantumCircuit: A randomly generated circuit with true depth ≤ `max_depth`
                        and size ≤ `max_size`.
    """
    rng = np.random.default_rng(seed)

    _DEFAULT_GATESET = {
        "I": 1.0,
        "H": 1.0, "S": 1.0, "Hdag": 1.0, "Sdag": 1.0,
        "X": 1.0, "Y": 1.0, "Z": 1.0,
        "Xdag": 1.0, "Ydag": 1.0, "Zdag": 1.0,
        "CNOT": 1.0, "CNOTdag": 1.0,
        "CZ": 1.0, "CZdag": 1.0,
    }

    weighted = _prepare_weighted_gates(gateset or _DEFAULT_GATESET, num_qudits)
    gate_objs  = [g for (g, d, p) in weighted]
    dag_flags  = [d for (g, d, p) in weighted]
    probs      = np.array([p for (g, d, p) in weighted], dtype=np.float64)
    probs     /= probs.sum()

    qc = QuantumCircuit(num_qudits, dim, name="random_circuit")
    count = 0

    # Build up to max_depth parallel layers
    for _ in range(max_depth):
        if count >= max_size:
            break

        used = set()  # qudits occupied in this layer
        layer_gates = int(rng.integers(1, num_qudits + 1))  # attempt this many placements

        for _ in range(layer_gates):
            if count >= max_size:
                break

            idx = int(rng.choice(len(gate_objs), p=probs))
            gate = gate_objs[idx]
            is_dag = bool(dag_flags[idx])

            if gate.n_qudits == 1:
                free = [q for q in range(num_qudits) if q not in used]
                if not free:
                    break
                q = int(rng.choice(free))
                qc.append(gate, (q,), dagger=is_dag)
                used.add(q)
                count += 1
                continue

            # 2-qudit gate
            if num_qudits < 2:
                continue  # should not happen if _prepare_weighted_gates filtered properly

            if adjacent_only:
                pairs = [(i, i + 1) for i in range(num_qudits - 1)
                         if i not in used and (i + 1) not in used]
                if not pairs:
                    break
                i, j = pairs[int(rng.integers(len(pairs)))]
                # randomize orientation (control/target) when it matters
                if int(rng.integers(0, 2)) == 1:
                    i, j = j, i
            else:
                free = [q for q in range(num_qudits) if q not in used]
                if len(free) < 2:
                    break
                i, j = rng.choice(free, size=2, replace=False)
                i, j = int(i), int(j)

            qc.append(gate, (i, j), dagger=is_dag)
            used.update((i, j))
            count += 1

    return qc

# ================================================================================
# Density Matrix Helper Functions ================================================
# ================================================================================

def get_zero_density_matrix(n: int, d: int) -> np.ndarray:
    """
    Creates the density matrix for the pure state |0...0><0...0|.

    Returns:
        np.ndarray: A d^n x d^n density matrix.
    """
    size = d**n
    rho = xp.zeros((size, size), dtype=backend.COMPLEX_DTYPE)
    rho[0, 0] = 1.0
    return rho

def is_valid_density_matrix(rho: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Checks if a matrix is a valid density matrix.
    
    A matrix rho is a valid density matrix if it is square, has a trace of 1,
    and is Hermitian (equal to its own conjugate transpose).

    Args:
        rho: The matrix to check.
        tol: The numerical tolerance for the checks.

    Returns:
        bool: True if the matrix is a valid density matrix, False otherwise.
    """
    # Check 1: Must be a 2D square matrix
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        return False
        
    # Check 2: Trace must be 1
    trace = xp.trace(rho)
    if not xp.isclose(trace, 1.0, atol=tol):
        return False
        
    # Check 3: Must be Hermitian
    if not xp.allclose(rho, xp.conjugate(rho.T), atol=tol):
        return False
        
    return True

def purity(rho: np.ndarray) -> float:
    """
    Calculates the purity of a density matrix, Tr(rho^2).
    
    Purity is 1.0 for a pure state and < 1.0 for a mixed state.

    Returns:
        float: The purity of the state.
    """
    if not is_valid_density_matrix(rho):
        raise ValueError("Input is not a valid density matrix.")
    
    return xp.real(xp.trace(rho @ rho))

def density_matrix_to_probvector(rho: np.ndarray) -> np.ndarray:
    """
    Extracts the measurement probability vector from a density matrix.

    The probability of measuring each computational basis state is given by the
    diagonal elements of the density matrix.

    Returns:
        np.ndarray: A 1D array of size d^n with the measurement probabilities.
    """
    if not is_valid_density_matrix(rho):
        raise ValueError("Input is not a valid density matrix.")
        
    return xp.real(xp.diag(rho))


# ================================================================================
# Circuit Generator Functions ====================================================
# ================================================================================


def univ_input_state_circuit(n, d):
    """
    Construct a universal input state circuit on 2n qudits of local dimension d.

    The circuit prepares n maximally entangled pairs (generalized Bell states)
    between the first n and the last n qudits. For each i in [0, n-1], it applies:

      1. A Hadamard gate H on qudit i (creating superposition).
      2. A controlled-X (CNOT) from qudit i (control) to qudit i+n (target),
         entangling the registers.

    Parameters
    ----------
    n : int
        Number of entangled pairs (half the total number of qudits).
        The total system size is 2n qudits.
    d : int
        Local dimension of each qudit (d ≥ 2).

    Returns
    -------
    QuantumCircuit
        A circuit on 2n qudits of dimension d that prepares the universal input state.
    """
    qc_univ = QuantumCircuit(2 * n, d)
    for i in range(n):
        qc_univ.H(i)
        qc_univ.CX(i, i + n)
    return qc_univ

def encoding_circuit(n: int, d: int, base: int = 2) -> QuantumCircuit:
    """
    Creates the circuit on fig. 3 from this article:
    http://doi.org/10.1088/0253-6102/55/5/11

    Places H on qudits from 0 to n-2.
    
    Places base^i (i from 0 to n-2) CZ gates from qudit n-2 (control) to n-1 (target),
    n-2 to n, ..., 0 to n.
    
    The total number of gates in the circuit:
    total_gates = n-1 + (1-base^(n-1))/(1-base)
    
    Parameters
    ----------
        n (int): Number of qudits
        d (int): Dimension
        base (int): Base which is raised to powers 0...n-2
    
    Returns
    -------
        QuantumCircuit
    
    Raises
    ------
        ValueError if `n` < 2
    """
    if n < 2:
        raise ValueError(f"n must be > 1, provided n: {n}")
    qc = QuantumCircuit(n, d, name='encoding_circuit')
    for i in range(n-2, -1, -1):
        qc.H(i)
        for j in range(base ** (n-2-i)):
            qc.CZ(i, n-1)
    return qc


# ================================================================================
# Schmidt rank & rank vector =====================================================
# ================================================================================

def _rank_from_rref(M: np.ndarray, d: int) -> int:
    """
    Compute the rank of a matrix M over integers mod d.
    """
    rref = Tableau._rref_mod_d(M.copy(), d)
    rank = np.sum(np.any(rref!= 0, axis=1))
    return rank


def _get_stabilizer_blocks(tab: Tableau) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the stabilizer X and Z n by n blocks of a tableau.
    """
    n = tab.n

    if tab.full:
        X_stab = tab._X[n:]
        Z_stab = tab._Z[n:]
    else:
        X_stab = tab._X
        Z_stab = tab._Z

    return X_stab, Z_stab

def schmidt_rank(tab: Tableau, 
                         A: set[int] | list[int] | tuple[int],
                         B: set[int] | list[int] | tuple[int]) -> tuple[int, int]:
    """
    Compute the Schmidt exponent k and Schmidt rank d^k across partition A | B.

    Given a pure stabilizer state (n stabilizers on n qudits),
    the Schmidt rank across bipartition A|B is d^k.

    Partition requirements:
    - A and B must be disjoint sets of qudit indices.
    - A | B must cover all qudits {0,1,...,n-1}.
    
    This function:
      - extracts stabilizer generators,
      - restricts them to subsystem A and B,
      - computes ranks r_A and r_B via RREF mod d,
      - uses the formula k = (r_A + r_B - n) // 2.

    Parameters
    ----------
    tab : Tableau
        The stabilizer tableau (full or stabilizer-only).
    A : Iterable[int]
        Indices of qudits making subsystem A.
    B : Iterable[int]
        Indices of qudits making subsystem B.

    Returns
    -------
    (k, d^k) : Tuple[int, int]
        k   = number of d-dimensional Bell pairs across A|B
        d^k = Schmidt rank across this bipartition
    """
    n = tab.n
    d = tab.d

    # validate partitions
    A_set = set(A)
    B_set = set(B)

    if A_set & B_set:
        raise ValueError(f"Partitions A and B overlap: A & B = {A_set & B_set}")

    all_indices = set(range(n))
    if A_set | B_set != all_indices:
        raise ValueError(
            f"A|B must cover all qudits. Got A|B = {A_set|B_set}, expected {all_indices}"
        )
    
    X_stab, Z_stab = _get_stabilizer_blocks(tab)

    # build matrices X_A|Z_A and X_B|Z_B
    A = list(A)
    B = list(B)

    X_A = X_stab[:, A]
    Z_A = Z_stab[:, A]
    M_A = np.concatenate([X_A, Z_A], axis=1)

    X_B = X_stab[:, B]
    Z_B = Z_stab[:, B]
    M_B = np.concatenate([X_B, Z_B], axis=1)

    # compute ranks
    r_A = _rank_from_rref(M_A, d)
    r_B = _rank_from_rref(M_B, d)

    # k = number of Bell pairs
    k = (r_A + r_B - n) // 2
    rank = d ** k

    return k, rank


def schmidt_rank_vector(tab: Tableau) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Schmidt rank vector for a stabilizer state.

    For each qudit i, computes the Schmidt rank across partition:
        A = {i}
        B = all other qudits

    Returns two arrays:
      - k_vec[i] = number of d-dimensional Bell pairs between qudit i and the rest
      - r_vec[i] = d^(k_vec[i]) = actual Schmidt rank for qudit i vs rest

    Parameters
    ----------
    tab : Tableau
        The stabilizer tableau.

    Returns
    -------
    (k_vec, r_vec) : Tuple[np.ndarray, np.ndarray]
        k_vec : np.ndarray of shape (n,)
            Schmidt exponents for each qudit.
        r_vec : np.ndarray of shape (n,)
            Schmidt ranks = d^k_i for each qudit.
    """
    n = tab.n
    all_qudits = set(range(n))

    k_vec = []
    r_vec = []

    for i in range(n):
        A = {i}
        B = all_qudits - A
        k_i, r_i = schmidt_rank(tab, A, B)
        k_vec.append(k_i)
        r_vec.append(r_i)

    return np.array(k_vec), np.array(r_vec)