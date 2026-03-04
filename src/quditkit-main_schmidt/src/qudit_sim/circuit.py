from __future__ import annotations
from typing import Callable, Union, Dict, Tuple, Literal, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib import colors as mcolors
from collections import Counter
import copy
import random
import numpy as np
from .backend import xp
from . import backend
from .predefined_gates import *
from .gate_class import Gate, NoiseChannel

# optional dependency ---------------------------------------------------------
try:
    from qiskit.circuit import Gate as QiskitGate, QuantumCircuit as QiskitCircuit
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False


# -----------------------------------------------------------------------------
# Circuit container -----------------------------------------------------------
class QuantumCircuit:
    def __init__(self, num_qudits: int, dim: int, name: str | None = None):
        self.n = num_qudits
        self.d = dim
        self.name = name
        # Store tuples: (Gate, indices, dagger_flag)
        self._ops: list[tuple[Gate, tuple[int, ...], bool]] = []
        
        # --- Noise Registry System ---
        # The key is a unique model ID (int).
        # The value is the normalized probability array.
        self.noise_registry: dict[int, np.ndarray] = {}
        # A cache to find existing models to avoid storing duplicate arrays.
        # The key is a hashable tuple of the probability array.
        self._noise_model_cache: dict[tuple, int] = {}
        # A counter to generate new model IDs.
        self._next_noise_model_id = 0
        
        # --- Cache for Kraus Operators ---
        self._kraus_operator_cache: dict[int, list[np.ndarray]] = {}

    def __repr__(self):
        return (
            f"QuantumCircuit(num_qudits={self.n}, dim={self.d}, "
            f"name={self.name!r}, size={self.size}, depth={self.depth})"
        )

    def __str__(self):
        return (
            f"<QuantumCircuit: "
            f"name={self.name!r}, n={self.n}, d={self.d}, size={self.size}, depth={self.depth}>"
        )
    
    def __getitem__(self, idx: int) -> Tuple[Gate, Tuple[int, ...], bool]:
        """
        Allow qc[idx] to return the (gate, targets, dagger) tuple
        corresponding to the operation at position idx in qc._ops.
        """
        return self._ops[idx]
    
    def __iter__(self):
        yield from self._ops

    def __len__(self):
        return len(self._ops)

    def __copy__(self) -> QuantumCircuit:
        """Hook for copy.copy(): make a shallow copy of this circuit."""
        new = QuantumCircuit(self.n, self.d, name=self.name)
        new._ops = list(self._ops)
        
        new.noise_registry = self.noise_registry.copy()
        new._noise_model_cache = self._noise_model_cache.copy()
        new._next_noise_model_id = self._next_noise_model_id
        
        return new

    def copy(self) -> QuantumCircuit:
        """Return a shallow copy of this circuit."""
        return copy.copy(self)
    
    @property
    def ops(self) -> list[tuple[Gate, tuple[int, ...], bool]]:
        """
        Return the list of operations (gates) in the circuit.
        """
        return self._ops
    
    @property
    def size(self) -> int:
        """Total number of gates in the circuit, excluding measurements."""
        return sum(1 for gate, _, _ in self._ops if gate.name != "M")
    
    @property
    def depth(self) -> int:
        """
        Calculate circuit depth = number of parallel layers in the ASAP schedule.
        Measurement gates are not counted.
        """
        cols = self._asap_schedule_packed()
        return sum(1 for col in cols if any(g.name != "M" for g, _, _ in col))
    
    def _asap_schedule_packed(self):
        unitary_ops = [op for op in self._ops if op[0].name != "M"]
        measurement_ops = [op for op in self._ops if op[0].name == "M"]

        next_free = {q: 0 for q in range(self.n)}
        cols, latest = [], -1

        for gate, qudits, dagger in unitary_ops:
            qs = set(qudits)                     # <- only touched wires
            col = max(next_free[q] for q in qs)
            latest = max(latest, col)
            while len(cols) <= col:
                cols.append([])
            cols[col].append((gate, qudits, dagger))
            for q in qs:
                next_free[q] = col + 1

        if measurement_ops:
            i = latest + 1
            while len(cols) <= i:
                cols.append([])
            cols[i].extend(measurement_ops)
        return cols

    def change_dim(self, new_d: int):
        """Change the dimension of the qudits in this circuit."""
        self.d = new_d

    def append(self, gate: Gate, qudits: tuple[int, ...], dagger: bool = False):
        """
        Appends a new gate to the circuit, specifying which qudits it acts on and whether it is a dagger gate.
        """
        if type(qudits) is int:
            qudits = (qudits,)
        if any(q < 0 or q >= self.n for q in qudits):
            raise IndexError("qudit index out of range")
        if len(qudits) != gate.n_qudits:
            raise ValueError(f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qudits)}")
        if gate.n_qudits == 2 and qudits[0] == qudits[1]:
            raise ValueError(f"Cannot apply {gate.name} to the same qudit (#{qudits[0]}) twice")
        self._ops.append((gate, qudits, dagger))

    # --------------------------------------------------------------------------------------------------------------
    # Alternative gate addition methods
    # --------------------------------------------------------------------------------------------------------------
    
    def I(self, q: int):
        """Add I on qudit q."""
        return self.append(I_gate, q)
    
    def Idag(self, q: int):
        """Add I† on qudit q."""
        return self.append(I_gate, q, dagger=True)
    
    def X(self, q: int):
        """Add X on q."""
        return self.append(X_gate, q)
    
    def Xdag(self, q: int):
        """Add X† on q."""
        return self.append(X_gate, q, dagger=True)

    def Y(self, q: int):
        """Add Y on q."""
        return self.append(Y_gate, q)
    
    def Ydag(self, q: int):
        """Add Y† on q."""
        return self.append(Y_gate, q, dagger=True)

    def Z(self, q: int):
        """Add Z on q."""
        return self.append(Z_gate, q)
    
    def Zdag(self, q: int):
        """Add Z† on q."""
        return self.append(Z_gate, q, dagger=True)
    
    def H(self, q: int):
        """Add H on q."""
        return self.append(H_gate, q)
    
    def Hdag(self, q: int):
        """Add H† on q."""
        return self.append(H_gate, q, dagger=True)

    def S(self, q: int):
        """Add S on q."""
        return self.append(S_gate, q)
    
    def Sdag(self, q: int):
        """Add S† on q."""
        return self.append(S_gate, q, dagger=True)

    def CNOT(self, c: int, t: int):
        """Add CNOT(c,t)."""
        return self.append(CNOT_gate, (c, t))
    
    def CNOTdag(self, c: int, t: int):
        """Add CNOT†(c,t)."""
        return self.append(CNOT_gate, (c, t), dagger=True)
    
    def CX(self, c: int, t: int):
        """Alias of CNOT(c,t)."""
        return self.append(CNOT_gate, (c, t))

    def CXdag(self, c: int, t: int):
        """Alias of CNOT†(c,t)."""
        return self.append(CNOT_gate, (c, t), dagger=True)

    def CZ(self, c: int, t: int):
        """Add CZ(c,t)."""
        return self.append(CZ_gate, (c, t))

    def CZdag(self, c: int, t: int):
        """Add CZ†(c,t)."""
        return self.append(CZ_gate, (c, t), dagger=True)

    def SWAP(self, q1: int, q2: int):
        """Add SWAP(q1,q2)."""
        return self.append(SWAP_gate, (q1, q2))

    def SWAPdag(self, q1: int, q2: int):
        """Add SWAP†(q1,q2)."""
        return self.append(SWAP_gate, (q1, q2), dagger=True)

    def _W(self, a: int, b: int, color: str = "#FF0000"):
        a = a % self.d
        b = b % self.d
        
        weyl_gate = W_gate.copy(name=f"W_{a},{b}", color=color)
        
        weyl_gate.a = a
        weyl_gate.b = b
        
        weyl_gate.matrix_fn = lambda d_val: gates.W(d_val, a, b)
        weyl_gate.dagger_matrix_fn = lambda d_val: gates.W_dag(d_val, a, b)
        
        return weyl_gate

    def W(self, q: int, a: int, b: int, color: str = "#FF0000"):
        """Add Weyl(a,b) on q."""
        weyl_gate = self._W(a, b, color=color)
        return self.append(weyl_gate, q, dagger=False)

    def Wdag(self, q: int, a: int, b: int, color: str = "#FF0000"):
        """Add Weyl†(a,b) on q."""
        weyl_gate = self._W(a, b, color=color)
        return self.append(weyl_gate, q, dagger=True)

    def M(self, q: int):
        """Add measurement on q."""
        return self.append(M_gate, q)
    
    def P(self, q: int):
        """ Add a placeholder gate on q."""
        return self.append(P_gate, q)

    def Pdag(self, q: int):
        """ Add a daggered placeholder gate on q."""
        return self.append(P_gate, q, dagger=True)

    def GROUP(self, q1: int, q2:int):
        """ Add a placeholder group gate on qudits q1,q2."""
        return self.append(GROUP_gate, (q1,q2))

    # def NOISE(self, q: int):
    #     """ Add a placeholder noise gate on qudit q."""
    #     return self.append(NOISE_gate, q)
    
    
    # -----------------------------------------------------------------------
    # Op-list utilities (indexing on self._ops)
    # -----------------------------------------------------------------------
    def find(self, name: str) -> list[tuple[int, tuple[int, ...]]]:
        """
        Return all indices i and qudits qs such that self._ops[i][0].name == name.
        Matching is exact (case-sensitive).
        
        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        return [(i, qs) for i, (g, qs, _) in enumerate(self._ops) if g.name == name]

    def insert_at_position(
        self,
        pos: int,
        gate: Gate,
        qudits: int | tuple[int, ...],
        dagger: bool = False,
    ) -> None:
        """
        Insert (gate, qudits, dagger) at list index `pos` in self._ops.

        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        if not isinstance(pos, int):
            raise TypeError("pos must be an int")
        if pos < 0 or pos > len(self._ops):
            raise IndexError(f"pos {pos} out of range 0..{len(self._ops)}")

        # normalize & validate targets
        if isinstance(qudits, int):
            qudits = (qudits,)
        if any(q < 0 or q >= self.n for q in qudits):
            raise IndexError("qudit index out of range")
        if len(qudits) != gate.n_qudits:
            raise ValueError(f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qudits)}")
        if gate.n_qudits == 2 and qudits[0] == qudits[1]:
            raise ValueError(f"Cannot apply 2-qudit gate {gate.name} to the same wire twice")

        self._ops.insert(pos, (gate, tuple(qudits), bool(dagger)))

    def replace_at_position(
        self,
        pos: int,
        gate: Gate,
        qudits: int | tuple[int, ...],
        dagger: bool = False,
    ) -> None:
        """
        Replace the op at index `pos` with (gate, qudits, dagger).
        
        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        if not isinstance(pos, int):
            raise TypeError("pos must be an int")
        if pos < 0 or pos >= len(self._ops):
            raise IndexError(f"pos {pos} out of range 0..{len(self._ops)-1}")

        # normalize & validate targets
        if isinstance(qudits, int):
            qudits = (qudits,)
        if any(q < 0 or q >= self.n for q in qudits):
            raise IndexError("qudit index out of range")
        if len(qudits) != gate.n_qudits:
            raise ValueError(f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qudits)}")
        if gate.n_qudits == 2 and qudits[0] == qudits[1]:
            raise ValueError(f"Cannot apply 2-qudit gate {gate.name} to the same wire twice")

        self._ops[pos] = (gate, tuple(qudits), bool(dagger))

    def delete_at_position(self, pos: int) -> None:
        """
        Delete the operation at index `pos` from self._ops.
        
        NOTE: This operates on program order (the raw op list), not on ASAP columns.
        """
        if not isinstance(pos, int):
            raise TypeError("pos must be an int")
        if pos < 0 or pos >= len(self._ops):
            raise IndexError(f"pos {pos} out of range 0..{len(self._ops)-1}")
        del self._ops[pos]

    # --------------------------------------------------------------------------------------------------------------

    def insert(
        self,
        gate: Gate,
        qudits: int | tuple[int, ...],
        column: int,
        dagger: bool = False
    ):
        """
        Inserts a new gate at index `column` by adding a new column
        to the circuit. Supports both 1- and 2-qudit gates.

        All existing gates that were in column `column` or later are
        shifted right by one column.

        Args:
            gate    (Gate): a Gate instance (e.g. S_gate, CNOT_gate)
            qudits  (int | tuple[int, ...]): which wire(s) the gate acts on
            column (int):   zero-based column index at which to insert
            dagger  (bool): if True, insert the †-version of `gate`

        Raises:
            ValueError  if `gate.n_qudits` is mismatched, or `column < 0`
            IndexError  if `qudit` out of range
        """
        qts = (qudits,) if isinstance(qudits, int) else qudits
        if len(qts) != gate.n_qudits:
            raise ValueError(f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qts)}")
        if any(q < 0 or q >= self.n for q in qts):
            raise IndexError("qudit index out of range")
        if column < 0:
            raise ValueError(f"Column must be >=0, got {column}")

        cols = self._asap_schedule()

        while len(cols) <= column:
            cols.append([])

        cols.insert(column, [(gate, qts, dagger)])

        new_ops: list[tuple[Gate,tuple[int,...],bool]] = []
        for col_ops in cols:
            new_ops.extend(col_ops)

        self._ops = new_ops

    def replace(
        self,
        gate: Gate,
        qudits: int | tuple[int, ...],
        column: int,
        dagger: bool = False
    ):
        """
        Replace or insert a gate at the given column.

        Args:
            gate: Gate instance to insert.
            qudits: single int (1-qudit) or tuple of two ints (2-qudit).
            column: zero-based ASAP-scheduled column index.
            dagger: if True, apply dagger version.

        Raises:
            ValueError: if gate.n_qudits mismatched or column invalid.
            IndexError: if any qudit index out of range.
        """
        if isinstance(qudits, int):
            qts = (qudits,)
        else:
            qts = qudits
        if len(qts) != gate.n_qudits:
            raise ValueError(f"Gate {gate.name} expects {gate.n_qudits} qudits, got {len(qts)}")
        for q in qts:
            if not (0 <= q < self.n):
                raise IndexError(f"Qudit index {q} out of range 0..{self.n-1}")
        if column < 0:
            raise ValueError(f"Column index must be ≥0, got {column}")

        cols = self._asap_schedule()
        if column >= len(cols):
            raise ValueError(f"Column {column} is out of range (0..{len(cols)-1})")

        col_ops = cols[column]
        removed_indices = [idx for idx, (_, existing_qidx, _) in enumerate(col_ops) 
                           if any(q in existing_qidx for q in qts)]
        if removed_indices:
            insert_pos = removed_indices[0]
            for idx in reversed(removed_indices):
                col_ops.pop(idx)
        else:
            insert_pos = len(col_ops)

        col_ops.insert(insert_pos, (gate, tuple(qts), dagger))

        new_ops: list[tuple[Gate, tuple[int, ...], bool]] = []
        for ops in cols:
            new_ops.extend(ops)
        self._ops = new_ops

    def delete(
        self,
        qudit: int,
        column: int
    ):
        """
        Delete the gate at the specified column on the given qudit.
        If the qudit is part of a multi-qudit gate, that entire gate is deleted.

        Args:
            qudit: wire index to target.
            column: zero-based ASAP-scheduled column index.

        Raises:
            ValueError: if column invalid or no gate at that position.
            IndexError: if qudit index out of range.
        """
        if not (0 <= qudit < self.n):
            raise IndexError(f"Qudit index {qudit} out of range 0..{self.n-1}")
        if column < 0:
            raise ValueError(f"Column index must be ≥0, got {column}")

        cols = self._asap_schedule()
        if column >= len(cols):
            raise ValueError(f"Column {column} is out of range 0..{len(cols)-1}")

        col_ops = cols[column]
        for idx, (_, qidx, _) in enumerate(col_ops):
            if qudit in qidx:
                col_ops.pop(idx)
                new_ops: list[tuple[Gate, tuple[int, ...], bool]] = []
                for ops in cols:
                    new_ops.extend(ops)
                self._ops = new_ops
                return
        raise ValueError(f"No gate to delete at column {column} on qudit {qudit}")

    def compose(
        self,
        other: QuantumCircuit,
        in_place: bool = False
    ) -> QuantumCircuit:
        """
        Append `other` circuit to this one.

        If `other` has more qudits, the circuit is extended to include them.

        Args:
            other: Another QuantumCircuit to append.
            in_place: If True, modifies the current circuit. If False (default),
                      returns a new QuantumCircuit.

        Returns:
            The modified circuit (either self or a new object).

        Raises:
            ValueError: if the dimensions (`d`) of the circuits differ.
        """
        if self.d != other.d:
            raise ValueError(
                f"Cannot compose circuits with different dimensions: {self.d} vs {other.d}"
            )

        ## Select the target circuit
        if in_place:
            target_circuit = self
        else:
            target_circuit = self.copy()

        # Adjust qudit count
        target_circuit.n = max(self.n, other.n)

        # Append gates from the second circuit
        for gate, qidx, dagger in other._ops:
            if isinstance(gate, NoiseChannel):
                # Use add_noise_channel to merge the noise model
                # into the target circuit's registry.
                model = other.noise_registry[gate.model_id]
                model_reshaped = model.reshape((other.d, other.d))
                target_circuit.add_noise_channel(qidx[0], model_reshaped, color=gate.color)
            else:
                target_circuit.append(gate, qidx, dagger)
        
        return target_circuit
    
    def compose_vertical(
        self,
        other: QuantumCircuit,
        in_place: bool = False
    ) -> QuantumCircuit:
        """
        Vertically compose two circuits by stacking `other`'s qudits below `self`.

        Args:
            other: Another QuantumCircuit to stack below.
            in_place: If True, modifies the current circuit. If False (default),
                      returns a new QuantumCircuit.

        Returns:
            The modified circuit (either self or a new object).

        Raises:
            ValueError: if the qudit dimensions d differ.
        """
        if self.d != other.d:
            raise ValueError(
                f"Cannot vertically compose circuits with different dimensions: {self.d} vs {other.d}"
            )

        ## Select the target circuit
        if in_place:
            target_circuit = self
        else:
            target_circuit = self.copy()

        offset = self.n
        
        # Update total qudit count
        target_circuit.n = self.n + other.n
        
        # Append gates from the second circuit with shifted qudit indices
        for gate, qidx, dagger in other._ops:
            shifted_qidx = tuple(q + offset for q in qidx)
            if isinstance(gate, NoiseChannel):
                model = other.noise_registry[gate.model_id]
                model_reshaped = model.reshape((other.d, other.d))
                target_circuit.add_noise_channel(shifted_qidx[0], model_reshaped, color=gate.color)
            else:
                target_circuit.append(gate, shifted_qidx, dagger)

        return target_circuit
    
    def reverse(self, in_place: bool = False) -> QuantumCircuit:
        """
        Return the inverse of this circuit: gates reversed in order and daggered.

        Args:
            in_place: If True, modifies the current circuit. If False (default),
                      returns a new QuantumCircuit.

        Returns:
            The reversed circuit (either self or a new object).
        """
        new_ops = []
        temp_qc_for_registry = QuantumCircuit(self.n, self.d)

        # Rebuild the circuit in reverse, preserving noise models
        for gate, qidx, dagger in reversed(self._ops):
            if isinstance(gate, NoiseChannel):
                # Re-add the noise channel
                model = self.noise_registry[gate.model_id]
                model_reshaped = model.reshape((self.d, self.d))
                noise_instance = temp_qc_for_registry._create_noise_channel_instance(model_reshaped, color=gate.color)
                new_ops.append((noise_instance, qidx, dagger))
            else:
                # Add the daggered version of the original gate
                new_ops.append((gate, qidx, not dagger))

        if in_place:
            target_circuit = self
        else:
            target_circuit = QuantumCircuit(self.n, self.d, name=self.name)

        target_circuit._ops = new_ops
        target_circuit.noise_registry = temp_qc_for_registry.noise_registry
        target_circuit._noise_model_cache = temp_qc_for_registry._noise_model_cache
        target_circuit._next_noise_model_id = temp_qc_for_registry._next_noise_model_id
        
        return target_circuit
    
    # -----------------------------------------------------------------------
    # Simulation methods
    # -----------------------------------------------------------------------
    def measure(self, qudits: int | tuple[int, ...]) -> None:
        """
        Adds a measurement operation to the circuit for a specific qudit.

        Args:
            qudits: The integer index/tuple of indices of the qudits to measure.
        """
        if isinstance(qudits, int):
            qudits = (qudits,)
        for q in qudits:
            if not (0 <= q < self.n):
                raise IndexError(f"Qudit index {q} is out of range.")
            self.append(M_gate, (q,))
    
    def measure_all(self):
        """
        Adds measurement operations for all qudits in the circuit.
        """
        for qudit in range(self.n):
            self.measure(qudit)
        
    # helper function
    def _perform_measurement(self, psi: np.ndarray, n: int, d: int, qudit: int) -> tuple[np.ndarray, int]:
        """
        Performs a measurement on a single qudit, collapses the state, and returns the outcome.
        """
        psi_nd = psi.reshape([d] * n)
        
        axes_to_sum = tuple(i for i in range(n) if i != qudit)
        probs = xp.sum(xp.abs(psi_nd)**2, axis=axes_to_sum)
        probs /= xp.sum(probs)

        measured_val = xp.random.choice(xp.arange(d, dtype=xp.int32), p=probs)

        new_psi_nd = xp.zeros_like(psi_nd)
        sl = [slice(None)] * n
        sl[qudit] = measured_val
        new_psi_nd[tuple(sl)] = psi_nd[tuple(sl)]
        
        norm = xp.linalg.norm(new_psi_nd)
        if norm > 1e-9:
            new_psi_nd /= norm
        
        non_zero_indices = xp.where(xp.abs(new_psi_nd.flatten()) > 1e-9)[0]
        if len(non_zero_indices) > 0:
            first_non_zero_idx = non_zero_indices[0]
            phase_value = new_psi_nd.flatten()[first_non_zero_idx]
            new_psi_nd /= phase_value

        return new_psi_nd.reshape(d**n), int(measured_val)

    def sample(
        self,
        shots: int,
        psi0: np.ndarray | None = None,
        normalize: bool = True,
        seed: int | None = None,
        return_digits: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
        """
        Execute the circuit once, obtain the final statevector, and sample `shots`
        joint measurement outcomes across all qudits (computational basis).

        Args:
            shots: Number of samples to draw.
            psi0: Optional initial statevector of length d**n. If None, |0...0⟩ is used.
            normalize: Whether to normalize the final state before sampling.
            seed: Optional RNG seed for reproducibility.
            return_digits: If True, also return the per-qudit outcomes.

        Returns:
            If return_digits is True:
                (psi, digits, flat)
                    psi    : np.ndarray, shape (d**n,), complex
                            Final statevector after executing the circuit once.
                    digits : np.ndarray, shape (shots, n), int in [0, d-1]
                            Per-qudit outcomes for each shot; row i is (q0,...,q_{n-1}).
                    flat   : np.ndarray, shape (shots,), int in [0, d**n - 1]
                            Flat indices into a length-d**n basis vector; each is one
                            joint outcome (i.e., the position a “1” would occupy).
            If return_digits is False:
                (psi, flat)
                    psi    : np.ndarray as above
                    flat   : np.ndarray as above
        """
        if shots <= 0:
            raise ValueError(f"`shots` must be positive, got {shots}.")

        psi, _ = self.execute(psi0=psi0, normalize=normalize)
        psi = xp.asarray(psi)

        probs = xp.abs(psi) ** 2
        s = probs.sum()
        if not xp.isclose(s, 1.0):
            if s == 0:
                raise ValueError("Final statevector has zero norm; cannot sample.")
            probs = probs / s

        rng = xp.random.default_rng(seed)
        flat = rng.choice(probs.size, size=shots, replace=True, p=probs).astype(xp.int64, copy=False)

        if not return_digits:
            return psi, flat

        # Convert flat indices to base-d digits per qudit (q0,…,q_{n-1}); shape -> (shots, n)
        digits_tuple = xp.unravel_index(flat, (self.d,) * self.n)
        digits = xp.column_stack(digits_tuple).astype(xp.int64, copy=False)

        return psi, digits, flat
    
    def execute(self, psi0: np.ndarray | None = None, normalize: bool = True, print_state=False) -> tuple[np.ndarray, dict]:
        """
        Executes the circuit and returns the final statevector and classical measurement results.
        """
        n, d = self.n, self.d
        size = d**n
        psi = xp.zeros(size, dtype=backend.COMPLEX_DTYPE) if psi0 is None else psi0.copy().astype(backend.COMPLEX_DTYPE)
        
        classical_results = {}
        
        if psi0 is None:
            psi[0] = 1
            
        if print_state:
            print(f"Initial state:\n{psi}")

        for gate, qidx, dag in self:
            if gate.matrix_fn is None and gate._apply_fn is None:
                if gate.name == "M":
                    measured_qudit = qidx[0]
                    psi, outcome = self._perform_measurement(psi, n, d, measured_qudit)
                    key = measured_qudit
                    classical_results[key] = outcome
                    if print_state:
                        print(f"Measured qudit {measured_qudit}, outcome={outcome}")
                else:
                    if print_state:
                        print(f"Skipping placeholder gate {gate.name} on qudit(s) {', '.join(map(str, qidx))}")
                    continue
            else:
                psi = gate.apply(psi, n, d, qidx, dag)
                if print_state:
                    print(f"\nState after {gate.name}{'†' if dag else ''}" \
                        f" on qudit{'s' if len(qidx) > 1 else ''}" \
                        f" {', '.join(map(str, qidx))}:\n{psi}")
        if normalize:
            nrm = xp.linalg.norm(psi)
            if nrm == 0:
                raise ValueError("Normalization error: final statevector is zero.")
            psi = (psi / nrm).astype(backend.COMPLEX_DTYPE, copy=False)
        else:
            psi = psi.astype(backend.COMPLEX_DTYPE, copy=False)
        if xp.__name__ == 'cupy':
            return xp.asnumpy(psi), classical_results
        return psi, classical_results

    def execute_density_matrix(
        self,
        initial_state: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Executes the circuit using density matrix simulation.

        This method evolves a density matrix through the circuit. Unitary gates
        are applied as rho' = U * rho * U_dagger. NoiseChannel placeholders
        trigger the application of the corresponding Kraus operators.

        Args:
            initial_state: The initial state for the simulation. Can be:
                - None: Starts in the pure |0...0><0...0| state.
                - A 1D statevector: Starts in the pure rho = |psi><psi| state.
                - A 2D density matrix: Starts in the provided rho.

        Returns:
            np.ndarray: The final d^n x d^n density matrix.
        """
        n, d = self.n, self.d
        size = d**n

        # --- Initialize the density matrix ---
        if initial_state is None:
            rho = xp.zeros((size, size), dtype=backend.COMPLEX_DTYPE)
            rho[0, 0] = 1
        elif initial_state.ndim == 1:  # It's a statevector
            psi = initial_state.astype(backend.COMPLEX_DTYPE)
            rho = xp.outer(psi, xp.conjugate(psi))
        elif initial_state.ndim == 2 and initial_state.shape == (size, size):
            rho = initial_state.astype(backend.COMPLEX_DTYPE)
        else:
            raise ValueError(f"Invalid initial_state. Must be a vector of size {size} or a matrix of size ({size}, {size}).")

        # Clear the cache at the beginning of each run to ensure freshness.
        self._kraus_operator_cache.clear()

        # --- Evolve the density matrix ---
        for gate, qidx, dag in self:
            rho = gate.apply_density(rho, self, qidx, dag)

        return rho
    
    def get_unitary(self) -> np.ndarray:
        """
        Constructs and returns the full d^n x d^n unitary matrix for the entire circuit.

        Returns:
            np.ndarray: The unitary matrix representing the circuit.
        """
        n, d = self.n, self.d
        U_total = xp.eye(d**n, dtype=backend.COMPLEX_DTYPE)

        for gate, qudits, dagger in self._ops:
            if gate.matrix_fn is None and gate._apply_fn is None:
                continue
            U_gate = self._get_gate_unitary(gate, qudits, dagger)
            U_total = U_gate @ U_total

        return U_total
    
    # helper function for get_unitary
    def _get_gate_unitary(self, gate: Gate, qudits: tuple[int, ...], dagger: bool) -> np.ndarray:
        """
        Constructs the full d^n x d^n operator for a single gate by applying it
        to each computational basis vector.
        """
        n, d = self.n, self.d
        size = d**n

        U_gate = xp.zeros((size, size), dtype=backend.COMPLEX_DTYPE)

        for i in range(size):
            basis_vec = xp.zeros(size, dtype=backend.COMPLEX_DTYPE)
            basis_vec[i] = 1

            U_gate[:, i] = gate.apply(basis_vec, n, d, qudits, dagger)
            
        return U_gate
    # -----------------------------------------------------------------------
    # Qiskit integration & draw
    # -----------------------------------------------------------------------
    def to_qiskit(self) -> QiskitCircuit:
        if not _HAS_QISKIT:
            raise ImportError("Qiskit is not installed")
        qc = QiskitCircuit(self.n)
        for gate, qidx, dag in self:
            instr = gate.to_qiskit_instruction(dag)
            qudits = list(qidx)
            # swap qudit order for CNOTb so that Qiskit flips the 0/1 labels
            if gate.name.upper() == "CNOTB":
                qudits = qudits[::-1]
            qc.append(instr, qudits)
        return qc

    @classmethod
    def from_qiskit(cls, qc: QiskitCircuit):
        try:
            from my_genQC.utils.config_loader import get_obj_from_str
        except ImportError:
            raise ImportError("This class method depends on my_genQC package. Please install it to use this feature.")

        n = qc.num_qubits
        d = 2  # Qiskit only supports qubits
        name = None

        qudit_circuit = cls(num_qudits=n, dim=d, name=name)

        for instruction in qc.data:
            gate_name = instruction.operation.name

            dagger = False
            if "†" in gate_name:
                gate_name = gate_name.replace("†", "")
                dagger = True

            qudits_ids = tuple(qubit._index for qubit in instruction.qubits)

            gate = get_obj_from_str(f"qudit_sim.predefined_gates.{gate_name}_gate")

            qudit_circuit.append(gate, qudits_ids, dagger)

        return qudit_circuit


        # --- Noise Registry System ---
        # The key is a unique model ID (int).
        # The value is the normalized probability array.
        self.noise_registry: dict[int, np.ndarray] = {}
        # A cache to find existing models to avoid storing duplicate arrays.
        # The key is a hashable tuple of the probability array.
        self._noise_model_cache: dict[tuple, int] = {}
        # A counter to generate new model IDs.
        self._next_noise_model_id = 0

        # --- Cache for Kraus Operators ---
        self._kraus_operator_cache: dict[int, list[np.ndarray]] = {}


    def draw_qiskit(self, *args, path=None, **kwargs):
        """Draw using Qiskit (kwargs passed to QuantumCircuit.draw)
        Args:
            1. *args: 'mpl' (outputs matplotlib figure)
                      'text' (outputs ASCII art)
            2. path: str, optional, path to save the figure (default: '')
            3. **kwargs: keyword arguments for Qiskit draw method
        """
        qc = self.to_qiskit()
        fig = qc.draw(*args, **kwargs)
        if path:
            fig.savefig(path)
            print(f"\nSaved circuit diagram to '{path}'")
        plt.show()

    # -----------------------------------------------------------------------
    # Custom visualization method
    # -----------------------------------------------------------------------
    def _asap_schedule(self):
        """
        ASAP schedule: each gate goes into the earliest column
        in which all of its *own* qudits are free.
        Measurement gates are grouped in a final column.
        """
        unitary_ops = [op for op in self._ops if op[0].name != "M"]
        measurement_ops = [op for op in self._ops if op[0].name == "M"]

        next_free = {q: 0 for q in range(self.n)}
        cols: list[list[tuple]] = []
        latest_col_used = -1

        for gate, qudits, dagger in unitary_ops:
            lo, hi = min(qudits), max(qudits)
            span = range(lo, hi + 1)
            col = max(next_free[q] for q in span)

            latest_col_used = max(latest_col_used, col)

            while len(cols) <= col:
                cols.append([])
            cols[col].append((gate, qudits, dagger))

            for q in span:
                next_free[q] = col + 1

        if measurement_ops:
            final_col_idx = latest_col_used + 1
            while len(cols) <= final_col_idx:
                cols.append([])
            cols[final_col_idx].extend(measurement_ops)

        return cols

    def _logical_schedule(self):
        """
        Logical (program-order) scheduling: one column per gate,
        in the exact order you appended them.
        """
        return [[(gate, qudits, dagger)] for gate, qudits, dagger in self._ops]

    def draw(
        self,
        scheduling:     str   = "asap",         # "asap", "logical" or "asap_packed"
        font_color:     str   = "black",        # color for labels
        show_edges:     bool  = False,          # whether to show gate edges (default=False)
        gate_colors:    list  = ["#EFBB04",   # custom colors (in hex): pauli, H/S, CNOT, CZ, other
                                 "#0DB8B8",
                                 "#4A90E2",
                                 "#9B59B6",
                                 "#878CED"],
        save_path:      str   | None = None,
        dpi:            int   = 300,
        show_title:     bool  = False,
        show_info:      bool  = False,
        show_image:     bool  = True
    ):
        """
        Draws a circuit diagram using matplotlib.
        Args:
            scheduling:     str, "asap", "logical" or "asap_packed" (default="asap")
                **asap packs gates as left as possible,**
                **logical keeps the order you appended them.**
                **asap_packed creates maximally parallelized layers.**
            font_color:     str, color for gate labels (default="black")
            show_edges:     bool, whether to show gate edges (default=False)
                **If None, edges are not drawn.**
                **If set, edges are drawn in this color.**
            gate_colors:    list, custom colors (in hex) for gates 
                **(default=["#0DB8B8", "#FF6F61", "#4A90E2", "#9B59B6", "#878CED"]),**
                **pauli, H/S, CNOT, CZ, other.**
            save_path:      str  | path to save the figure (default=None)
                **For the best quality use a vector format like .svg or .pdf.**
            dpi (int): Dots per inch for saving raster formats (like .png).
            show_title:     bool | whether to show the circuit title (default=False)
            show_info:      bool | whether to show circuit info (default=False)
            show_image:     bool | whether to display the image (default=True)

        Important:
            the window maximizing button does not trigger automatic resizing of fonts and lines,
            so after clicking on this button, you have to trigger the scaling manually by using 
            matplotlib's Pan/Zoom button in the left bottom corner of the figure window, and then 
            clicking on the "Reset original view" button.
            Saving the figure correctly triggers a resize.
        """
        base_spacing:   float = 1.5             # center‑to‑center gap on both wires & CNOTs (default=1.5)
        gate_w:         float = 1               # 1‑qudit gate width (default=1)
        gate_h:         float = 1               # 1‑qudit gate height (also sets CNOT dot/circle scale) (default=1)

        if not show_edges:
            edgecolor = None
        else:
            edgecolor = "black"

        # pick scheduling method
        if scheduling == "asap":
            cols = self._asap_schedule()
        elif scheduling == "asap_packed":
            cols = self._asap_schedule_packed()
        elif scheduling == "logical":
            cols = self._logical_schedule()
        else:
            raise ValueError(f"Unknown scheduling '{scheduling}' – use 'asap' or 'logical'")
        ncols = len(cols)

        # Global style settings
        mpl.rcParams['path.snap'] = False
        mpl.rcParams['text.antialiased'] = True
        mpl.rcParams['lines.antialiased'] = True
        mpl.rcParams['patch.antialiased'] = True

        # uniform x‑positions for every column
        x_wire_start = base_spacing
        x0 = x_wire_start + gate_w
        xs = [x0 + i*base_spacing for i in range(ncols)]

        # figure setup
        total_width  = xs[-1] + gate_w if xs else x_wire_start + gate_w
        total_height = self.n * base_spacing
        fig, ax = plt.subplots(figsize=(min(total_width, 40), min(total_height, 20)))
        ax.set_aspect('equal', adjustable='box')
        
        window_title = self.name if self.name else "QuantumCircuit"
        fig.canvas.manager.set_window_title(window_title)

        # lists to store text & line artists that need dynamic resizing
        text_artists = []
        line_artists = []

        if show_info:
            info_str = (
                f"name='{self.name}', n={self.n}, d={self.d}, size={self.size}, depth={self.depth}"
            )
            title_obj = ax.set_title(info_str, color=font_color)
            text_artists.append({'artist': title_obj, 'type': 'title'})
        elif show_title and self.name:
            title_obj = ax.set_title(self.name, color=font_color)
            text_artists.append({'artist': title_obj, 'type': 'title'})

        # draw horizontal wires and labels
        ys = {i: (self.n-1-i)*base_spacing for i in range(self.n)}
        x_wire_end_full = xs[-1] + gate_w if xs else x_wire_start + gate_w

        wire_end_x = {i: x_wire_end_full for i in range(self.n)}
        for col_idx, col in enumerate(cols):
            for gate, qudits, dagger in col:
                if gate.name == "M":
                    qudit_idx = qudits[0]
                    wire_end_x[qudit_idx] = xs[col_idx]

        for i in range(self.n):
            current_x_end = wire_end_x[i]
            h_line, = ax.plot(
            [x_wire_start, current_x_end],
            [ys[i], ys[i]],
            color='black',
            linewidth=1.5,
            zorder=0,
            solid_capstyle='butt',
            )
            line_artists.append({'artist': h_line, 'type': 'h_wire'})
            
            label_obj = ax.text(
                x_wire_start - 0.1, ys[i],
                rf"$q_{{{i}}}$",
                ha='right', va='center', zorder=3
            )
            text_artists.append({'artist': label_obj, 'type': 'wire_label'})

        # draw each gate
        for col_idx, col in enumerate(cols):
            x = xs[col_idx]
            for gate, qudits, dagger in col:
                if gate.color:
                    fc = gate.color
                else:
                    pauli_color, hs_color, cnot_color, cz_color, other_color = gate_colors
                    nm = gate.name.upper().rstrip("†")
                    if nm in {"I","X","Y","Z"}: 
                        fc = pauli_color
                    elif nm in {"H","S"}: 
                        fc = hs_color
                    elif nm.startswith("CNOT"): 
                        fc = cnot_color
                    elif nm.startswith("CZ"):
                        fc = cz_color
                    elif nm == "M":
                        fc = 'gray'
                    else: 
                        fc = other_color

                if gate.n_qudits == 1:
                    y = ys[qudits[0]]

                    if gate.name == "M":
                        box = patches.Rectangle((x - gate_w/2, y - gate_h/2), gate_w, gate_h, facecolor=fc, edgecolor=edgecolor, zorder=2)
                        ax.add_patch(box)
                        line_artists.append({'artist': box, 'type': 'gate_edge'})
                        arc = patches.Arc(
                            (x, y - gate_h * 0.1), width=gate_w * 0.7, height=gate_h * 0.6,
                            angle=0, theta1=0, theta2=180, color='white', linewidth=3, zorder=3
                        )
                        ax.add_patch(arc)
                        line_artists.append({'artist': arc, 'type': 'meas_symbol'})
                        
                        arrow, = ax.plot(
                            [x, x + gate_w * 0.35], [y - gate_h * 0.1, y + gate_h * 0.25],
                            color='white', linewidth=3, zorder=3
                        )
                        line_artists.append({'artist': arrow, 'type': 'meas_symbol'})
                    elif gate.name == "NOISE_CHANNEL" or (gate.a is not None and gate.b is not None):
                        identity_condition = gate.a == 0 and gate.b == 0
                        base_radius = gate_w / 2
                        num_pts = 120
                        wiggle_amplitude = base_radius * 0.05
                        freq = 16
                        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
                        radii = base_radius - wiggle_amplitude * np.cos(freq * angles + np.pi)
                        xs_edge = x + radii * np.cos(angles)
                        ys_edge = y + radii * np.sin(angles)

                        vertices = np.column_stack([xs_edge, ys_edge])
                        vertices = np.vstack([vertices, vertices[0]])
                        codes = [patches.Path.MOVETO] + [patches.Path.LINETO] * (len(vertices) - 2) + [patches.Path.CLOSEPOLY]
                        path = patches.Path(vertices, codes)
                        wiggly = patches.PathPatch(
                            path,
                            facecolor=fc if not identity_condition else 'white',
                            edgecolor='black',
                            linewidth=1.5,
                            zorder=2
                        )
                        ax.add_patch(wiggly)
                        line_artists.append({'artist': wiggly, 'type': 'noise_edge'})

                        if gate.a is not None and gate.b is not None:
                            if identity_condition:
                                txt = rf"$\mathit{{{'I'}}}$"
                            else:
                                if dagger:
                                    txt = rf"$\mathit{{W}}_{{{gate.a},{gate.b}}}^{{\dagger}}$"
                                else:
                                    txt = rf"$\mathit{{W}}_{{{gate.a},{gate.b}}}$"
                            text_obj = ax.text(x, y, txt, ha='center', va='center', color=font_color, zorder=3)
                            text_artists.append({'artist': text_obj, 'type': 'gate_label', 'name_len': 1 if identity_condition else len(gate.name) - 1})
                    else:
                        box = patches.Rectangle((x - gate_w/2, y - gate_h/2), gate_w, gate_h, facecolor=fc, edgecolor=edgecolor, zorder=2)
                        ax.add_patch(box)
                        line_artists.append({'artist': box, 'type': 'gate_edge'})
                        if dagger:
                            txt = rf"$\mathit{{{gate.name}}}^{{\dagger}}$"
                        else:
                            txt = rf"$\mathit{{{gate.name}}}$"
                        text_obj = ax.text(x, y, txt, ha='center', va='center', color=font_color, zorder=3)
                        text_artists.append({'artist': text_obj, 'type': 'gate_label', 'name_len': len(gate.name)})
                else:
                    if gate.name.startswith("CNOT"):
                        c, t = qudits
                        y_c, y_t = ys[c], ys[t]
                        v_line, = ax.plot([x, x], [y_c, y_t], linewidth=2, color=fc, zorder=1, solid_capstyle='butt')
                        line_artists.append({'artist': v_line, 'type': 'v_wire'})

                        dot = patches.Circle((x, y_c), gate_h * 0.2, facecolor=fc, edgecolor=None, zorder=3)
                        ax.add_patch(dot)

                        radius = gate_h * 0.3
                        target_circ = patches.Circle((x, y_t), radius, facecolor='white', edgecolor=fc, linewidth=2, zorder=2)
                        ax.add_patch(target_circ)
                        line_artists.append({'artist': target_circ, 'type': 'cnot_symbol_line'})

                        h_line_cross, = ax.plot([x - radius, x + radius], [y_t, y_t], color=fc, linewidth=2, zorder=3, solid_capstyle='butt')
                        line_artists.append({'artist': h_line_cross, 'type': 'cnot_symbol_line'})
                        
                        if not dagger:
                            v_line_cross, = ax.plot([x, x], [y_t - radius, y_t + radius], color=fc, linewidth=2, zorder=3, solid_capstyle='butt')
                            line_artists.append({'artist': v_line_cross, 'type': 'cnot_symbol_line'})
                    elif gate.name.startswith("CZ"):
                        c, t = qudits
                        y_c, y_t = ys[c], ys[t]

                        # vertical line from control to the *edge* of the Z box on the target
                        # (the “edge” is whichever side is closer to the control wire)
                        sign = 1 if y_c > y_t else -1  # top wire has larger y
                        y_edge = y_t + sign * (gate_h * 0.5)

                        v_line, = ax.plot(
                            [x, x], [y_c, y_edge],
                            linewidth=2, color=fc, zorder=1
                        )
                        line_artists.append({'artist': v_line, 'type': 'v_wire'})

                        # control dot (same style/size as CNOT's control)
                        dot = patches.Circle((x, y_c), gate_h * 0.2, facecolor=fc, zorder=3)
                        ax.add_patch(dot)

                        # target: Z box centered on the target wire
                        box = patches.Rectangle(
                            (x - gate_w/2, y_t - gate_h/2),
                            gate_w, gate_h,
                            facecolor=fc, zorder=2
                        )
                        ax.add_patch(box)

                        if dagger:
                            txt = rf"$\mathit{{{'Z'}}}^{{\dagger}}$"
                        else:
                            txt = rf"$\mathit{{{'Z'}}}$"
                        text_obj = ax.text(x, y_t, txt, ha='center', va='center', color=font_color, zorder=3)
                        text_artists.append({'artist': text_obj, 'type': 'gate_label', 'name_len': 1})
                        
                    elif gate.name.startswith("SWAP"):
                        q1, q2 = qudits
                        y1, y2 = ys[q1], ys[q2]

                        # vertical connector
                        v_line, = ax.plot(
                            [x, x], [y1, y2],
                            linewidth=2, color='black', zorder=1, solid_capstyle='round'
                        )
                        line_artists.append({'artist': v_line, 'type': 'swap_v_wire'})

                        cross_scale = 0.3  # fraction of 1-qudit box; controls cross size
                        cross_half = cross_scale * min(gate_w, gate_h)

                        # upper cross
                        l1u, = ax.plot(
                            [x - cross_half, x + cross_half],
                            [y1 - cross_half, y1 + cross_half],
                            color='black', linewidth=2, zorder=3, solid_capstyle='round'
                        )
                        l2u, = ax.plot(
                            [x - cross_half, x + cross_half],
                            [y1 + cross_half, y1 - cross_half],
                            color='black', linewidth=2, zorder=3, solid_capstyle='round'
                        )
                        line_artists.append({'artist': l1u, 'type': 'swap_cross'})
                        line_artists.append({'artist': l2u, 'type': 'swap_cross'})

                        # lower cross
                        l1d, = ax.plot(
                            [x - cross_half, x + cross_half],
                            [y2 - cross_half, y2 + cross_half],
                            color='black', linewidth=2, zorder=3, solid_capstyle='round'
                        )
                        l2d, = ax.plot(
                            [x - cross_half, x + cross_half],
                            [y2 + cross_half, y2 - cross_half],
                            color='black', linewidth=2, zorder=3, solid_capstyle='round'
                        )
                        line_artists.append({'artist': l1d, 'type': 'swap_cross'})
                        line_artists.append({'artist': l2d, 'type': 'swap_cross'})
                    elif gate.name.startswith("GROUP"):
                        gate_text = gate.name[6:]
                        if dagger:
                            gate_text = rf"$\mathit{{{gate_text}}}^{{\dagger}}$"
                        else:
                            gate_text = rf"$\mathit{{{gate_text}}}$"
                        start_q, end_q = min(qudits), max(qudits)
                        y_top = ys[start_q]
                        y_bottom = ys[end_q]

                        box_x = x - gate_w / 2
                        box_y = y_bottom - gate_h / 2
                        box_width = gate_w
                        box_height = y_top - y_bottom + gate_h

                        rect = patches.Rectangle((box_x, box_y), box_width, box_height,
                                                 facecolor=fc, edgecolor=edgecolor, zorder=1)
                        ax.add_patch(rect)
                        line_artists.append({'artist': rect, 'type': 'gate_edge'})

                        y_center = (y_top + y_bottom) / 2
                        text_obj = ax.text(x, y_center, gate_text, ha='center', va='center', rotation=90,
                                           color='black', zorder=3)
                        text_artists.append({'artist': text_obj, 
                                             'type': 'group_label', 
                                             'name_len': len(gate.name[6:]), 
                                             'n_qudits': qudits[1] - qudits[0] + 1})

        # finalize plot
        ax.set_xlim(0, total_width)
        ax.set_ylim(-base_spacing, self.n * base_spacing)
        ax.axis('off')

        # --- DYNAMIC FONT SIZING HANDLER ---
        def on_resize(event):
            ax_bbox = ax.get_window_extent()
            if ax_bbox.width == 0 or ax_bbox.height == 0: return

            y_data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            y_scale = ax_bbox.height / y_data_range
            dpi = fig.get_dpi()

            # line widths
            # ----------------------------------------------
            gate_h_pixels = gate_h * y_scale
            h_wire_lw = max(0.5, gate_h_pixels * 0.03) # 0.xx% of gate height
            v_wire_lw = max(1.0, gate_h_pixels * 0.06)
            meas_lw = max(1.0, gate_h_pixels * 0.03)
            
            noise_edge_lw = max(0.5, gate_h_pixels * 0.03)
            
            for item in line_artists:
                artist = item['artist']
                if item['type'] == 'h_wire':
                    artist.set_linewidth(h_wire_lw)
                elif item['type'] == 'v_wire':
                    artist.set_linewidth(v_wire_lw)
                elif item['type'] == 'meas_symbol':
                    artist.set_linewidth(meas_lw)
                elif item['type'] == 'cnot_symbol_line':
                    artist.set_linewidth(v_wire_lw)
                elif item['type'] == 'noise_edge':
                    artist.set_linewidth(noise_edge_lw)
                elif item['type'] == 'gate_edge':
                    artist.set_linewidth(h_wire_lw)
                elif item['type'] == 'swap_v_wire':
                    artist.set_linewidth(v_wire_lw)
                elif item['type'] == 'swap_cross':
                    artist.set_linewidth(v_wire_lw)
            # ----------------------------------------------
            for item in text_artists:
                artist = item['artist']
                
                if item['type'] == 'title':
                    font_size_pt = fig.get_figwidth() * 1.5
                    artist.set_fontsize(font_size_pt)
                
                elif item['type'] == 'wire_label':
                    wire_lane_h_pixels = base_spacing * y_scale
                    font_height_pixels = wire_lane_h_pixels * 0.3
                    font_size_pt = font_height_pixels * (72 / dpi)
                    artist.set_fontsize(font_size_pt)
                
                elif item['type'] == 'gate_label':
                    gate_h_pixels = gate_h * y_scale
                    font_height_pixels = (gate_h_pixels * 0.6) / np.sqrt(item['name_len'])
                    font_size_pt = font_height_pixels * (72 / dpi)
                    artist.set_fontsize(font_size_pt)

                elif item['type'] == 'cnot_symbol':
                    circ_diameter_pixels = (gate_h * 0.3 * 2) * y_scale
                    font_height_pixels = circ_diameter_pixels * 0.9
                    font_size_pt = font_height_pixels * (72 / dpi)
                    artist.set_fontsize(font_size_pt)

                elif item['type'] == 'group_label':
                    gate_h_pixels = gate_h * y_scale
                    font_height_pixels = (gate_h_pixels * 0.6)  # optionally normalize by np.sqrt(np.sqrt(item['name_len']))
                    font_size_pt = font_height_pixels * (72 / dpi)
                    artist.set_fontsize(font_size_pt)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('resize_event', on_resize)
        ax.callbacks.connect('xlim_changed', on_resize)
        ax.callbacks.connect('ylim_changed', on_resize)
        
        plt.tight_layout()
        
        fig.canvas.draw()
        
        on_resize(None)

        if save_path:
            fig.tight_layout()
            fig.canvas.draw()
            on_resize(None)
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"\nSaved circuit diagram to '{save_path}'")

        if show_image:
            if show_info or show_title:
                fig.subplots_adjust(top=0.92)
                fig.canvas.draw()
            on_resize(None)
            plt.show()
            
    
    # -----------------------------------------------------------------------
    # Noise modelling
    # -----------------------------------------------------------------------
    
    def _prepare_weyl_probabilities(self, probs: np.ndarray | float) -> np.ndarray:
        """
        Validates and normalizes Weyl error probabilities using the circuit's dimension `self.d`.

        Args:
            probs: Either a single float `p` for isotropic noise, or a d x d matrix
                   of probabilities for each Weyl operator W_ab.

        Returns:
            A d*d-element flat array of normalized probabilities.
        """
        if isinstance(probs, (float, int)):
            p = float(probs)
            if not (0.0 <= p <= 1.0):
                raise ValueError("Error probability p must be between 0 and 1.")

            num_errors = self.d * self.d - 1
            if num_errors == 0:  # d=1 case
                return xp.array([1.0])

            prob_per_error = p / num_errors
            p_matrix = xp.full((self.d, self.d), prob_per_error, dtype=xp.float64)
            p_matrix[0, 0] = 1.0 - p

        else:  # It's a matrix
            p_matrix = xp.asarray(probs, dtype=xp.float64)
            if p_matrix.shape != (self.d, self.d):
                raise ValueError(f"Probability matrix must have shape ({self.d}, {self.d}), but got {p_matrix.shape}")
            if xp.any(p_matrix < 0):
                raise ValueError("Probabilities cannot be negative.")

        total_prob = xp.sum(p_matrix)
        if not xp.isclose(total_prob, 1.0):
            if total_prob == 0:
                raise ValueError("Sum of probabilities cannot be zero.")
            p_matrix /= total_prob  # Normalize

        return p_matrix.flatten()

    def _set_color(self, normalized_probs: np.ndarray, color: str):
        """
        Helper to set RGBA color with opacity based on error probability.
        """
        try:
            base_rgba = mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(f"Invalid color format: '{color}'")
        
        # Calculate opacity from the total error probability.
        # The probability of NO error (W_00) is the first element.
        prob_no_error = normalized_probs[0]
        opacity = 1.0 - prob_no_error

        # Create the RGBA color tuple for the gate.
        final_color = (*base_rgba[:3], opacity)

        return final_color

    def _create_noise_channel_instance(self, noise_model: np.ndarray | float, color: str) -> NoiseChannel:
        """
        Internal helper to create a NoiseChannel gate instance using the registry system.
        
        This method normalizes the provided noise model, checks if an identical
        model has been used before, and adds it to a central `noise_registry` if not.
        It then creates a `NoiseChannel` gate containing only an ID that
        points to the model in the registry. This `NoiseChannel` object is introduced
        into the circuit.
        """
        normalized_probs = self._prepare_weyl_probabilities(noise_model)
        model_key = tuple(normalized_probs)
        model_id = self._noise_model_cache.get(model_key)

        if model_id is None:
            model_id = self._next_noise_model_id
            self.noise_registry[model_id] = normalized_probs
            self._noise_model_cache[model_key] = model_id
            self._next_noise_model_id += 1

        final_color = self._set_color(normalized_probs, color)
        return NoiseChannel(model_id=model_id, color=final_color)
    
    def add_noise_channel(
        self,
        qudit: int,
        noise_model: np.ndarray | float,
        color: str = "#FF0000"
        ):
        """
        Adds a noise channel to the circuit.

        Args:
            qudit: The qudit on which the noise channel acts.
            noise_model: The probability model for this channel. Can be:
                - A float `p` (0 <= p <= 1) for isotropic noise.
                    - With probability `1-p`, no error occurs (W_00).
                    - With probability `p/(d^2-1)`, each of the other `W_ab` errors occurs.
                - A d x d numpy array `P` where `P[a, b]` is the probability
                  of the W_ab error.
                    - P[0, 0] (identity) is the probability that no error occurs.
            color: The base color for the noise gate visualization (default: "#FF0000").
                - opacity is set to the total error probability (1 - P[0,0]).

        Raises:
            ValueError: If qudit is not an integer.
            IndexError: If any qudit index is out of range.
        """
        # Sanity checks
        if not isinstance(qudit, int):
            raise ValueError("Qudit index must be an integer.")
        if qudit < 0 or qudit >= self.n:
            raise IndexError("Qudit index out of range.")
        
        noise_gate_instance = self._create_noise_channel_instance(noise_model, color)
        self.append(noise_gate_instance, (qudit,))

    def insert_noise_channel(
        self,
        pos: int | tuple[int, int],
        qudit_or_model: int | np.ndarray | float,
        model_or_none: np.ndarray | float | None = None,
        color: str = "#FF0000"
    ):
        """
        Inserts a noise channel into the circuit using one of two modes.

        Mode 1: ASAP-Scheduled Insertion
            qc.insert_noise_channel((qudit, column), noise_model, color="...")

        Mode 2: Program-Order Insertion
            qc.insert_noise_channel(position_index, qudit, noise_model, color="...")

        Args:
            pos: The position to insert at. Can be an integer for program order
                 or a tuple (qudit, column) for ASAP-scheduled order.
            qudit_or_model: If `pos` is an int, this must be the integer target qudit.
                            If `pos` is a tuple, this must be the noise_model.
            model_or_none: If `pos` is an int, this must be the noise_model.
                           If `pos` is a tuple, this argument should not be provided.
            color: The base color for the noise gate visualization.
        """
        # --- Dispatching logic based on the type of `pos` ---
        if isinstance(pos, tuple):
            # --- ASAP-based Mode ---
            # Signature: insert_noise_channel(pos=(qudit, column), noise_model=qudit_or_model)
            if model_or_none is not None:
                raise TypeError("Too many arguments for ASAP-scheduled insertion. Expected: (pos_tuple, noise_model, color=...).")
            
            target_qudit, column = pos
            noise_model = qudit_or_model

            if not (isinstance(target_qudit, int) and isinstance(column, int)):
                raise ValueError("If pos is a tuple, it must be of the form (qudit, column).")
            if not (0 <= target_qudit < self.n):
                raise IndexError(f"Qudit index {target_qudit} is out of range.")
            if column < 0:
                raise ValueError(f"Column index must be >= 0, got {column}")

            noise_gate_instance = self._create_noise_channel_instance(noise_model, color)
            self.insert(noise_gate_instance, target_qudit, column)

        elif isinstance(pos, int):
            # --- Program-based Mode ---
            # Signature: insert_noise_channel(pos=pos, qudit=qudit_or_model, noise_model=model_or_none)
            program_pos = pos
            target_qudit = qudit_or_model
            noise_model = model_or_none

            if noise_model is None:
                raise TypeError("Missing `noise_model` argument for program-order insertion. Expected: (pos_int, qudit, noise_model, color=...).")
            if not isinstance(target_qudit, int):
                 raise TypeError(f"For program-order insertion, the second argument must be the target qudit (an int), but got {type(target_qudit)}.")

            if not (0 <= program_pos <= len(self._ops)):
                raise IndexError(f"Position {program_pos} is out of range for insertion (0 to {len(self._ops)}).")
            if not (0 <= target_qudit < self.n):
                raise IndexError("Qudit index out of range.")
            
            noise_gate_instance = self._create_noise_channel_instance(noise_model, color)
            self.insert_at_position(program_pos, noise_gate_instance, (target_qudit,))
        
        else:
            raise TypeError("`pos` must be an integer or a tuple of two integers (qudit, column).")
    
    def replace_noise_channel(
        self,
        pos: int | tuple[int, int],
        qudit_or_model: int | np.ndarray | float,
        model_or_none: np.ndarray | float | None = None,
        color: str = "#FF0000"
    ):
        """
        Replaces a gate in the circuit with a noise channel using one of two modes.

        Mode 1: ASAP-Scheduled Replacement
            qc.replace_noise_channel((qudit, column), noise_model, color="...")

        Mode 2: Program-Order Replacement
            qc.replace_noise_channel(position_index, qudit, noise_model, color="...")

        Args:
            pos: The position to replace at. Can be an int or a tuple (qudit, column).
            qudit_or_model: If `pos` is an int, this must be the integer target qudit.
                            If `pos` is a tuple, this must be the noise_model.
            model_or_none: If `pos` is an int, this must be the noise_model.
                           If `pos` is a tuple, this argument should not be provided.
            color: The base color for the noise gate visualization.
        """
        if isinstance(pos, tuple):
            # --- ASAP-based Mode ---
            if model_or_none is not None:
                raise TypeError("Too many arguments for ASAP-scheduled replacement. Expected: (pos_tuple, noise_model, color=...).")
            
            target_qudit, column = pos
            noise_model = qudit_or_model

            if not (isinstance(target_qudit, int) and isinstance(column, int)):
                raise ValueError("If pos is a tuple, it must be of the form (qudit, column).")
            if not (0 <= target_qudit < self.n):
                raise IndexError(f"Qudit index {target_qudit} is out of range.")
            if column < 0:
                raise ValueError(f"Column index must be >= 0, got {column}")

            noise_gate_instance = self._create_noise_channel_instance(noise_model, color)
            self.replace(noise_gate_instance, target_qudit, column)

        elif isinstance(pos, int):
            # --- Program-based Mode ---
            program_pos = pos
            target_qudit = qudit_or_model
            noise_model = model_or_none

            if noise_model is None:
                raise TypeError("Missing `noise_model` argument for program-order replacement. Expected: (pos_int, qudit, noise_model, color=...).")
            if not isinstance(target_qudit, int):
                 raise TypeError(f"For program-order replacement, the second argument must be the target qudit (an int), but got {type(target_qudit)}.")

            if not (0 <= program_pos < len(self._ops)):
                raise IndexError(f"Position {program_pos} is out of range for replacement (0 to {len(self._ops) - 1}).")
            if not (0 <= target_qudit < self.n):
                raise IndexError("Qudit index out of range.")

            noise_gate_instance = self._create_noise_channel_instance(noise_model, color)
            self.replace_at_position(program_pos, noise_gate_instance, (target_qudit,))
            
        else:
            raise TypeError("`pos` must be an integer or a tuple of two integers (qudit, column).")

    # Note:
    # to delete noise gate use delete_at_position or delete
    
    def add_noise_global(self, noise_model: np.ndarray | float, color: str = "#FF0000"):
        """
        Modifies the circuit by adding a noise channel after every gate that is
        not a NoiseChannel placeholder.

        This method implements "noise stacking": it finds the end of any contiguous
        block of NoiseChannel placeholders that immediately follows a gate, and
        inserts the new noise channels there. It treats realized Weyl operators (W_a,b)
        as normal gates and will add noise after them.

        Args:
            noise_model: The global probability model to be used for every inserted channel.
            color: The base color for all inserted noise gate visualizations.
        """
        # Sanity checks
        try:
            self._prepare_weyl_probabilities(noise_model)
        except ValueError as e:
            raise ValueError(f"Invalid global noise model provided: {e}")

        offset = 0
        original_op_count = len(self._ops)

        for i in range(original_op_count):
            current_pos = i + offset
            gate, qudits, _ = self._ops[current_pos]

            if isinstance(gate, NoiseChannel) or gate.name == "NOISE_CHANNEL":
                continue

            base_insertion_pos = current_pos + 1
            
            while base_insertion_pos < len(self._ops):
                next_gate = self._ops[base_insertion_pos][0]
                if isinstance(next_gate, NoiseChannel) or next_gate.name == "NOISE_CHANNEL":
                    base_insertion_pos += 1
                else:
                    break
            
            # Insert new noise channels at the final calculated position.
            num_inserted_this_step = 0
            for q in qudits:
                insertion_pos = base_insertion_pos + num_inserted_this_step
                self.insert_noise_channel(insertion_pos, q, noise_model, color)
                num_inserted_this_step += 1

            # Update the offset by the number of gates we just inserted
            offset += num_inserted_this_step
    
    def add_noise_layer(
        self,
        layer_index: int,
        noise_model: Union[
            float,
            np.ndarray,
            Dict[int, Union[float, np.ndarray, Tuple[Union[float, np.ndarray], str]]]
        ],
        mode: Literal["packed", "visual"] = "packed",
        where: Literal["before", "after"] = "after",
        include_idle: bool = True,
        color: str = "#FF0000",
        in_place: bool = False
    ) -> Optional[QuantumCircuit]:
        """
        Adds a layer of noise channels to a specific gate layer in the circuit.

        This method identifies a gate layer based on a scheduling mode and inserts
        NoiseChannel gates either before or after it. The noise can be applied
        uniformly or specified on a per-qudit basis.

        Args:
            layer_index (int): The zero-based index of the gate layer to target.
            noise_model (Union[float, np.ndarray, Dict]): The noise specification.
                - float: Isotropic noise model applied with a single probability 'p'.
                - np.ndarray: A d x d matrix of probabilities for each Weyl error.
                - Dict: A dictionary to specify noise on a per-qudit basis.
                    - key (int): The qudit index.
                    - value (Union[float, np.ndarray, Tuple]): The noise for that qudit.
                        - float or np.ndarray: The noise model. The default color is used.
                        - Tuple (model, color): A tuple containing the noise model
                          (float or np.ndarray) and a custom hex color string.
            mode (Literal["packed", "visual"]): The scheduling algorithm to use for
                defining layers. "packed" uses the true parallel schedule
                (_asap_schedule_packed), while "visual" uses the layout-oriented
                schedule (_asap_schedule). Defaults to "packed".
            where (Literal["before", "after"]): Whether to insert the noise layer
                "before" or "after" the specified gate layer. Defaults to "after".
            include_idle (bool): If True, noise channels are added to all 'n' qudits.
                If False (default), noise is only added to qudits that are active
                (i.e., have a gate acting on them) in the specified layer.
            color (str): The default hex color for noise gates if not specified
                in a per-qudit dictionary. Defaults to "#FF0000".
            in_place (bool): If True, modifies the circuit directly. If False (default),
                returns a new, modified QuantumCircuit instance.

        Returns:
            Optional[QuantumCircuit]: A new QuantumCircuit object if in_place is False,
            otherwise None.

        Raises:
            ValueError: If an invalid mode, position, or noise_model format is provided.
            IndexError: If layer_index is out of bounds for the selected schedule.
        """
        if mode not in ["packed", "visual"]:
            raise ValueError("mode must be either 'packed' or 'visual'.")
        if where not in ["before", "after"]:
            raise ValueError("where must be either 'before' or 'after'.")
        if layer_index < 0:
            raise IndexError("layer_index cannot be negative.")

        target_circuit = self if in_place else self.copy()

        # Get the schedule which defines the layers
        if mode == "packed":
            schedule = target_circuit._asap_schedule_packed()
        else:  # mode == "visual"
            schedule = target_circuit._asap_schedule()

        if layer_index >= len(schedule):
            raise IndexError(f"layer_index {layer_index} is out of range for the schedule with {len(schedule)} layers.")

        # This function generates the list of noise operations to be inserted
        def _get_noise_ops_for_layer(current_layer):
            if include_idle:
                qudits_to_noise = set(range(target_circuit.n))
            else:
                if not current_layer:
                    return []
                qudits_to_noise = {q for _, qudits, _ in current_layer for q in qudits}

            ops_to_add = []
            for q_idx in sorted(list(qudits_to_noise)):
                model, gate_color = None, color
                if isinstance(noise_model, (float, int, list, np.ndarray)):
                    model = noise_model
                elif isinstance(noise_model, dict):
                    if q_idx not in noise_model: continue
                    spec = noise_model[q_idx]
                    model, gate_color = spec if isinstance(spec, tuple) else (spec, color)
                else:
                    raise TypeError(f"Unsupported type for noise_model: {type(noise_model)}")

                if model is not None:
                    noise_gate = target_circuit._create_noise_channel_instance(model, gate_color)
                    ops_to_add.append((noise_gate, (q_idx,), False))
            return ops_to_add

        # Rebuild the entire _ops list from the schedule, inserting noise at the correct layer
        new_ops = []
        for i, current_layer in enumerate(schedule):
            # If this is the target layer, potentially add noise BEFORE
            if i == layer_index and where == 'before':
                new_ops.extend(_get_noise_ops_for_layer(current_layer))

            # Add the actual gates from the current layer
            new_ops.extend(current_layer)

            # If this is the target layer, potentially add noise AFTER
            if i == layer_index and where == 'after':
                new_ops.extend(_get_noise_ops_for_layer(current_layer))
        
        target_circuit._ops = new_ops

        if not in_place:
            return target_circuit
        return None
    
    def insert_noise_at_coords(
        self,
        coords: list[tuple[int, int]] | None = None,
        noise_model: int | float | list | np.ndarray | None = None,
        rand_coords: list | tuple[tuple[int, int], tuple[int, int], float] | None = None,
        color: str = "#FF0000",
        seed: int | None = None,
        in_place: bool = False,
    ) -> QuantumCircuit | None:
        """
        Inserts noise channels at specified coordinates in the circuit's visual layout.

        This method places noise channels based on the ASAP visual schedule ('_asap_schedule').
        A new column is inserted for each unique column index provided, shifting existing
        gates to the right. All noise channels for the same original column index will be
        placed together in a single new parallel layer.

        Args:
            coords (list[tuple[int, int]] | None): A list of (qudit, column) coordinates.
            noise_model (int | float | list | np.ndarray | None): The noise model to apply. Can be:
                - A float | int `p` (0 <= p <= 1) for isotropic noise.
                - A d x d list | numpy array `P` where `P[a, b]` is the probability of the W_ab error.
            rand_coords (list | tuple | None): If provided, generates random coordinates.
                Format: ((q_start, q_end), (col_start, col_end), placement_prob)
                - (q_start, q_end): Inclusive range of qudit indices.
                - (col_start, col_end): Inclusive range of column indices.
                - placement_prob: Probability (0 to 1) of placing a noise channel at any
                  given coordinate within the defined grid.
            color (str): The base color for the noise gate visualization.
            seed (int | None): RNG seed for reproducibility when using `rand_coords`.
            in_place (bool): If True, modifies the circuit directly. If False (default),
                             returns a new, modified QuantumCircuit instance.

        Returns:
            QuantumCircuit | None: A new QuantumCircuit object if in_place is False,
                                   otherwise None.

        Raises:
            ValueError: If both `coords` and `rand_coords` are provided, if neither is
                        provided, if `noise_model` is missing, or if any arguments are
                        invalid (e.g., out of range, wrong format).
        """
        if (coords is not None and rand_coords is not None) or \
           (coords is None and rand_coords is None):
            raise ValueError("Provide either 'coords' or 'rand_coords', but not both.")
        
        if noise_model is None:
            raise ValueError("'noise_model' must be provided.")

        target_circuit = self if in_place else self.copy()

        try:
            noise_gate_instance = target_circuit._create_noise_channel_instance(noise_model, color)
        except ValueError as e:
            raise ValueError(f"Invalid noise_model or color provided: {e}") from e

        schedule = target_circuit._asap_schedule()
        final_coords = []

        # Coordinate Generation
        if rand_coords is not None:
            try:
                (q_range, col_range, prob) = rand_coords
                q_start, q_end = q_range
                col_start, col_end = col_range
            except (TypeError, ValueError):
                raise ValueError("`rand_coords` must be a list/tuple of the form "
                                 "((q_start, q_end), (col_start, col_end), prob).")

            if not (0 <= q_start <= q_end < target_circuit.n):
                raise ValueError(f"Qudit range ({q_start}, {q_end}) is out of bounds for {target_circuit.n} qudits.")
            if not (0 <= col_start <= col_end):
                raise ValueError("Column range must be non-negative and ordered.")
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"placement_prob must be between 0 and 1, got {prob}.")

            rng = xp.random.default_rng(seed)
            for q in range(q_start, q_end + 1):
                for c in range(col_start, col_end + 1):
                    if rng.random() < prob:
                        final_coords.append((q, c))
        else: # coords is not None
            if not isinstance(coords, list) or not all(isinstance(c, tuple) and len(c) == 2 for c in coords):
                 raise TypeError("`coords` must be a list of (qudit, column) tuples.")
            for q, c in coords:
                if not isinstance(q, int) or not (0 <= q < target_circuit.n):
                    raise ValueError(f"Invalid qudit index {q} in coordinates.")
                if not isinstance(c, int) or c < 0:
                    raise ValueError(f"Invalid column index {c} in coordinates.")
            final_coords = coords

        if not final_coords:
            return None if in_place else target_circuit

        # Group insertions by column index
        insertions_by_col = {}
        for q, c in final_coords:
            if c not in insertions_by_col:
                insertions_by_col[c] = []
            # Append the full operation tuple for the noise gate
            insertions_by_col[c].append((noise_gate_instance, (q,), False))
            
        # Insert new columns into the schedule, iterating in reverse order
        # to prevent insertion from invalidating subsequent column indices.
        for col_idx in sorted(insertions_by_col.keys(), reverse=True):
            noise_ops_for_col = insertions_by_col[col_idx]
            
            # Pad the schedule if inserting beyond its current length
            while len(schedule) <= col_idx:
                schedule.append([])
            
            schedule.insert(col_idx, noise_ops_for_col)
        
        # Rebuild the main _ops list from the modified schedule
        new_ops = [op for col_ops in schedule for op in col_ops]
        target_circuit._ops = new_ops
        
        if not in_place:
            return target_circuit
        return None
    
    
    
    def push_noise(self, end: bool = False, in_place: bool = False) -> QuantumCircuit:
        """
        Move all NoiseChannel placeholders to the start/end of the circuit.

        - Non-noise ops keep their original order.
        - For each qudit q, the noise ops that acted on q are appended at the start/end
        in the same order they originally appeared on q.
        Args:
            end: If True, moves noise channels to the end. If False (default), to the start.
            in_place: If True, modifies the circuit directly. If False, returns a new modified circuit.
        Returns self (in_place=True) or a reordered copy (in_place=False).
        """
        def _is_noise(g):
            return isinstance(g, NoiseChannel) or g.name == "NOISE_CHANNEL"

        ops = self._ops
        non_noise = [op for op in ops if not _is_noise(op[0])]
        noise     = [op for op in ops if _is_noise(op[0])]

        target = self if in_place else self.copy()
        if end:
            target._ops = non_noise + noise
        else:
            target._ops = noise + non_noise
        return target
    
    def realize_noise(self, color: str = "#FF0000", show_identities: bool = False, seed: int | None = None, in_place: bool = True) -> list[tuple] | tuple[QuantumCircuit, list[tuple]]:
        """
        Replaces all NoiseChannel gate placeholders in the circuit with specific, randomly
        sampled Weyl operators.

        For each NoiseChannel gate, a Weyl operator W_ab is sampled from its probability
        distribution in the `noise_registry`.
        - If an error occurs (a,b != 0,0), the NoiseChannel gate is replaced by a
          new, colored `W_{a,b}` gate.
        - If no error occurs (a,b == 0,0), the NoiseChannel gate is deleted from the circuit.

        After execution, the `noise_registry` will be empty as all noise channels
        have been converted into concrete gates.

        Args:
            color: The color for the new, realized Weyl error gates.
            show_identities: If True, explicitly show identity outcomes as `I`
                             gates in the circuit instead of deleting them. 
                             Identities are returned in applied_errors. Defaults to False.
            seed: An optional RNG seed for reproducibility.
            in_place: If True (default), modifies the circuit directly. If False,
                      returns a new modified circuit, preserving the original.

        Returns:
            A list of the error operations that were applied. Each entry is a tuple
            containing: (realized_gate, qudit_index, new_position_in_ops, (a, b)).
            If in_place is False, returns a tuple of (new_circuit, applied_errors_list).
        """
        if in_place:
            target_circuit = self
        else:
            target_circuit = self.copy()
            
        rng = xp.random.default_rng(seed)
        applied_errors = []
        new_ops = []

        # Iterate through the original operations and build a new list
        for gate, qudits, dagger in target_circuit._ops:

            # Check for NoiseChannel instance
            if isinstance(gate, NoiseChannel):
                # Get the noise model from the registry using the gate's model_id
                normalized_probs = target_circuit.noise_registry.get(gate.model_id)
                if normalized_probs is None:
                    continue # Should not happen if circuit is constructed correctly

                choice_index = rng.choice(target_circuit.d * target_circuit.d, p=normalized_probs)
                a_b_pair = xp.unravel_index(choice_index, (target_circuit.d, target_circuit.d))
                a, b = int(a_b_pair[0]), int(a_b_pair[1])

                is_identity = (a == 0 and b == 0)
                if is_identity and not show_identities:
                    pass # Skip adding a gate entirely
                else:
                    realized_gate = self._W(a, b, color=color)
                    new_ops.append((realized_gate, qudits, False))
                    new_pos = len(new_ops) - 1
                    # Append all realized outcomes, including identities if shown
                    applied_errors.append((realized_gate, qudits[0], new_pos, (a, b)))
            else:
                # This is a regular gate, keep it.
                new_ops.append((gate, qudits, dagger))
        
        # The noise registry is cleared because there are no more NoiseChannel gates
        target_circuit._ops = new_ops
        target_circuit.noise_registry.clear()
        target_circuit._noise_model_cache.clear()
        target_circuit._next_noise_model_id = 0

        if in_place:
            return applied_errors
        else:
            return target_circuit, applied_errors
        
    def clear_noise(self) -> None:
        """
        Removes all noise-related gates from the circuit in-place.

        This method iterates through the circuit and removes any gate that is
        a `NoiseChannel` placeholder. Any realized error gates (W_a,b)
        are left untouched. It also clears the circuit's noise registry.
        """
        # Build a new list containing only non-noise gates
        new_ops = []
        for gate, qudits, dagger in self._ops:
            if not isinstance(gate, NoiseChannel) and not gate.name == "NOISE_CHANNEL":
                new_ops.append((gate, qudits, dagger))

        # Replace the old operations list with the new, filtered list
        self._ops = new_ops

        # Clear all registry data as it's no longer relevant
        self.noise_registry.clear()
        self._noise_model_cache.clear()
        self._next_noise_model_id = 0
    # -------------------------------------------------------------------------------------------------

    # not used; the noise modelling functionality above is more general and flexible.
    def add_noise(
    self,
    mode: str = "prob",             # 'count' or 'prob'
    num_errors: int | None = None,
    error_prob: float | None = 0.5,
    error_gates: dict[str, float] | None = None,  # {"X":0.3,"Zdag":0.7,...}
    error_color: str = "#FF0000",
    seed: int | None = None
) -> tuple[QuantumCircuit, list[tuple[Gate, int, bool, int]]]:
        """
        Insert random single-qudit errors into a *new* circuit.

        Now accepts `error_gates` as a dict of one-qudit gate names -> probability.
        Valid names (case-insensitive): 'I','X','Y','Z' and their '...dag' variants.
        Examples: {"X":1, "Zdag":1}, {"I":0.2,"X":0.4,"Y":0.4}, etc.

        Behavior:
        - Probabilities are validated (no negatives) and normalized.
        - All-zero or empty dict -> ValueError.
        - Only one-qudit errors are allowed (I/X/Y/Z and daggers).
        - Two modes:
            * 'count': insert exactly `num_errors` errors at random positions.
            * 'prob' : after each gate, insert an error with probability `error_prob`.
        Returns:
        (noisy_circuit, applied_errors_list) where each error tuple is:
        (colored_gate, qudit_index, apply_dagger, position)
        """

        rng = xp.random.default_rng(seed)

        # ---- validate gates dict and build normalized sampling arrays ----
        if error_gates is None or len(error_gates) == 0:
            raise ValueError("error_gates must be a non-empty dict of probabilities.")

        def _parse_oneq(name: str) -> tuple[str, bool]:
            k = name.strip()
            is_dag = k.endswith("dag") or k.endswith("DAG")
            base = k[:-3] if is_dag else k
            base = base.upper()
            return base, is_dag

        base_map = {"I": I_gate, "X": X_gate, "Y": Y_gate, "Z": Z_gate}

        items: list[tuple[Gate, bool, float]] = []
        total = 0.0
        for k, p in error_gates.items():
            try:
                p = float(p)
            except Exception:
                raise ValueError(f"Probability for '{k}' is not a number: {p!r}")
            if p < 0:
                raise ValueError(f"Negative probability for '{k}': {p}")
            if p == 0:
                continue
            base, is_dag = _parse_oneq(k)
            if base not in base_map:
                raise ValueError(f"Unknown error gate '{k}'. Only I/X/Y/Z and their daggers are allowed.")
            g = base_map[base]
            if g.n_qudits != 1:
                raise ValueError(f"Gate '{k}' is not single-qudit.")
            items.append((g, is_dag, p))
            total += p

        if not items or total == 0.0:
            raise ValueError("All gate probabilities are zero (or gateset empty).")

        probs = xp.array([p for (_, _, p) in items], dtype=xp.float64)
        probs /= probs.sum()
        gates_arr = [g for (g, _, _) in items]
        dag_arr   = [d for (_, d, _) in items]

        # helper: create a colored copy of the chosen base gate
        def _make_colored_error_gate() -> tuple[Gate, bool]:
            idx = int(rng.choice(len(gates_arr), p=probs))
            base_gate = gates_arr[idx]
            apply_dagger = bool(dag_arr[idx])
            colored = Gate(
                name=base_gate.name,
                n_qudits=base_gate.n_qudits,
                matrix_fn=base_gate.matrix_fn,
                dagger_matrix_fn=base_gate.dagger_matrix_fn,
                apply_fn=base_gate._apply_fn,
                color=error_color,
            )
            return colored, apply_dagger

        noisy_qc = QuantumCircuit(self.n, self.d, f"{self.name if self.name else 'QuantumCircuit'}_noisy")
        applied_errors: list[tuple[Gate, int, bool, int]] = []

        if mode == "count":
            if num_errors is None or num_errors < 0:
                raise ValueError("A non-negative num_errors must be specified for 'count' mode.")
            n_orig_gates = len(self._ops)
            total_slots = n_orig_gates + num_errors
            if num_errors > total_slots:
                raise ValueError("num_errors exceeds total available slots.")
            error_positions = xp.sort(rng.choice(total_slots, size=num_errors, replace=False))

            orig_idx = 0
            err_idx = 0
            for i in range(total_slots):
                if err_idx < num_errors and i == error_positions[err_idx]:
                    e_gate, e_dag = _make_colored_error_gate()
                    e_qudit = int(rng.integers(0, self.n))
                    noisy_qc.append(e_gate, (e_qudit,), dagger=e_dag)
                    applied_errors.append((e_gate, e_qudit, e_dag, i))
                    err_idx += 1
                else:
                    gate, qudits, dagger = self._ops[orig_idx]
                    noisy_qc.append(gate, qudits, dagger)
                    orig_idx += 1

        elif mode == "prob":
            if error_prob is None or not (0.0 <= float(error_prob) <= 1.0):
                raise ValueError("An error_prob between 0 and 1 must be specified for 'prob' mode.")
            p = float(error_prob)
            pos = 0
            for gate, qudits, dagger in self._ops:
                noisy_qc.append(gate, qudits, dagger)
                pos += 1
                if float(rng.random()) < p:
                    e_gate, e_dag = _make_colored_error_gate()
                    e_qudit = int(rng.integers(0, self.n))
                    noisy_qc.append(e_gate, (e_qudit,), dagger=e_dag)
                    applied_errors.append((e_gate, e_qudit, e_dag, pos))
                    pos += 1
        else:
            raise ValueError("mode must be either 'count' or 'prob'.")

        return noisy_qc, applied_errors