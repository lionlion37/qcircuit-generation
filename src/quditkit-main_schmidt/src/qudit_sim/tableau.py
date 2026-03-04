from __future__ import annotations
import numpy as np
from functools import lru_cache
import os
import pathlib
import math
import itertools
import random
import sys
import json
import typing
from dataclasses import dataclass
from typing import Tuple, Optional
from . import backend
from .backend import xp
from . import gates
from .gate_class import Gate, NoiseChannel
from .circuit import QuantumCircuit

_ON_NUMPY = (xp.__name__ == "numpy")

try:
    # If this import succeeds, we have the vectorized Numba kernels.
    from .tableau_numba_fast import (
    nb_apply_circuit,
    nb_apply_gate,
    nb_apply_circuit_parallel,
    pack_circuit_from_qc,
    ID_TO_NAME,
)
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


def choose_signed_dtype_for(max_dim: int):
    """
    Return the smallest *signed* integer dtype that can hold
    any intermediate up to ±2*(max_dim-1)^2 exactly.
    """
    max_need = 2 * (max_dim - 1)**2
    if max_need < 2**7:
        return xp.int8
    elif max_need < 2**15:
        return xp.int16
    elif max_need < 2**31:
        return xp.int32
    else:
        return xp.int64

def _scalar_to_int(x):
    """Convert a scalar from xp (numpy/cupy) to Python int safely."""
    try:
        return int(x.item())
    except Exception:
        return int(x)  # fallback for plain Python ints


class Tableau:
    """
    Stabilizer tableau for n qudits of dimension d storing only tau exponents (mod 2d).
    X and Z blocks are stored separately, entries in [0, d-1].
    tau_exp is the exponent of tau (with tau^2 = omega), stored in [0, 2d-1].
    """
    def __init__(
        self,
        n: int,
        d: int,
        *,
        full: bool = True,
        X: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        tau_exp: Optional[np.ndarray] = None,
    ):
        self._n = n
        self._d = d
        self._full = full
        self._mod_d = d
        self._mod_2d = 2 * d

        # choose appropriate dtypes
        self._xz_dtype = choose_signed_dtype_for(self._d)
        self._tau_dtype = choose_signed_dtype_for(self._d)

        rows = 2 * n if full else n

        # Allocate defaults
        if X is not None:
            if X.shape != (rows, n):
                raise ValueError(f"Provided X has shape {X.shape}, expected {(rows, n)}")
            self._X = X.astype(self._xz_dtype) % self._mod_d
        else:
            self._X = xp.zeros((rows, n), dtype=self._xz_dtype)

        if Z is not None:
            if Z.shape != (rows, n):
                raise ValueError(f"Provided Z has shape {Z.shape}, expected {(rows, n)}")
            self._Z = Z.astype(self._xz_dtype) % self._mod_d
        else:
            self._Z = xp.zeros((rows, n), dtype=self._xz_dtype)

        if tau_exp is not None:
            if tau_exp.shape != (rows,):
                raise ValueError(f"Provided tau_exp has shape {tau_exp.shape}, expected {(rows,)}")
            self._tau_exp = tau_exp.astype(self._tau_dtype) % self._mod_2d
        else:
            self._tau_exp = xp.zeros((rows,), dtype=self._tau_dtype)

        # default initializations if not overridden
        if X is None and Z is None and tau_exp is None:
            if full:
                # full input: first n rows have X_i, next n rows have Z_i
                eye = xp.eye(n, dtype=self._xz_dtype)
                self._X[:n, :] = eye  # X stabilizers
                self._Z[n:2 * n, :] = eye  # Z stabilizers
            else:
                # all-|0> state: Z_i stabilizers
                eye = xp.eye(n, dtype=self._xz_dtype)
                self._Z[:n, :] = eye    
        
    # --- Properties for read-only access ---
    @property
    def n(self):
        return self._n

    @property
    def d(self):
        return self._d

    @property
    def full(self):
        return self._full

    @property
    def mod_d(self):
        return self._mod_d

    @property
    def mod_2d(self):
        return self._mod_2d

    @property
    def X(self):
        return self._X

    @property
    def Z(self):
        return self._Z

    @property
    def tau_exp(self):
        return self._tau_exp

    @classmethod
    def zero_state(cls, n: int, d: int) -> Tableau:
        """Create a tableau representing the all-|0> state."""
        return cls(n, d, full=False)

    @classmethod
    def full_state(cls, n: int, d: int) -> Tableau:
        """Return the full input tableau."""
        return cls(n, d, full=True)

    @classmethod
    def from_rows(cls, n: int, d: int, X: np.ndarray, Z: np.ndarray, tau_exp: np.ndarray, full: bool=False) -> Tableau:
        """Construct a custom tableau from raw stabilizer rows."""
        return cls(n, d, full=full, X=X, Z=Z, tau_exp=tau_exp)

    def copy(self) -> Tableau:
        t = Tableau(self._n, self._d, full=self._full)
        t._X = self._X.copy()
        t._Z = self._Z.copy()
        t._tau_exp = self._tau_exp.copy()
        return t

    def _normalize(self):
        # internal in-place reduction
        self._X %= self._mod_d
        self._Z %= self._mod_d
        self._tau_exp %= self._mod_2d

    def normalize(self):
        self._normalize()

    def change_dimension(self, new_d: int, in_place: bool = False):
        """
        Change the local qudit dimension d -> new_d by remapping tableau entries.

        This reduces the tableau blocks modulo the new bases:
        X := X mod new_d
        Z := Z mod new_d
        tau := tau mod (2*new_d)

        Parameters
        ----------
        new_d : int
            New local dimension (>= 2).
        in_place : bool, optional
            If True, mutate this tableau. If False, return a new tableau with the same
            shape and 'full' setting, but with entries reduced modulo the new d.

        Returns
        -------
        Tableau
            The updated tableau (self if in_place=True; a new instance otherwise).
        """
        if not isinstance(new_d, int) or new_d < 2:
            raise ValueError(f"new_d must be an integer >= 2, got {new_d!r}")

        rows, n = self._X.shape
        if in_place:
            self._X %= new_d
            self._Z %= new_d
            self._tau_exp %= (2 * new_d)
            self._d = int(new_d)
            return self

        # create same-shape tableau with the new dimension
        tgt = type(self)(n, int(new_d), full=self._full)
        tgt._X[:rows, :n] = self._X % new_d
        tgt._Z[:rows, :n] = self._Z % new_d
        tgt._tau_exp[:rows] = self._tau_exp % (2 * new_d)
        return tgt

    def pretty(self):
        self._normalize()
        lines = []
        rows = self._X.shape[0]
        for i in range(rows):
            x_row = self._X[i]
            z_row = self._Z[i]
            x_str = ",".join(str(_scalar_to_int(v)) for v in x_row)
            z_str = ",".join(str(_scalar_to_int(v)) for v in z_row)
            tau = _scalar_to_int(self._tau_exp[i])
            lines.append(f"Row {i}: X=[{x_str}] Z=[{z_str}] τ^{tau}")
        return "\n".join(lines)
    
    def print_stabilizers(self, show_omegas: bool = False) -> None:
        """
        Pretty, column-aligned table of all stabilizer generators.

        Layout
        ------
            phase | 0 | 1 | … | n-1
            ─────────────────────────
            ωᵖ τ  | P₀ | P₁ | … | P_{n-1}

        Parameters
        ----------
        show_omegas : bool, optional
            If True, display both the ω-power and τ bit in the phase column;
            if False, omit ω entirely and only display τ (or 1). Default: False.
        """
        self._normalize()
        SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
        sup = lambda n: str(n).translate(SUP)

        def single(a: int, b: int, d: int) -> str:
            a %= d; b %= d
            if a == b == 0:
                return "I"
            out = []
            if a:
                out.append("X" if a == 1 else f"X{sup(a)}")
            if b:
                out.append("Z" if b == 1 else f"Z{sup(b)}")
            return "".join(out)

        d = self._d
        n = self._X.shape[1]
        rows = self._X.shape[0]
        labels = ([f"d{i}" for i in range(n)] + [f"s{i}" for i in range(n)]) if self._full else [str(i) for i in range(rows)]

        entries = []
        for r in range(rows):
            tau_val = int(self._tau_exp[r] % (2 * d))
            omega_power = (tau_val // 2) % d
            has_tau = tau_val & 1
            if show_omegas:
                phase = (("ω" if omega_power == 1 else "ω" + sup(omega_power) if omega_power else "") + (" τ" if has_tau else "")).strip() or "1"
            else:
                phase = "τ" + sup(tau_val) if tau_val else "1"
            factors = [single(int(self._X[r, q]), int(self._Z[r, q]), d) for q in range(n)]
            entries.append([labels[r], phase, *factors])

        widths = [max(len(col) for col in col_vals) for col_vals in zip(*entries, ["#", "phase", *map(str, range(n))])]
        header = " | ".join(txt.ljust(widths[i]) for i, txt in enumerate(["#", "phase", *map(str, range(n))]))
        print(header)
        top = "─" * len(header)
        mid = "-" * len(header)
        print(top)
        for r, row in enumerate(entries):
            print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
            if self._full and r == n - 1:
                print(mid)

    def __str__(self):
        self._normalize()
        rows = self._X.shape[0]
        n = self._n

        x_strs = [[str(_scalar_to_int(v)) for v in self._X[i]] for i in range(rows)]
        z_strs = [[str(_scalar_to_int(v)) for v in self._Z[i]] for i in range(rows)]
        tau_strs = [str(_scalar_to_int(self._tau_exp[i])) for i in range(rows)]

        x_widths = [max(len(x_strs[i][j]) for i in range(rows)) for j in range(n)]
        z_widths = [max(len(z_strs[i][j]) for i in range(rows)) for j in range(n)]
        tau_width = max(len(s) for s in tau_strs + ["τ"])

        if self._full:
            labels = [f"d{i}" for i in range(n)] + [f"s{i}" for i in range(n)]
        else:
            labels = [str(i) for i in range(rows)]
        row_w = max(len("#"), max(len(s) for s in labels))

        def rpad(s, w): return s.ljust(w)
        def pad(s, w):  return s.rjust(w)

        x_headers = [f"x{j}" for j in range(n)]
        z_headers = [f"z{j}" for j in range(n)]
        for j in range(n):
            x_widths[j] = max(x_widths[j], len(x_headers[j]))
            z_widths[j] = max(z_widths[j], len(z_headers[j]))
        row_label = rpad("#", row_w)
        x_block_hdr = " ".join(pad(x_headers[j], x_widths[j]) for j in range(n))
        z_block_hdr = " ".join(pad(z_headers[j], z_widths[j]) for j in range(n))
        tau_hdr = pad("τ", tau_width)

        header = f"{row_label} | {x_block_hdr} | {z_block_hdr} | {tau_hdr}"
        sep = "─" * len(header)
        sep_full = "-" * len(header)

        lines = [header, sep]
        for i in range(rows):
            rn = rpad(labels[i], row_w)
            x_block = " ".join(pad(x_strs[i][j], x_widths[j]) for j in range(n))
            z_block = " ".join(pad(z_strs[i][j], z_widths[j]) for j in range(n))
            tau_block = pad(tau_strs[i], tau_width)
            line = f"{rn} | {x_block} | {z_block} | {tau_block}"
            lines.append(line)
            if self._full and i == n - 1:
                lines.append(sep_full)
        return "\n".join(lines)

    def __repr__(self):
        return (f"Tableau(n={self._n}, d={self._d}, full={self._full}, "
                f"rows={self._X.shape[0]})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tableau):
            return False
        if self._n != other._n or self._d != other._d or self._full != other._full:
            return False
        a = self.copy()
        b = other.copy()
        a._normalize()
        b._normalize()
        if not xp.array_equal(a._X, b._X):
            return False
        if not xp.array_equal(a._Z, b._Z):
            return False
        if not xp.array_equal(a._tau_exp, b._tau_exp):
            return False
        return True

    def get_row(self, idx: int) -> np.ndarray:
        """
        Return a single tableau row as [x0,...,x_{n-1}, z0,...,z_{n-1}, tau].

        Parameters
        ----------
        idx : int
            Row index. Supports negative indices (like Python lists), where -1
            refers to the last row.

        Returns
        -------
        np.ndarray
            1D array of length 2n+1 with the row contents.
        """
        if not isinstance(idx, int):
            raise TypeError(f"Row index must be int, got {type(idx).__name__}")

        n_rows = self.X.shape[0]
        if idx < 0:
            idx += n_rows
        if idx < 0 or idx >= n_rows:
            raise IndexError(f"Row index {idx} out of range for {n_rows} rows")

        return xp.concatenate([self.X[idx], self.Z[idx], [self.tau_exp[idx]]])

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Bracket indexing for tableau rows.

        Equivalent to ``get_row(idx)``. Supports negative indices.

        Parameters
        ----------
        idx : int
            Row index. Negative values count from the end.

        Returns
        -------
        np.ndarray
            1D array of length 2n+1 containing the row values
            in the order [x0,...,x_{n-1}, z0,...,z_{n-1}, tau].

        Examples
        --------
        >>> tab[1]
        array([0, 3, 2, 5, 9])
        """
        return self.get_row(idx)
    
    def apply_gate(self, gate: Gate, targets: Tuple[int, ...], dagger: bool = False) -> None:
        """
        Apply a gate to this tableau."""
        if isinstance(targets, int):
            targets = (targets,)
        name = getattr(gate, "name", None)
        if name is None:
            raise ValueError("Gate missing name attribute for dispatch")
        name_upper = name.upper()
        if name_upper == "I":
            return
        elif name == "X":
            self._apply_X(targets[0], dagger)
        elif name == "Y":
            self._apply_Y(targets[0], dagger)
        elif name == "Z":
            self._apply_Z(targets[0], dagger)
        elif name_upper == "H":
            self._apply_H(targets[0], dagger=dagger)
        elif name_upper == "S":
            self._apply_S(targets[0], dagger=dagger)
        elif name_upper in ("CNOT", "CX"):
            self._apply_CNOT(targets[0], targets[1], dagger=dagger)
        elif name_upper == "CZ":
            self._apply_CZ(targets[0], targets[1], dagger=dagger)
        elif name_upper == "SWAP":
            self._apply_SWAP(targets[0], targets[1], dagger=dagger)
        elif gate.a is not None and gate.b is not None:
            self._apply_weyl(gate.a, gate.b, targets[0], dagger)
        else:
            pass # raise NotImplementedError(f"Gate {name} not implemented in tableau")

    def _apply_circuit_python(self, qc: QuantumCircuit, *, print_tableau: bool = False) -> Tableau:
        """
        Apply a QuantumCircuit to this tableau.
        Args:
            qc (QuantumCircuit): The circuit to apply.
            print_tableau (bool): If True, print the tableau after each gate.
        """
        if print_tableau:
            print("Initial tableau:")
            print(self)

        for gate, targets, dagger in qc:
            if print_tableau:
                name = getattr(gate, "name", "").upper()
                suffix = "†" if dagger else ""
                if gate.a == 0 and gate.b == 0:
                    suffix = ""
                suffix_q = "s" if len(targets) > 1 else ""
                print(f"\n--- Applying {name}{suffix} on qudit{suffix_q} {', '.join(map(str, targets))} ---")
            self.apply_gate(gate, targets, dagger=dagger)
            if print_tableau:
                print(self)
        return self

    def _apply_circuit_numba(self, qc, *, parallel: bool = False, print_tableau: bool = False) -> Tableau:
        """
        Apply `qc` using the Numba-accelerated kernels.
        Important (but optional): before using this method, precompile the Numba kernel
        by calling `warmup_numba(d)` for each individual dimension `d`
        (from `src/qudit_sim/utils.py`).
        Speeds up the execution 4-5x for n < 150,
        2-3x for 150 <= n < 500,
        and 1.5-2x for n >= 500.
        Args:
            qc (QuantumCircuit): The circuit to apply.
            parallel (bool): If True, use the parallel Numba kernel.
            print_tableau (bool): If True, print the tableau after each gate.
        Notes:
            Currently, parallel provides no speedup (or even makes things slower).
        """
        if not (_HAS_NUMBA and _ON_NUMPY):
            return self._apply_circuit_python(qc, print_tableau=print_tableau)

        G = pack_circuit_from_qc(qc)
        if G.size == 0:
            return self

        def _wb():
            if X is not self._X:
                self._X[...] = X
            if Z is not self._Z:
                self._Z[...] = Z
            if T is not self._tau_exp:
                self._tau_exp[...] = T

        if print_tableau:
            print("Initial tableau:")
            print(self)

            # operate on contiguous views; write back before each print
            X = np.ascontiguousarray(self._X)
            Z = np.ascontiguousarray(self._Z)
            T = np.ascontiguousarray(self._tau_exp)

            for k in range(G.shape[0]):
                gid = int(G[k, 0]); q1 = int(G[k, 1]); q2 = int(G[k, 2]); dag = int(G[k, 3])
                name = ID_TO_NAME.get(gid, "?")
                if name == "W_ab":
                    name = f"W_{abs(q2)},{abs(dag)}"
                    suffix = "†" if q2 < 0 and dag < 0 else ""
                    targets = f"{q1}"
                    suffix_q = ""
                else:
                    suffix = "†" if dag else ""
                    targets = f"{q1}" if q2 < 0 else f"{q1}, {q2}"
                    suffix_q = 's' if q2 >= 0 else ""
                print(f"\n--- Applying {name}{suffix} on qudit{suffix_q} {targets} ---")

                nb_apply_gate(X, Z, T, gid, q1, q2, dag, self._mod_d, self._mod_2d)
                print(self)
            return self

        X = np.ascontiguousarray(self._X)
        Z = np.ascontiguousarray(self._Z)
        T = np.ascontiguousarray(self._tau_exp)
        if not parallel:
            nb_apply_circuit(X, Z, T, G, self._mod_d, self._mod_2d)
        else:
            nb_apply_circuit_parallel(X, Z, T, G, self._mod_d, self._mod_2d)
        return self

    def apply_circuit(
        self,
        qc,
        *,
        engine: str | None = None,   # None | "numba" | "python"
        parallel: bool = False,      # only relevant for engine="numba"
        print_tableau: bool = False,
    ) -> Tableau:
        """
        Apply a QuantumCircuit to this tableau.
        Important:
            parallel=True currently has no speedup (or even makes things slower).
            When using engine='numba', before using this method, precompile 
            the Numba kernel by calling `warmup_numba(d)` for each individual 
            dimension `d` (from `src/qudit_sim/utils.py`).
        
        Args:
            engine : None | "numba" | "python"
                None (default): auto-select fastest (Numba on NumPy; else Python).
                "numba": force Numba engine (error if unavailable).
                "python": force reference Python engine.
            parallel : bool
                Use the auto-parallel Numba kernel (ignored by Python engine).
                currently has no speedup (or even makes things slower).
            print_tableau : bool
                Print the tableau after each gate.

        Returns:
            self
        """
        if engine is None:
            # auto
            if _HAS_NUMBA and _ON_NUMPY:
                # printing uses per-gate kernel either way; batch uses parallel flag
                return self._apply_circuit_numba(qc, parallel=parallel, print_tableau=print_tableau)
            # fallback
            return self._apply_circuit_python(qc, print_tableau=print_tableau)

        if engine == "python":
            return self._apply_circuit_python(qc, print_tableau=print_tableau)

        if engine == "numba":
            if not (_HAS_NUMBA and _ON_NUMPY):
                raise RuntimeError("Numba engine requested but unavailable (needs NumPy backend and Numba kernels).")
            return self._apply_circuit_numba(qc, parallel=parallel, print_tableau=print_tableau)

        raise ValueError("engine must be None, 'numba', or 'python'.")
    
    def _statevector_from_blocks(
        self,
        X_block,
        Z_block,
        tau_exp,
        d: int,
        *,
        normalize: bool = True,
    ):
        """
        Build the joint +1 eigenstate for a set of stabilizers specified by
        (X_block, Z_block, tau_exp) on n_qudits = X_block.shape[1].
        Returns an xp.ndarray of length d**n_qudits.
        """
        n_rows, n_qudits = X_block.shape
        if Z_block.shape != (n_rows, n_qudits):
            raise ValueError("Shape mismatch between X_block and Z_block.")
        if tau_exp.shape[0] != n_rows:
            raise ValueError("tau_exp length must match number of rows in X/Z blocks.")

        # Build single-qudit X and Z matrices
        X_mat = gates.X(d).astype(backend.COMPLEX_DTYPE)
        Z_mat = gates.Z(d).astype(backend.COMPLEX_DTYPE)

        # τ² = ω
        tau = xp.exp(1j * xp.pi * (d**2 + 1/d))

        # Assemble each stabilizer operator
        mats = []
        for r in range(n_rows):
            factors = []
            for q in range(n_qudits):
                x = int(X_block[r, q])
                z = int(Z_block[r, q])
                
                # operator = X^x · Z^z
                op_q = xp.linalg.matrix_power(X_mat, x) @ xp.linalg.matrix_power(Z_mat, z)
                factors.append(op_q)

            # Tensor them together to form the full d**n_op x d**n_op operator
            full_op = factors[0]
            for op_q in factors[1:]:
                full_op = xp.kron(full_op, op_q)

            # Apply the global τ-phase from the tableau
            mats.append((tau ** int(tau_exp[r])) * full_op)

        # Find the joint +1 eigenvector of the sum of stabilizer operators
        M = sum(mats)
        w, v = xp.linalg.eig(M)
        inds = xp.where(xp.isclose(w, len(mats)))[0]
        if inds.size == 0:
            raise ValueError("No +1 eigenvector found for this stabilizer set.")
        vec = v[:, int(inds[0])]
        if normalize:
            vec = vec / xp.linalg.norm(vec)
            vec = vec.astype(backend.COMPLEX_DTYPE, copy=False)
        return vec
    
    def to_statevector(self, normalize: bool = True):
        """
        Return the n-qudit statevector (length d**n) corresponding to this tableau.

        Works for both stabilizer-only and full (destabilizers + stabilizers) tableaus:
        - If full, uses only the stabilizer half (rows n..2n-1).
        - If stabilizer-only, uses rows 0..n-1.

        Raises
        ------
        ValueError
            If the tableau does not define a unique +1 eigenstate (e.g., mixed/invalid).
        """
        self._normalize()
        d, n = self._d, self._n

        if self._full:
            # take stabilizers only
            Xb = self._X[n:, :]
            Zb = self._Z[n:, :]
            ta = self._tau_exp[n:]
        else:
            Xb = self._X
            Zb = self._Z
            ta = self._tau_exp

        return self._statevector_from_blocks(Xb, Zb, ta, d, normalize=normalize)

    def to_statevector_universal(self, normalize: bool = True):
        """
        Return the 2n-qudit universal (Choi-like) statevector, length d**(2n).

        Requires the internal X/Z to have shape (2n, n). Raises if incompatible.

        Raises
        ------
        ValueError
            If shapes are incompatible or if no +1 eigenstate is found.
        """
        self._normalize()
        d, n = self._d, self._n

        # Part A: The Evolved Half (from the 2n x n universal tableau)
        X_evolved = self._X
        Z_evolved = self._Z
        tau_exp   = self._tau_exp

        # shape checks
        if X_evolved.shape != (2*n, n) or Z_evolved.shape != (2*n, n) or tau_exp.shape[0] != 2*n:
            raise ValueError(
                "Tableau shape incompatible with universal/Choi construction: "
                f"X/Z must be (2n, n); got X={X_evolved.shape}, Z={Z_evolved.shape}."
            )

        # Part B: The Unchanged Half (initial basis operators on the second register)
        X_unchanged = xp.vstack([
            xp.eye(n, dtype=self._xz_dtype),
            xp.zeros((n, n), dtype=self._xz_dtype),
        ])
        Z_unchanged = xp.vstack([
            xp.zeros((n, n), dtype=self._xz_dtype),
            -xp.eye(n, dtype=self._xz_dtype),  # Z† exponents
        ])

        # Combine them to form the 2n x 2n tableau for the Choi state stabilizers
        X_block = xp.hstack([X_evolved, X_unchanged])
        Z_block = xp.hstack([Z_evolved, Z_unchanged])

        return self._statevector_from_blocks(X_block, Z_block, tau_exp, d, normalize=normalize)

    @classmethod
    def random_tableau(cls, n: int, d: int, *, full: bool = False, seed: int | None = None) -> Tableau:   
        """
        Return a tableau with random stabilizers:
        - X and Z entries uniform in [0, d-1]
        - τ exponents uniform in [0, 2d-1]

        This produces a custom tableau:
        full=False ⇒ n rows, full=True ⇒ 2n rows.
        Optional `seed` for reproducibility.
        """
        if seed is not None:
            xp.random.seed(seed)
        rows = n if not full else 2 * n
        xz_dtype = choose_signed_dtype_for(d)
        tau_dtype = choose_signed_dtype_for(d)

        X = xp.random.randint(0, d, size=(rows, n), dtype=xz_dtype)
        Z = xp.random.randint(0, d, size=(rows, n), dtype=xz_dtype)
        tau_exp = xp.random.randint(0, 2 * d, size=(rows,), dtype=tau_dtype)

        return cls(n, d, full=full, X=X, Z=Z, tau_exp=tau_exp)

    # -- Update Rules --
    def _apply_I(self, j: int, dagger: bool = False) -> None:
        """
        Identity on qudit j: no change.

        X / Z tableau entries unchanged.
        τ-exponent unchanged.
        """
        return

    def _apply_X(self, j: int, dagger: bool = False) -> None:
        """
        **X** (or **X†**) on qudit *j*.

        Update rules
        ------------
            τ  ←  τ  -  dir · 2 · Z_j           (mod 2d)

        No change to X / Z tableau entries.
        """
        dir_ = 1 if not dagger else -1
        zj   = self._Z[:, j]
        self._tau_exp = (self._tau_exp - dir_ * 2 * zj) % self._mod_2d
    
    def _apply_Y(self, j: int, dagger: bool = False) -> None:
        """
        Pauli **Y = τ·X·Z** (and **Y†**) on qudit *j*.

        Conventions
        -----------
        dir = +1  →  Y          (conjugation by Y)
        dir = -1  →  Y†         (conjugation by Y†)

        Update rules
        ------------
            Let  x = X_j ,  z = Z_j .

            τ  ←  τ  +  dir · 2·(x - z)        (mod 2d)

        No change to X / Z tableau entries.
        """
        dir_ = 1 if not dagger else -1
        xj   = self._X[:, j]
        zj   = self._Z[:, j]
        self._tau_exp = (self._tau_exp + dir_ * 2 * (xj - zj)) % self._mod_2d
    
    def _apply_Z(self, j: int, dagger: bool = False) -> None:
        """
        **Z** (or **Z†**) on qudit *j*.

        Update rules
        ------------
            τ  ←  τ  +  dir · 2 · X_j           (mod 2d)

        No change to X / Z tableau entries.
        """
        dir_ = 1 if not dagger else -1
        xj   = self._X[:, j]
        self._tau_exp = (self._tau_exp + dir_ * 2 * xj) % self._mod_2d
            
    def _apply_H(self, j: int, dagger: bool = False) -> None:
        """
        Hadamard on qudit *j* († if *dagger* is True).

        Conventions
        -----------
        • `dir = +1`  →  **H**  
        • `dir = -1`  →  **H†**

        Update rules
        ------------
        Let *(x_old, z_old)* be the column (X, Z) of qudit *j* **before** the gate.

            X_j  ←  -dir · z_old               (mod d)
            Z_j  ←   dir · x_old               (mod d)

        Phase column (τ-exponents, modulus 2d):

            τ   ←   τ  +  2 · X_j · Z_j        (mod 2d)
        """
        dir_ = 1 if not dagger else -1

        x_old = self._X[:, j].copy()
        z_old = self._Z[:, j].copy()

        self._X[:, j] = (-dir_ * z_old) % self._mod_d
        self._Z[:, j] = ( dir_ * x_old) % self._mod_d

        x_new = self._X[:, j]
        z_new = self._Z[:, j]
        self._tau_exp = (self._tau_exp + 2 * x_new * z_new) % self._mod_2d

    def _apply_S(self, j: int, dagger: bool = False) -> None:
        """
        Phase gate **S** († if *dagger* is True) on qudit *j*.

        Conventions
        -----------
        • `dir = +1`  →  **S**  
        • `dir = -1`  →  **S†**

        Update rules
        ------------
        Let *x = X_j* (before the gate).

        Z_j  ←  Z_j  +  dir · x         (mod d)

        τ   ←  τ  +  dir · Δ            (mod 2d)

            with  Δ = x²                (even d)
                  Δ = x(x - 1)          (odd  d)

        No change to any other tableau entry.
        """
        dir_ = 1 if not dagger else -1

        xj = self._X[:, j]
        self._Z[:, j] = (self._Z[:, j] + dir_ * xj) % self._mod_d

        delta = xj * xj if self._mod_d % 2 == 0 else xj * (xj - 1)
        self._tau_exp = (self._tau_exp + dir_ * delta) % self._mod_2d
    
    def _apply_CNOT(self, c: int, t: int, dagger: bool = False) -> None:
        """
        CNOT/SUM between control *c* and target *t* († if *dagger* is True).

        Conventions
        -----------
        • `dir = +1`  →  **CNOT**  
        • `dir = -1`  →  **CNOT†**

        Update rules
        ------------
            X_t  ←  X_t  +  dir · X_c           (mod d)
            Z_c  ←  Z_c  -  dir · Z_t           (mod d)

        Phase column **unchanged**.
        """
        if c == t:
            return
        
        dir_ = 1 if not dagger else -1

        self._X[:, t] = (self._X[:, t] + dir_ * self._X[:, c]) % self._mod_d
        self._Z[:, c] = (self._Z[:, c] - dir_ * self._Z[:, t]) % self._mod_d

    def _apply_CZ(self, c: int, t: int, dagger: bool = False) -> None:
        """
        CZ (controlled phase) between qudits *c* and *t* († if *dagger* is True).

        Basis action
        ------------
        |c, t⟩  →  ω^{dir · c t} |c, t⟩,  where  ω = exp(2πi / d).

        Conventions
        -----------
        • `dir = +1`  →  **CZ**
        • `dir = -1`  →  **CZ†**

        Update rules
        ------------
            Z_{c}  ←  Z_{c}  +  dir · X_{t}               (mod d)
            Z_{t}  ←  Z_{t}  +  dir · X_{c}               (mod d)
            τ      ←  τ       +  2 · dir · X_{c} X_{t}    (mod 2d)

        `X` columns **unchanged** (only Z-columns and the phase exponent τ update).
        """
        if c == t:
            return
        
        dir_ = -1 if dagger else 1

        self.Z[:, c] = (self.Z[:, c] + dir_ * self.X[:, t]) % self._mod_d
        self.Z[:, t] = (self.Z[:, t] + dir_ * self.X[:, c]) % self._mod_d
        self.tau_exp[:] = (self.tau_exp + 2 * dir_ * (self.X[:, c] * self.X[:, t])) % self._mod_2d

    def _apply_SWAP(self, q1: int, q2: int, dagger: bool = False) -> None:
        """
        SWAP between qudits *q1* and *q2* († ignored — SWAP = SWAP†).

        Basis action
        ------------
        |q1, q2⟩  →  |q2, q1⟩

        Conventions
        -----------
        • SWAP is Hermitian and involutory:  SWAP = SWAP†  and  SWAP² = I  
        • The ``dagger`` flag is ignored.

        Update rules
        ------------
        Exchange the X/Z columns of the two qudits in every stabilizer row:
            X_{⋅, q1} ↔ X_{⋅, q2}
            Z_{⋅, q1} ↔ Z_{⋅, q2}

        Phase column **unchanged** (τ is not modified).  
        No other tableau entries change.
        """
        if q1 == q2:
            return
        
        self._X[:, [q1, q2]] = self._X[:, [q2, q1]]
        self._Z[:, [q1, q2]] = self._Z[:, [q2, q1]]
    
    def _apply_weyl(self, a: int, b: int, j: int, dagger: bool = False) -> None:
        """
        Applies the Weyl operator W_ab = τ^(-ab)X^a*Z^b to qudit j

        Update rule:
        τ  ←  τ  +  (τ_exp  +  2*b*X_j  -  2*a*Z_j)  mod 2d

        The X and Z matrices are unchanged.
        """
        if a == 0 and b == 0:
            return
        
        dir_ = -1 if dagger else 1
        
        self._tau_exp = (self._tau_exp + 2 * dir_ * b * self._X[:, j] - 2 * dir_ * a * self._Z[:, j]) % self._mod_2d
    
    # ===============================================================================================================
    # Measurement (for prime d & full=True)
    # ===============================================================================================================
    def _is_prime(self, d: int) -> bool:
        """Checks if a number is prime."""
        if d < 2:
            return False
        if d in (2, 3):
            return True
        if d % 2 == 0 or d % 3 == 0:
            return False
        i = 5
        while i * i <= d:
            if d % i == 0 or d % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def _coerce_rng(rng=None, seed=None):
        """
        Return a numpy Generator:
        - if rng is provided, use it;
        - else if seed is provided, create a fresh Generator from that seed;
        - else create an unseeded Generator.
        """
        if rng is not None:
            return rng
        return xp.random.default_rng(seed)

    @staticmethod
    @lru_cache(maxsize=None)
    def _mod_inverse(a: int, m: int) -> int:
        """Calculates the modular multiplicative inverse of a under modulo m."""
        return pow(a, -1, m)

    def _product_logic(self, X1, Z1, tau1, X2, Z2, tau2) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Contains the core logic for multiplying two stabilizer rows, g1 * g2.

        This general function takes the raw row components (X, Z, and tau arrays/values)
        as input and returns the components of their product.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, int]
            A tuple containing the (new_X_row, new_Z_row, new_tau_exp).
        """
        # New X and Z parts are the element-wise sum mod d
        new_X = (X1 + X2) % self._mod_d
        new_Z = (Z1 + Z2) % self._mod_d

        # Phase update rule: phase_new = phase_1 + phase_2 + 2 * sum(Z1 * X2)
        phase_update = 2 * xp.sum(Z1 * X2)
        new_tau_exp = (tau1 + tau2 + phase_update) % self._mod_2d
        
        return new_X, new_Z, new_tau_exp

    def _get_row_product(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Calculates the product of two stabilizer rows, g_i * g_j, using their indices.
        
        This is a convenience wrapper around the more general _product_logic function.
        """
        return self._product_logic(
            self.X[i], self.Z[i], self.tau_exp[i],
            self.X[j], self.Z[j], self.tau_exp[j]
        )

    def _power(self, base_X, base_Z, base_tau, m: int):
        """
        Calculates the m-th power of a stabilizer row, S^m, using the
        efficient exponentiation by squaring algorithm.
        """
        # Exponentiation by squaring for stabilizer rows
        res_X, res_Z, res_tau = xp.zeros_like(base_X), xp.zeros_like(base_Z), 0
        
        # S^d = I, so we only need powers up to d.
        # S^0 = I, which is handled by the loop initialization.
        m %= self.d
        if m == 0:
            return res_X, res_Z, res_tau

        temp_X, temp_Z, temp_tau = base_X, base_Z, base_tau
        
        while m > 0:
            if m % 2 == 1: # If exponent is odd, multiply by the current power of S
                res_X, res_Z, res_tau = self._product_logic(res_X, res_Z, res_tau, temp_X, temp_Z, temp_tau)
            # Square the current power of S for the next iteration
            temp_X, temp_Z, temp_tau = self._product_logic(temp_X, temp_Z, temp_tau, temp_X, temp_Z, temp_tau)
            m //= 2
            
        return res_X, res_Z, res_tau

    def _row_add_mult(self, k: int, p: int, m: int):
        """
        Performs the generalized row operation: row_k -> row_k + m * row_p.
        In stabilizer terms, this is equivalent to g_k -> g_k * (g_p)^m.
        """
        if m == 0: return
        
        # Calculate (g_p)^m
        g_p_m_X, g_p_m_Z, g_p_m_tau = self._power(self.X[p], self.Z[p], self.tau_exp[p], m)
        
        # Multiply g_k by the result
        self.X[k], self.Z[k], self.tau_exp[k] = self._product_logic(
            self.X[k], self.Z[k], self.tau_exp[k],
            g_p_m_X, g_p_m_Z, g_p_m_tau
        )

    def measure(self,
                qudits_to_measure: typing.Union[int, list, tuple],
                desired: typing.Optional[typing.Union[int, list, tuple]] = None,
                rng=None, seed=None,
                in_place: bool = False) -> Tuple[Tableau, dict]:
        """
        Measures one or more qudits in the tableau using the CHP algorithm.
        Measurements happen in order of ascending qudit index, regardless of the input order.
        
        This method simulates a projective measurement in the Z-basis for each specified
        qudit. The tableau is updated to reflect the collapsed state after measurement.
        The algorithm is optimized to have a runtime complexity of O(q*n^2*log(d)), where
        q is the number of qudits being measured; O(n^3*log(d)) if measuring all qudits.
        
        Parameters
        ----------
        - qudits_to_measure : int, list, or tuple
            The index or indices of the qudit(s) to measure.
        - desired : int, list, tuple, or None
            If provided, forces the measurement outcome(s) to the specified value(s).
            If None (default), outcomes are random.
            If a list/tuple, its length must match the number and order of qudits being measured.
            If a single int, it can only be used when measuring a single qudit.
            If the outcome on the qudit is deterministic, the forced value is ignored;
            a tuple (det_value, "det") is stored in the outcomes dictionary.
            If the outcome on the qudit is random, a tuple (forced_value, "rand") is stored;
            tableau is updated accordingly to the forced value.
            If desired is out of range [0, d-1], a ValueError is raised.
        - rng: a numpy.random.Generator to draw randomness from, or
        - seed: an int to seed a local Generator for this call only.
        If both None, a fresh unseeded Generator is used.
        - in_place : bool, optional
            If False (default), a copy of the tableau is measured and returned.
            If True, the measurement is performed directly on the current tableau.

        Returns
        -------
        Tuple[Tableau, dict]
            A tuple containing:
            - The Tableau object after measurement and collapse.
            - A dictionary mapping each measured qudit index to its classical outcome.
                If `desired` is provided, the dictionary of outcomes stores values
                as tuples (forced_value, "rand") or (det_value, "det") at each provided qudit.
        """
        if not self._full:
            raise ValueError("Measurement is only supported for full tableaus.")
        if not self._is_prime(self._d):
            raise ValueError(f"Measurement is only supported for prime dimensions, but d={self._d}.")

        rng = self._coerce_rng(rng, seed)
        
        if isinstance(qudits_to_measure, int):
            qudits_to_measure = [qudits_to_measure]
        else:
            qudits_to_measure = list(qudits_to_measure)

        if len(set(qudits_to_measure)) != len(qudits_to_measure):
            raise ValueError("Duplicate qudit indices in qudits_to_measure.")
        
        desired_map: dict[int, int] = {}
        if desired is not None:
            if isinstance(desired, (list, tuple)):
                if len(desired) != len(qudits_to_measure):
                    raise ValueError(
                        "If `desired` is a list/tuple, it must match the number of measured qudits."
                    )
                for q, v in zip(qudits_to_measure, desired):
                    v_int = int(v)
                    if not (0 <= v_int < self._d):
                        raise ValueError(f"Desired outcome {v_int} for qudit {q} not in [0, {self._d - 1}]")
                    desired_map[q] = v_int
            elif isinstance(desired, int):
                if len(qudits_to_measure) != 1:
                    raise ValueError(
                        "Scalar `desired` can only be used when measuring a single qudit."
                    )
                v_int = int(desired)
                if not (0 <= v_int < self._d):
                    raise ValueError(f"Desired outcome {v_int} not in [0, {self._d - 1}]")
                desired_map[qudits_to_measure[0]] = v_int
            else:
                raise TypeError("`desired` must be int, list, tuple, or None.")
        
        qudits_to_measure = sorted(qudits_to_measure)
        
        target_tableau = self if in_place else self.copy()
        n, d, mod_d = target_tableau.n, target_tableau.d, target_tableau.mod_d
        X, Z, tau_exp = target_tableau.X, target_tableau.Z, target_tableau.tau_exp
        
        outcomes = {}

        for i in qudits_to_measure: # Measure qudit i
            if not (0 <= i < n):
                raise ValueError(f"Qudit index {i} is out of bounds for n={n}.")
            
            stab_X = X[n:]
            # Find a stabilizer with an X component on qudit i
            pivot_row_relative_idx = xp.where(stab_X[:, i] != 0)[0]

            if len(pivot_row_relative_idx) > 0:
                # --- Case 1: Random Measurement ---
                p = pivot_row_relative_idx[0]
                p_full_idx = p + n
                
                # Direct Row Reduction
                pivot_val_inv = target_tableau._mod_inverse(_scalar_to_int(X[p_full_idx, i]), d)
                for k in range(2 * n):
                    if k != p_full_idx and X[k, i] != 0:
                        m = (mod_d - _scalar_to_int(X[k, i])) * pivot_val_inv % mod_d
                        target_tableau._row_add_mult(k, p_full_idx, m)
                
                # Normalization
                base_X, base_Z, base_tau = X[p_full_idx].copy(), Z[p_full_idx].copy(), tau_exp[p_full_idx]

                m = target_tableau._mod_inverse(_scalar_to_int(base_X[i]), d)
                normalized_X, normalized_Z, normalized_tau = target_tableau._power(base_X, base_Z, base_tau, m)

                # Assign the final result as the new destabilizer at row p
                X[p], Z[p], tau_exp[p] = normalized_X, normalized_Z, normalized_tau
                
                # Set the new stabilizer at row p_full_idx to be Z_i with a phase for the outcome
                X[p_full_idx, :] = 0
                Z[p_full_idx, :] = 0
                Z[p_full_idx, i] = 1
                
                if desired is None:
                    # Original behavior
                    outcome = int(rng.integers(0, d))
                    outcomes[i] = outcome
                    tau_exp[p_full_idx] = (-2 * outcome) % target_tableau.mod_2d
                else:
                    # Desired behavior
                    forced = desired_map[i]
                    outcomes[i] = (forced, "rand")
                    tau_exp[p_full_idx] = (-2 * forced) % target_tableau.mod_2d
                    
            else:
                # --- Case 2: Deterministic Measurement ---
                destab_X = X[:n]
                temp_X, temp_Z, temp_tau_exp = xp.zeros(n, dtype=target_tableau._xz_dtype), xp.zeros(n, dtype=target_tableau._xz_dtype), 0
                destab_indices_with_xi = xp.where(destab_X[:, i] != 0)[0]
                
                for j in destab_indices_with_xi:
                    stab_to_add_idx = j + n
                    m = _scalar_to_int(X[j, i])
                    
                    # Calculate (S_j)^m
                    powered_stab_X, powered_stab_Z, powered_stab_tau = target_tableau._power(
                        X[stab_to_add_idx], Z[stab_to_add_idx], tau_exp[stab_to_add_idx], m
                    )
                    # Add to the accumulator
                    temp_X, temp_Z, temp_tau_exp = target_tableau._product_logic(
                        temp_X, temp_Z, temp_tau_exp,
                        powered_stab_X, powered_stab_Z, powered_stab_tau
                    )

                outcome = int((-temp_tau_exp // 2) % mod_d)
                
                if desired is None:
                    # Original behavior
                    outcomes[i] = outcome
                else:
                    # Desired behavior (ignored since deterministic)
                    outcomes[i] = (outcome, "det")
                
        return target_tableau, outcomes

    def measure_all(self,
                    desired: typing.Optional[typing.Union[list, tuple]] = None,
                    rng=None, 
                    seed=None, 
                    in_place: bool = False) -> Tuple[Tableau, np.ndarray]:
        """
        Measures all qudits in the tableau by calling the measure method.

        If `desired` is None (default):
        Returns (tableau, outcomes) where `outcomes` is a 1D array of ints of length n,
        corresponding to the measurement outcomes on each qudit in ascending order.
        
        If `desired` is provided (list/tuple of length n, in ascending qudit order),
        it forces the measurement outcome(s) to the specified value(s).
        Returns (tableau, outcomes) where:
        where `outcomes` is a 1D array of tuples (int, str) of length n,
        where each tuple is (forced_value, "rand") or (det_value, "det"),
        depending on whether the outcome was random or deterministic.
        If the measurement outcome on a qudit for which there is a value in desired is deterministic,
        the forced value is ignored, and you get (det_value, "det") in the outcomes array.
        
        If you pass a seed, a single Generator is created
        for the whole shot so each qudit consumes the next random number, i.e. you
        get reproducible but *different* values across qudits.
        
        Returns
        -------
        Tuple[Tableau, np.ndarray]
            A tuple containing:
            - The Tableau object after measurement and collapse.
            - A 1D numpy array of all measurement outcomes.
                If desired is provided, the array contains tuples
                (forced_value, "rand") or (det_value, "det") at each index.
        """
        rng = self._coerce_rng(rng, seed)
        target_tableau, outcomes_dict = self.measure(qudits_to_measure=range(self.n),
                                                     desired=desired,
                                                     rng=rng, 
                                                     in_place=in_place)

        outcomes_array = xp.array([outcomes_dict[i] for i in range(self.n)])

        return target_tableau, outcomes_array
    
    def post_measure(self,
                    qudits_to_measure: typing.Union[int, list[int], tuple[int, ...]],
                    desired: typing.Optional[typing.Union[int, list[int], tuple[int, ...]]] = None,
                    rng: typing.Optional[np.random.Generator] = None,
                    seed: typing.Optional[int] = None,
                    in_place: bool = False) -> tuple[Tableau, dict[int, typing.Any]]:
        """
        Performs projective measurement(s) and returns a reduced tableau.

        This method simulates a Z-basis measurement and then projects the
        tableau onto the remaining unmeasured subspace, returning a new,
        smaller tableau that describes the post-measurement state.

        Algorithm:
        1.  Calls `self.measure()` on a copy to get the collapsed full
            tableau and measurement outcomes. This operation leaves a simple
            `Z_q` stabilizer (in row `n+q`) for each measured qudit `q`.
        2.  For each measured qudit `q` (and its pivot row `p = n+q`):
            a.  Iterates through all *other* stabilizer rows `r` that are
                not being removed.
            b.  If `Z[r, q] != 0`, it zeroes out this entry by
                performing the row operation `S_r <- S_r * (S_p)^m`,
                where `m` is chosen such that the Z-component on qudit `q`
                is eliminated.
            c.  The paired destabilizer `D_r` (in row `r-n`) has its
                phase corrected via `tau[D_r] <- tau[D_r] - m*tau[S_p]`
                to maintain canonical commutation relations.
        3.  Constructs a new tableau by removing the columns for all
            `measured_qudits` and the paired rows (`q` and `n+q`) for
            each measured qudit.

        Args:
            qudits_to_measure: The index or indices of the qudit(s) to
                measure.
            desired: Optional. If provided, forces the measurement
                outcome(s) to the specified value(s). See `measure()`
                docstring for details on behavior.
            rng: A numpy.random.Generator to use for random outcomes.
            seed: An integer to seed a new Generator if `rng` is None.
            in_place: If True, the current tableau is modified to
                become the new reduced tableau. If False (default), a new
                `Tableau` instance is returned, and `self` is unchanged.

        Returns:
            A tuple `(reduced_tableau, outcomes)` where:
            - `reduced_tableau`: The new `Tableau` instance of size
              `2(n-k) x 2(n-k+1)` (or `self` if `in_place=True`), where `k` is
              the number of measured qudits.
            - `outcomes`: A dictionary mapping each measured qudit index
              to its classical outcome, identical to the output from
              `self.measure()`.

        Raises:
            ValueError: If called on a non-full tableau or for a non-prime
                dimension `d` (via the call to `self.measure()`).
        """
        measured_tab, outcomes = self.measure(
            qudits_to_measure=qudits_to_measure,
            desired=desired,
            rng=rng,
            seed=seed,
            in_place=False,
        )

        if isinstance(qudits_to_measure, int):
            measured_qudits = [qudits_to_measure]
        else:
            measured_qudits = sorted(list(qudits_to_measure))

        n, d = measured_tab.n, measured_tab.d
        X, Z, tau = measured_tab.X, measured_tab.Z, measured_tab.tau_exp

        # Z-elimination + destabilizer phase update
        for q in measured_qudits:
            pivot_row = n + q  # stabilizer for Z_q

            for r in range(n, 2 * n):
                if r == pivot_row or (r - n) in measured_qudits:
                    continue  # skip pivot and measured stabilizers
                coeff = int(Z[r, q])
                if coeff != 0:
                    m = (-coeff) % d
                    measured_tab._row_add_mult(r, pivot_row, m)

                    # Update the paired destabilizer's tau-phase
                    paired_destab = r - n
                    tau[paired_destab] = (tau[paired_destab] - m * tau[pivot_row]) % measured_tab.mod_2d

        # Remove measured qudits' columns and paired rows
        keep_qudits = [i for i in range(n) if i not in measured_qudits]
        rows_to_remove = set()
        for q in measured_qudits:
            rows_to_remove.add(q) # paired destabilizer
            rows_to_remove.add(n + q) # stabilizer Z_q
        keep_rows = [r for r in range(2 * n) if r not in rows_to_remove]

        new_X = X[xp.ix_(keep_rows, keep_qudits)]
        new_Z = Z[xp.ix_(keep_rows, keep_qudits)]
        new_tau = tau[keep_rows].copy()
        new_n = len(keep_qudits)

        if in_place:
            self._X = new_X
            self._Z = new_Z
            self._tau_exp = new_tau
            self._n = new_n
            return self, outcomes

        reduced_tab = Tableau(
            n=new_n,
            d=d,
            full=True,
            X=new_X,
            Z=new_Z,
            tau_exp=new_tau,
        )
        return reduced_tab, outcomes
    
    def measurement_kinds(self,
                      qudits: typing.Optional[typing.Union[int, list[int], tuple[int, ...]]] = None
                      ) -> dict[int, str]:
        """
        Classify whether measuring given qudits in the Z basis is random or deterministic.

        A qudit i is:
        - "rand" if some stabilizer generator has X[:, i] ≠ 0
        - "det"  otherwise

        Parameters
        ----------
        qudits : int | list[int] | tuple[int, ...] | None, optional
            Qudit indices to check. If None (default), checks all qudits 0..n-1.

        Returns
        -------
        dict[int, str]
            Mapping {qudit_index: "rand" | "det"}.
        """
        if not self._full:
            raise ValueError("Classification requires a full tableau.")

        n = self.n
        if qudits is None:
            qs = list(range(n))
        elif isinstance(qudits, int):
            qs = [qudits]
        else:
            qs = list(qudits)

        bad = [q for q in qs if q < 0 or q >= n]
        if bad:
            raise ValueError(f"Qudit index/indices out of bounds for n={n}: {bad}")

        stab_X = self.X[n:]
        col_has_nonzero = xp.any(stab_X != 0, axis=0)

        return {q: ("rand" if bool(col_has_nonzero[q]) else "det") for q in qs}

    
    def to_probvector(self, shots: int, rng=None, seed=None) -> np.ndarray:
        """
        Simulates measuring the state multiple times to estimate its probability vector.

        It repeatedly calls the `measure_all` method on a copy of the tableau
        to sample measurement outcomes and builds a probability distribution from
        the frequency of these outcomes.

        The accuracy of the resulting probability vector depends on the number of shots.

        Parameters
        ----------
        shots : int
            The number of times to sample the measurement outcome.

        Returns
        -------
        np.ndarray
            A column vector of size d**n, where each entry represents the
            estimated probability of measuring the corresponding basis state.
        """
        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")
        
        rng = self._coerce_rng(rng, seed)
        
        shots = int(shots)
        
        n, d = self.n, self.d
        results = {}

        # Sample 'shots' times from the tableau.
        for _ in range(shots):
            _, outcomes = self.measure_all(rng=rng, in_place=False)

            # Increment the count for the measured outcome.
            outcome_tuple = tuple(outcomes)
            results[outcome_tuple] = results.get(outcome_tuple, 0) + 1

        # Convert the frequency map into a probability vector.
        prob_vector = xp.zeros((d**n,), dtype=float)
        
        for outcome_tuple, count in results.items():
            index = 0
            for i, val in enumerate(reversed(outcome_tuple)):
                index += val * (d**i)

            prob_vector[index] = count / shots
            
        return prob_vector

    # ===============================================================================================================
    # Fast sampling
    # ===============================================================================================================
    @staticmethod
    def _rref_mod_d(matrix: np.ndarray, d: int) -> np.ndarray:
        """
        Computes the Row-Reduced Echelon Form of a matrix over the finite field Z_d (prime d).
        """
        A = matrix.copy()
        n_rows, n_cols = A.shape
        pivot_row = 0
        for j in range(n_cols):  # Iterate through columns
            if pivot_row >= n_rows:
                break

            # Find a pivot in the current column
            i = pivot_row
            while i < n_rows and A[i, j] == 0:
                i += 1

            if i < n_rows:  # Pivot found at (i, j)
                # Swap the pivot row into place
                A[[pivot_row, i]] = A[[i, pivot_row]]

                # Normalize the pivot row
                inv = Tableau._mod_inverse(int(A[pivot_row, j]), d)
                A[pivot_row, :] = (A[pivot_row, :] * inv) % d

                # Eliminate other entries in the column
                for k in range(n_rows):
                    if k != pivot_row:
                        factor = A[k, j]
                        A[k, :] = (A[k, :] - factor * A[pivot_row, :]) % d

                pivot_row += 1

        return A

    def sample_measurements(self, shots: int, rng=None, seed=None) -> np.ndarray:
        """
        Efficiently samples measurement outcomes using the affine subspace method.

        This method is significantly faster than calling measure_all() repeatedly.
        It first computes the affine subspace of all possible measurement outcomes
        and then generates random samples from this space.

        The complexity is O(n^3 * log(d)) for a one-time setup, followed by
        a per-shot cost of O(k*n) where k is the rank of the stabilizer X-matrix.

        Uses a single RNG stream (seeded if provided) across all shots.
        
        Args:
            shots (int): The number of measurement samples to generate.

        Returns:
            np.ndarray: A 2D array of shape (shots, n) where each row is a
                        single measurement outcome.
        """
        if not self._full:
            raise ValueError("This sampling method is only supported for full tableaus.")
        if not self._is_prime(self._d):
            raise ValueError(f"This sampling method is only supported for prime dimensions, but d={self._d}.")

        rng = self._coerce_rng(rng, seed)
        
        # --- Setup Phase (One-time cost) ---

        # Get a single measurement outcome to serve as the "offset vector" v0
        v0 = self.measure_all(rng=rng, in_place=False)[1]

        # Get the basis vectors for the linear subspace from the stabilizer X-matrix
        stabilizer_X = self.X[self.n:, :]
        rref_matrix = Tableau._rref_mod_d(stabilizer_X, self.d)
        
        # The basis vectors are the non-zero rows of the RREF matrix
        basis_vectors = rref_matrix[xp.any(rref_matrix, axis=1)]
        k = basis_vectors.shape[0]

        # --- Sampling Phase (Per-shot cost) ---

        # If k=0, the subspace is trivial; all outcomes are just v0
        if k == 0:
            return xp.tile(v0, (shots, 1))

        # Generate all random coefficients at once for efficiency
        coeffs = rng.integers(0, self.d, size=(shots, k))
        
        # Use broadcasting to compute all linear combinations
        # This computes s = c_1*b_1 + c_2*b_2 + ... for each shot
        linear_combinations = xp.dot(coeffs, basis_vectors) % self.d
        
        # Add the offset vector v0 to each linear combination to get the final samples
        samples = (v0 + linear_combinations) % self.d

        return samples
    
    def to_probvector_fast(self, shots: int, rng=None, seed=None) -> np.ndarray:
        """
        Estimates the probability vector by efficiently sampling measurement outcomes.

        This method invokes sample_measurements() and reconstructs the full
        probability vector from the collected samples. It is much faster than
        to_probvector() for a large number of shots.

        Uses rng/seed to create a single RNG stream for all shots.
        
        Args:
            shots (int): The number of samples to use for the estimation.

        Returns:
            np.ndarray: A 1D array of size d**n representing the estimated
                        probability of measuring each basis state.
        """
        if shots <= 0:
            raise ValueError("Number of shots must be a positive integer.")
        
        rng = self._coerce_rng(rng, seed)
        
        shots = int(shots)
        
        n, d = self.n, self.d
        
        # Get all samples using the fast method
        samples = self.sample_measurements(shots, rng=rng)
        
        # Find unique outcomes and their counts
        unique_samples, counts = xp.unique(samples, return_counts=True, axis=0)
        
        # Convert outcomes to flat indices for the probability vector
        powers = xp.power(d, xp.arange(n - 1, -1, -1, dtype=xp.int64))
        indices = xp.dot(unique_samples, powers)
        
        # Create the probability vector
        prob_vector = xp.zeros((d**n,), dtype=float)
        
        # Populate it with the calculated probabilities
        probabilities = counts / shots
        prob_vector[indices] = probabilities
        
        return prob_vector
    
    # ===============================================================================================================
    # Get exact probvector (use when d^n is small)
    # ===============================================================================================================
    
    def enumerate_outcomes(self) -> np.ndarray:
        """
        Systematically generates all possible measurement outcomes for the state.

        This method determines the affine subspace of outcomes and iterates through
        every element, returning the complete set. The complexity is
        O(n^3) + d^k * O(n), where k is the rank of the stabilizer X-matrix.

        Returns:
            np.ndarray: A 2D array of shape (d^k, n) containing all possible
                        measurement outcomes.
        """
        if not self._full:
            raise ValueError("This method is only supported for full tableaus.")
        if not self._is_prime(self._d):
            raise ValueError(f"This method is only supported for prime dimensions, but d={self._d}.")

        n, d = self.n, self.d
        v0 = self.measure_all(in_place=False)[1]

        stabilizer_X = self.X[n:, :]
        rref_matrix = Tableau._rref_mod_d(stabilizer_X, d)
        basis_vectors = rref_matrix[xp.any(rref_matrix, axis=1)]
        k = basis_vectors.shape[0]

        if k == 0:
            return v0.reshape(1, n)
            
        num_outcomes = d**k
        outcomes = xp.zeros((num_outcomes, n), dtype=v0.dtype)
        outcomes[0] = v0
        current_v = v0.copy()
        
        if num_outcomes > 1:
            coeffs = xp.zeros(k, dtype=int)
            for i in range(1, num_outcomes):
                j = 0
                while True:
                    # Subtract the old contribution of basis_vectors[j]
                    current_v = (current_v - coeffs[j] * basis_vectors[j]) % d
                    coeffs[j] = (coeffs[j] + 1) % d
                    # Add the new contribution
                    current_v = (current_v + coeffs[j] * basis_vectors[j]) % d
                    
                    if coeffs[j] != 0:
                        break
                    j += 1
                outcomes[i] = current_v

        return outcomes

    def to_probvector_exact(self) -> np.ndarray:
        """
        Computes the exact probability vector of the stabilizer state.

        This method systematically enumerates all possible measurement outcomes
        and calculates their exact, uniform probability. It is deterministic
        and avoids any sampling error.

        Returns:
            np.ndarray: A 1D array of size d**n representing the exact
                        probability of measuring each basis state.
        """
        n, d = self.n, self.d
        
        # Get all possible outcomes
        all_outcomes = self.enumerate_outcomes()
        num_outcomes = all_outcomes.shape[0]

        if num_outcomes == 0:
            return xp.zeros((d**n,), dtype=float)

        # The probability is uniform over the subspace
        probability = 1.0 / num_outcomes

        # Create the probability vector
        prob_vector = xp.zeros((d**n,), dtype=float)

        # Convert outcomes to indices and place the probability
        powers = xp.power(d, xp.arange(n - 1, -1, -1, dtype=xp.int64))
        indices = xp.dot(all_outcomes, powers)
        
        prob_vector[indices] = probability
        
        return prob_vector
    
    # ===============================================================================================================
    # Sample from noisy tableaus (Monte-Carlo)
    # ===============================================================================================================
    def sample_measurements_noisy(
        self,
        qc: QuantumCircuit,
        shots: int,
        color: str = "#FF0000",
        show_identities: bool = False,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        return_circuits: bool = False,
    ) -> tuple[np.ndarray, list[QuantumCircuit] | None]:
        """
        Monte-Carlo sampling from a noisy circuit using stabilizer tableaus.

        Parameters
        ----------
        qc : QuantumCircuit
            A circuit that may contain noise-gate placeholders. Each shot will
            realize a concrete noisy instance via `qc.realize_noise(...)`.
        shots : int
            Number of Monte-Carlo shots.
        color : str, optional
            Color to use for Weyl Operators in the drawn circuits.
        show_identities : bool, optional
            If True, identity gates are added to the generated circuits.
        rng : numpy.random.Generator | None, optional
            Random number generator to use. If None, one is created from `seed`.
            If both `rng` and `seed` are provided, `rng` is used.
        seed : int | None, optional
            Seed to create a fresh RNG when `rng` is None.
        return_circuits : bool, optional
            If True, also return the list of realized per-shot noisy circuits.

        Returns
        -------
        outcomes : np.ndarray
            Array of shape (shots, n) with integer outcomes in base-d for each qudit.
        circuits : list[QuantumCircuit] | None
            If `return_circuits=True`, a list of the realized noisy circuits (length = shots).
            Otherwise, None.

        Notes
        -----
        This method does not mutate `self`. A fresh tableau is used per shot.
        """
        if shots <= 0:
            raise ValueError("shots must be a positive integer.")
        
        rng = self._coerce_rng(rng, seed)

        per_shot_noise_seeds = rng.integers(0, np.iinfo(np.int64).max, size=shots, dtype=np.int64)
        per_shot_meas_seeds = rng.integers(0, np.iinfo(np.int64).max, size=shots, dtype=np.int64)

        n, d = self.n, self.d
        outcomes = np.empty((shots, n), dtype=np.int64)
        circuits_list = [] if return_circuits else None

        for i in range(shots):
            noisy_qc, _ = qc.realize_noise(..., seed=int(per_shot_noise_seeds[i]), in_place=False)
            
            tab = type(self)(n, d, full=True)
            tab.apply_circuit(noisy_qc)

            _, shot_outcome = tab.measure_all(in_place=False, seed=int(per_shot_meas_seeds[i]))
            
            outcomes[i] = np.asarray(shot_outcome, dtype=np.int64)

        return outcomes, circuits_list
    
    def to_probvector_noisy(
        self,
        qc: QuantumCircuit,
        shots: int,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Estimate the output probability vector via Monte-Carlo sampling of a noisy circuit.

        Parameters
        ----------
        qc : QuantumCircuit
            A circuit that may contain noise-gate placeholders. Each shot will
            realize a concrete noisy instance via `qc.realize_noise(...)`.
        shots : int
            Number of Monte-Carlo shots used for the estimate.
        rng : numpy.random.Generator | None, optional
            Random number generator to use. If None, one is created from `seed`.
            If both `rng` and `seed` are provided, `rng` is used.
        seed : int | None, optional
            Seed to create a fresh RNG when `rng` is None.

        Returns
        -------
        prob_vector : np.ndarray
            A length d**n vector of probabilities.
        Notes
        -----
        The mapping from an n-digit base-d string x_{n-1}...x_0 to index is
        sum_{k=0}^{n-1} x_k * d^{n-1-k}.
        """
        outcomes, _ = self.sample_measurements_noisy(
            qc=qc, shots=shots, rng=rng, seed=seed, return_circuits=False
        )

        n, d = self.n, self.d
        unique_samples, counts = np.unique(outcomes, return_counts=True, axis=0)
        powers = np.power(d, np.arange(n - 1, -1, -1, dtype=np.int64))
        indices = np.dot(unique_samples, powers)
        prob_vector = np.zeros((d**n,), dtype=float)
        prob_vector[indices] = counts.astype(float) / float(shots)
        return prob_vector
    
    
    # ===============================================================================================================
    # Push NoiseChannels to the end of the circuit
    # ===============================================================================================================
    @classmethod
    def clean_run(cls, qc):
        """
        Run qc once, skipping NoiseChannel ops but caching
        (q, X[:,q], Z[:,q], model_id) for each noise insertion point.
        """
        T = cls(qc.n, qc.d, full=True)
        noise_frames = []

        for gate, qudits, dagger in qc._ops:
            if isinstance(gate, NoiseChannel):
                q = qudits[0]
                Xcol = T.X[:, q].copy()
                Zcol = T.Z[:, q].copy()
                noise_frames.append((q, Xcol, Zcol, gate.model_id))
            else:
                T.apply_gate(gate, qudits, dagger)

        return T, noise_frames
    
    @staticmethod
    def _sample_delta_taus(qc, T_clean, noise_frames, shots: int, rng=None, seed=None):
        """
        Sample Δτ for all noise frames & all shots.
        Output shape: (shots, 2n)
        """
        if rng is None:
            rng = xp.random.default_rng(seed)

        d = qc.d
        mod_2d = xp.int64(2 * d)
        n = qc.n
        rows = 2 * n

        delta_taus = xp.zeros((shots, rows), dtype=T_clean._tau_dtype)

        for (q, Xcol, Zcol, model_id) in noise_frames:
            probs = qc.noise_registry[model_id].reshape(-1)

            # sample S indices for this noise location
            choice = rng.choice(d*d, size=shots, p=probs)

            # convert to a_j, b_j
            a_j, b_j = xp.unravel_index(choice, (d, d))

            contrib = (2 * b_j[:,None] * Xcol[None,:] -
                       2 * a_j[:,None] * Zcol[None,:])

            delta_taus = (delta_taus + contrib) % mod_2d

        return delta_taus
    
    def sample_measurements_noisy_fast(self, qc, shots: int, rng=None, seed=None):
        """
        Main fast noisy sampling:
        1) clean run: T_clean, noise_frames
        2) sample Δτ vectors
        3) apply phase + measure using CHP
        """
        if rng is None:
            rng = xp.random.default_rng(seed)

        T_clean, noise_frames = Tableau.clean_run(qc)
        delta_taus = Tableau._sample_delta_taus(qc, T_clean, noise_frames, shots, rng)

        mod_2d = T_clean.mod_2d
        base_tau = T_clean._tau_exp.copy()

        results = xp.zeros((shots, qc.n), dtype=xp.int64)

        for s in range(shots):
            Ts = T_clean.copy()
            Ts._tau_exp = (base_tau + delta_taus[s]) % mod_2d
            _, res = Ts.measure_all(in_place=False)
            results[s] = res

        return results
    
    def to_probvector_noisy_fast(
        self,
        qc: QuantumCircuit,
        shots: int,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Works the same as **to_probvector_noisy**,
        but uses **sample_measurements_noisy_fast**
        to obtain samples.
        """
        outcomes = self.sample_measurements_noisy_fast(
            qc=qc, shots=shots, rng=rng, seed=seed
        )

        n, d = self.n, self.d
        unique_samples, counts = np.unique(outcomes, return_counts=True, axis=0)
        powers = np.power(d, np.arange(n - 1, -1, -1, dtype=np.int64))
        indices = np.dot(unique_samples, powers)
        prob_vector = np.zeros((d**n,), dtype=float)
        prob_vector[indices] = counts.astype(float) / float(shots)
        return prob_vector