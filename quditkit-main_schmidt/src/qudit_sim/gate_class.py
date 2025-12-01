from __future__ import annotations
from typing import Callable
import numpy as np
from .backend import xp
from . import gates

try:
    from qiskit.circuit import Gate as QiskitGate
    _HAS_QISKIT = True
except ImportError:
    _HAS_QISKIT = False

# ---------------------------------------------------------------------------
# Gate class for both matrix & direct-vector application --------------------
class Gate:
    def __init__(
        self,
        name: str,
        n_qudits: int,
        matrix_fn: Callable[[int], np.ndarray],
        dagger_matrix_fn: Callable[[int], np.ndarray] | None = None,
        apply_fn: Callable[[np.ndarray, int, int, tuple[int, ...], bool], np.ndarray] | None = None,
        color: str | None = None,
        a: float | None = None,
        b: float | None = None
    ):
        self.name = name
        self.n_qudits = n_qudits
        self.matrix_fn = matrix_fn
        self.dagger_matrix_fn = dagger_matrix_fn
        self._apply_fn = apply_fn
        self.color = color  # for visualization, if needed
        self.a = a          # for Weyl operators
        self.b = b          # for Weyl operators

    def __repr__(self):
        return f"Gate(name={self.name!r}, n_qudits={self.n_qudits})"
    
    def __str__(self):
        return f"<Gate {self.name}: {self.n_qudits}-qudit>"

    def copy(self, *, name: str | None = None, color: str | None = None) -> Gate:
        """
        Return a (shallow) copy of this Gate. Callable attributes are reused by reference.
        You may optionally override `name` and/or `color`.

        Examples
        --------
        g2 = g.copy()                          # exact duplicate
        g3 = g.copy(name=f"{g.name}_clone")    # same gate, new name
        """
        return Gate(
            name=name if name is not None else self.name,
            n_qudits=self.n_qudits,
            matrix_fn=self.matrix_fn,
            dagger_matrix_fn=self.dagger_matrix_fn,
            apply_fn=self._apply_fn,
            color=color if color is not None else self.color,
            a=self.a,
            b=self.b
        )

    def __copy__(self) -> Gate:
        return self.copy()

    def matrix(self, d: int, dagger: bool = False) -> np.ndarray:
        if dagger:
            if not self.dagger_matrix_fn:
                raise ValueError(f"No dagger matrix available for gate {self.name}")
            return self.dagger_matrix_fn(d)
        return self.matrix_fn(d)

    def apply(
        self,
        psi: np.ndarray,
        n: int,
        d: int,
        qudits: tuple[int, ...],
        dagger: bool = False,
    ) -> np.ndarray:
        # If there’s a specialized apply_fn (e.g. reshape+index CNOT),
        # just hand psi off to it.
        if self._apply_fn is not None:
            if self.a is not None and self.b is not None:
                # This is a Weyl gate
                return self._apply_fn(psi, n, d, qudits, dagger, a=self.a, b=self.b)
            else:
                # CNOT/CZ/SWAP
                return self._apply_fn(psi, n, d, qudits, dagger)

        # if this is a single‐qudit *system* (n==1), 
        # just do the direct d×d mat‑vec once
        if n == 1:
            U = self.matrix(d, dagger)
            return U @ psi

        # Otherwise do a tensordot update for a k‑qudit gate
        Umat = self.dagger_matrix_fn(d) if dagger else self.matrix_fn(d)
        k    = self.n_qudits

        U = Umat.reshape((d,)*k + (d,)*k)
        psi_nd = psi.reshape((d,)*n)
        axes_U   = list(range(k, 2*k))
        axes_psi = list(qudits)
        contracted = xp.tensordot(U, psi_nd, axes=(axes_U, axes_psi))
        rest = [i for i in range(n) if i not in qudits]

        perm = []
        for i in range(n):
            if i in qudits:
                perm.append(qudits.index(i))
            else:
                perm.append(k + rest.index(i))

        out = xp.transpose(contracted, perm)
        return out.reshape(-1)

    @staticmethod
    def _apply_operator_to_density_matrix(
        rho: np.ndarray,
        operator: np.ndarray,
        qudits: tuple[int, ...],
        n: int,
        d: int
    ) -> np.ndarray:
        """
        Applies rho' = U * rho * U_dagger by explicitly constructing the full d^n x d^n operator.
        """
        from .circuit import QuantumCircuit
        k = len(qudits)

        # Create a temporary gate object that holds the small operator.
        temp_gate = Gate("temp", k, lambda d_val: operator)
        
        # Create a temporary circuit to access the _get_gate_unitary helper.
        temp_qc = QuantumCircuit(n, d)
        
        # Use the helper to construct the full d^n x d^n operator.
        U = temp_qc._get_gate_unitary(temp_gate, qudits, False)

        # Get the conjugate transpose for the right-multiplication.
        U_dag = xp.conjugate(U.T)

        # Perform the matrix multiplication.
        return U @ rho @ U_dag
    
    def apply_density(
        self,
        rho: np.ndarray,
        circuit,
        qudits: tuple[int, ...],
        dagger: bool = False,
    ) -> np.ndarray:
        """
        Applies this gate to a density matrix.
        For unitary gates, this calculates rho' = U * rho * U_dagger.
        """
        # Placeholder gates like "M" or "P" have no matrix_fn and should be skipped.
        if self.matrix_fn is None:
            return rho
            
        # Get the small d^k x d^k unitary for this gate.
        U = self.matrix(circuit.d, dagger)
        
        # Use the static helper to apply it to the full density matrix.
        return Gate._apply_operator_to_density_matrix(
            rho, U, qudits, circuit.n, circuit.d
        )
        
    def to_qiskit_instruction(self, dagger: bool = False):
        if not _HAS_QISKIT:
            raise ImportError("Qiskit is not installed")
        name = f"{self.name}†" if dagger else self.name
        return QiskitGate(name, self.n_qudits, [])
    
class NoiseChannel(Gate):
    """Represents a noise channel.
    
    Stores a `model_id` that can be used
    to look probability matrix up in a circuit's noise registry.
    """
    def __init__(self, model_id: int, color: tuple | None = None):
        """
        Args:
            model_id: The integer ID corresponding to a noise model in a circuit's registry.
            color: The RGBA color tuple for this gate, with opacity reflecting the error probability.
        """
        # Call the parent Gate constructor with fixed properties for a noise channel
        super().__init__(
            name="NOISE_CHANNEL",
            n_qudits=1,
            matrix_fn=None,
            color=color
        )
        self.model_id = model_id

    # override apply_density to implement Kraus operator sum
    def apply_density(
        self,
        rho: np.ndarray,
        circuit,
        qudits: tuple[int, ...],
        dagger: bool = False, # dagger is ignored for noise
    ) -> np.ndarray:
        """
        Applies this noise channel to a density matrix using the Kraus operator sum representation.
        Implements on-the-fly computation and caching of Kraus operators.
        """
        # Check the circuit's cache for our Kraus operators.
        kraus_operators = circuit._kraus_operator_cache.get(self.model_id)

        # If not in the cache, compute and store them.
        if kraus_operators is None:
            # Retrieve the probability array from the permanent registry.
            prob_array = circuit.noise_registry.get(self.model_id)
            if prob_array is None:
                # This should not happen if the circuit is constructed correctly.
                return rho

            d = circuit.d
            prob_matrix = prob_array.reshape((d, d))
            
            # Compute the list of d x d Kraus operators.
            kraus_operators = []
            for a in range(d):
                for b in range(d):
                    prob = prob_matrix[a, b]
                    if prob > 1e-12: # Only create operators for non-zero probabilities.
                        # E_k = sqrt(p_k) * W_k
                        kraus_op = xp.sqrt(prob) * gates.W(d, a, b)
                        kraus_operators.append(kraus_op)
            
            # Store the computed list in the cache for future use in this run.
            circuit._kraus_operator_cache[self.model_id] = kraus_operators

        # Apply the Kraus sum: rho' = sum(E_k * rho * E_k_dagger)
        rho_new = xp.zeros_like(rho)
        for E_k in kraus_operators:
            term = Gate._apply_operator_to_density_matrix(
                rho, E_k, qudits, circuit.n, circuit.d
            )
            rho_new += term

        return rho_new
