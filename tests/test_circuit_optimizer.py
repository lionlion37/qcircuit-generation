"""Tests for the native quditkit circuit optimizer.

Verifies two properties for every case:
1. Gate count is reduced (or unchanged) — never increases.
2. The circuit unitary is preserved exactly.
"""

import numpy as np
import pytest
from qudit_sim.circuit import QuantumCircuit
from qudit_sim.predefined_gates import H_gate, X_gate, Z_gate, CNOT_gate, SWAP_gate

from my_genQC.platform.backends.circuit_optimizer import optimize_ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unitary(ops, n_qubits):
    if not ops:
        return np.eye(2**n_qubits, dtype=complex)
    qc = QuantumCircuit(num_qudits=n_qubits, dim=2)
    for g, q, d in ops:
        qc.append(g, q, dagger=d)
    return qc.get_unitary()


def _assert_preserves_unitary(ops, n_qubits):
    U_before = _unitary(ops, n_qubits)
    opt = optimize_ops(list(ops))
    U_after = _unitary(opt, n_qubits)
    assert len(opt) <= len(ops), f"optimizer increased gate count: {len(ops)} -> {len(opt)}"
    assert np.allclose(U_before, U_after, atol=1e-10), (
        f"unitary mismatch after optimization\n"
        f"  before ({len(ops)} gates): {[(g.name, q) for g,q,_ in ops]}\n"
        f"  after  ({len(opt)} gates): {[(g.name, q) for g,q,_ in opt]}"
    )
    return opt


# ---------------------------------------------------------------------------
# Self-inverse cancellation
# ---------------------------------------------------------------------------

class TestSelfInverseCancellation:
    def test_h_h(self):
        ops = [(H_gate, (0,), False), (H_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 0

    def test_x_x(self):
        ops = [(X_gate, (0,), False), (X_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 0

    def test_z_z(self):
        ops = [(Z_gate, (0,), False), (Z_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 0

    def test_cnot_cnot(self):
        ops = [(CNOT_gate, (0, 1), False), (CNOT_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 0

    def test_swap_swap(self):
        ops = [(SWAP_gate, (0, 1), False), (SWAP_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 0

    def test_swap_order_normalised(self):
        """SWAP(1,0) and SWAP(0,1) should cancel."""
        ops = [(SWAP_gate, (1, 0), False), (SWAP_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 0

    def test_no_cancel_different_qubits(self):
        """H(0) and H(1) must NOT cancel each other."""
        ops = [(H_gate, (0,), False), (H_gate, (1,), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 2

    def test_no_cancel_different_gate(self):
        """H(0) and X(0) must NOT cancel."""
        ops = [(H_gate, (0,), False), (X_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 2


# ---------------------------------------------------------------------------
# Commutation-enabled cancellation
# ---------------------------------------------------------------------------

class TestCommutationCancellation:
    def test_h_disjoint_h(self):
        """H(0) commutes past Z(1) and cancels with H(0)."""
        ops = [(H_gate, (0,), False), (Z_gate, (1,), False), (H_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 1  # only Z(1) remains

    def test_cnot_z_on_control_cnot(self):
        """Z on control commutes past CNOT; the two CNOTs cancel."""
        ops = [(CNOT_gate, (0, 1), False), (Z_gate, (0,), False), (CNOT_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 1  # only Z(0) remains

    def test_cnot_x_on_target_cnot(self):
        """X on target commutes past CNOT; the two CNOTs cancel."""
        ops = [(CNOT_gate, (0, 1), False), (X_gate, (1,), False), (CNOT_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 1  # only X(1) remains

    def test_z_on_target_blocks(self):
        """Z on CNOT target does NOT commute; cancellation must not happen."""
        ops = [(CNOT_gate, (0, 1), False), (Z_gate, (1,), False), (CNOT_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 3  # nothing should cancel

    def test_x_on_control_blocks(self):
        """X on CNOT control does NOT commute; cancellation must not happen."""
        ops = [(CNOT_gate, (0, 1), False), (X_gate, (0,), False), (CNOT_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 3

    def test_h_blocks_same_qubit(self):
        """H between two CNOTs on the same qubits blocks cancellation."""
        ops = [(CNOT_gate, (0, 1), False), (H_gate, (0,), False), (CNOT_gate, (0, 1), False)]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 3

    def test_same_control_cnots_commute(self):
        """CNOT(0,1) and CNOT(0,2) share only the control and commute."""
        ops = [
            (CNOT_gate, (0, 1), False),
            (CNOT_gate, (0, 2), False),
            (CNOT_gate, (0, 1), False),  # should cancel with first
        ]
        opt = _assert_preserves_unitary(ops, 3)
        assert len(opt) == 1  # only CNOT(0,2) remains

    def test_same_target_cnots_commute(self):
        """CNOT(0,2) and CNOT(1,2) share only the target and commute."""
        ops = [
            (CNOT_gate, (0, 2), False),
            (CNOT_gate, (1, 2), False),
            (CNOT_gate, (0, 2), False),  # should cancel with first
        ]
        opt = _assert_preserves_unitary(ops, 3)
        assert len(opt) == 1  # only CNOT(1,2) remains


# ---------------------------------------------------------------------------
# Template rewriting
# ---------------------------------------------------------------------------

class TestTemplates:
    def test_hzh_to_x(self):
        ops = [(H_gate, (0,), False), (Z_gate, (0,), False), (H_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 1
        assert opt[0][0].name == "X"

    def test_hxh_to_z(self):
        ops = [(H_gate, (0,), False), (X_gate, (0,), False), (H_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 1
        assert opt[0][0].name == "Z"

    def test_cnot3_to_swap(self):
        ops = [
            (CNOT_gate, (0, 1), False),
            (CNOT_gate, (1, 0), False),
            (CNOT_gate, (0, 1), False),
        ]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 1
        assert opt[0][0].name == "SWAP"

    def test_hhcnotHH_to_cnot_flipped(self):
        """H(a)·H(b)·CNOT(a,b)·H(a)·H(b) → CNOT(b,a)."""
        ops = [
            (H_gate, (0,), False),
            (H_gate, (1,), False),
            (CNOT_gate, (0, 1), False),
            (H_gate, (0,), False),
            (H_gate, (1,), False),
        ]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 1
        assert opt[0][0].name == "CNOT"
        assert opt[0][1] == (1, 0)  # flipped

    def test_hzh_followed_by_hzh(self):
        """Two consecutive H·Z·H blocks both reduce."""
        ops = [
            (H_gate, (0,), False), (Z_gate, (0,), False), (H_gate, (0,), False),
            (H_gate, (1,), False), (Z_gate, (1,), False), (H_gate, (1,), False),
        ]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 2  # X(0) and X(1)


# ---------------------------------------------------------------------------
# Fixed-point / chained reductions
# ---------------------------------------------------------------------------

class TestFixedPoint:
    def test_hzh_produces_x_which_then_cancels(self):
        """H·Z·H → X, then X·X → I (two-iteration reduction)."""
        ops = [
            (H_gate, (0,), False), (Z_gate, (0,), False), (H_gate, (0,), False),
            (X_gate, (0,), False),  # cancels with the X produced above
        ]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 0

    def test_chain_reduces_to_empty(self):
        """H·H·H·H = I via two cancellation rounds."""
        ops = [(H_gate, (0,), False)] * 4
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 0


# ---------------------------------------------------------------------------
# No-op / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_circuit(self):
        opt = optimize_ops([])
        assert opt == []

    def test_single_gate_unchanged(self):
        ops = [(H_gate, (0,), False)]
        opt = _assert_preserves_unitary(ops, 1)
        assert len(opt) == 1

    def test_no_optimization_possible(self):
        """A circuit with no cancellable patterns stays the same length."""
        ops = [
            (H_gate, (0,), False),
            (CNOT_gate, (0, 1), False),
            (Z_gate, (1,), False),
        ]
        opt = _assert_preserves_unitary(ops, 2)
        assert len(opt) == 3


# ---------------------------------------------------------------------------
# Randomized unitary-preservation
# ---------------------------------------------------------------------------

GATE_POOL_1Q = [H_gate, X_gate, Z_gate]
GATE_POOL_2Q = [CNOT_gate, SWAP_gate]


def _random_ops(n_qubits, n_gates, rng):
    ops = []
    for _ in range(n_gates):
        if n_qubits == 1 or rng.random() < 0.5:
            g = GATE_POOL_1Q[rng.integers(len(GATE_POOL_1Q))]
            q = (int(rng.integers(n_qubits)),)
        else:
            g = GATE_POOL_2Q[rng.integers(len(GATE_POOL_2Q))]
            q = tuple(int(x) for x in rng.choice(n_qubits, 2, replace=False))
        ops.append((g, q, False))
    return ops


@pytest.mark.parametrize("seed", range(20))
def test_random_unitary_preservation(seed):
    """Optimizer must preserve the circuit unitary on random circuits."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 4))
    n_gates = int(rng.integers(0, 16))
    ops = _random_ops(n, n_gates, rng)
    _assert_preserves_unitary(ops, n)
