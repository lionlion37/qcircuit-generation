"""Native quditkit circuit optimizer.

Operates directly on ``QuantumCircuit.ops`` — a list of
``(Gate, tuple[int, ...], bool)`` tuples — without any qiskit round-trip.

Convention: for multi-qubit gates ``op[1] = (control, target)``.
"""

from __future__ import annotations

_SELF_INVERSE = frozenset({"H", "X", "Z", "CNOT", "SWAP"})


def _qubits(op) -> frozenset:
    return frozenset(op[1])


def _normalize_op(op):
    """Normalize SWAP qubit ordering so (a,b) and (b,a) are treated as identical."""
    gate, qubits, dagger = op
    if gate.name == "SWAP" and len(qubits) == 2 and qubits[0] > qubits[1]:
        return (gate, (qubits[1], qubits[0]), dagger)
    return op


def _normalize_ops(ops):
    return [_normalize_op(op) for op in ops]


def _commutes(a, b) -> bool:
    """Return True if gates a and b commute and can be reordered."""
    qa = frozenset(a[1])
    qb = frozenset(b[1])
    if not qa & qb:
        return True  # disjoint qubits always commute

    na = a[0].name
    nb = b[0].name

    # Z on CNOT control commutes (acts_on[0] is control)
    if na == "Z" and nb == "CNOT":
        if list(qa)[0] == b[1][0]:
            return True
    if nb == "Z" and na == "CNOT":
        if list(qb)[0] == a[1][0]:
            return True

    # X on CNOT target commutes (acts_on[1] is target)
    if na == "X" and nb == "CNOT":
        if list(qa)[0] == b[1][1]:
            return True
    if nb == "X" and na == "CNOT":
        if list(qb)[0] == a[1][1]:
            return True

    # Two CNOTs sharing only control commute; two CNOTs sharing only target commute
    if na == "CNOT" and nb == "CNOT":
        ctrl_a, tgt_a = a[1][0], a[1][1]
        ctrl_b, tgt_b = b[1][0], b[1][1]
        if ctrl_a == ctrl_b and tgt_a != tgt_b:
            return True
        if tgt_a == tgt_b and ctrl_a != ctrl_b:
            return True

    return False


def _ops_match(a, b) -> bool:
    return a[0].name == b[0].name and a[1] == b[1]


def _cancellation_pass(ops: list) -> list:
    """Cancel self-inverse gate pairs, commuting past non-blocking gates."""
    n = len(ops)
    cancelled = [False] * n

    for i in range(n):
        if cancelled[i]:
            continue
        if ops[i][0].name not in _SELF_INVERSE:
            continue

        for j in range(i + 1, n):
            if cancelled[j]:
                continue
            if _qubits(ops[i]) & _qubits(ops[j]):
                if _ops_match(ops[i], ops[j]):
                    cancelled[i] = cancelled[j] = True
                    break
                elif _commutes(ops[i], ops[j]):
                    continue  # look past this gate
                else:
                    break  # blocked

    return [op for k, op in enumerate(ops) if not cancelled[k]]


def _apply_templates(ops: list) -> list:
    """Single left-to-right pass of 3-gate template matching."""
    from qudit_sim.predefined_gates import X_gate, Z_gate, SWAP_gate, CNOT_gate

    result = []
    i = 0
    while i < len(ops):
        if i + 2 <= len(ops) - 1:
            a, b, c = ops[i], ops[i + 1], ops[i + 2]
            na, nb, nc = a[0].name, b[0].name, c[0].name
            qa, qb, qc = a[1], b[1], c[1]

            # H·Z·H → X
            if na == "H" and nb == "Z" and nc == "H" and qa == qb == qc:
                result.append((X_gate, qa, False))
                i += 3
                continue

            # H·X·H → Z
            if na == "H" and nb == "X" and nc == "H" and qa == qb == qc:
                result.append((Z_gate, qa, False))
                i += 3
                continue

            # CNOT(a,b)·CNOT(b,a)·CNOT(a,b) → SWAP(a,b)
            if na == "CNOT" and nb == "CNOT" and nc == "CNOT":
                if qa == qc and qb == (qa[1], qa[0]):
                    lo, hi = min(qa[0], qa[1]), max(qa[0], qa[1])
                    result.append((SWAP_gate, (lo, hi), False))
                    i += 3
                    continue

        # 5-gate window: H(a)·H(b)·CNOT(a,b)·H(a)·H(b) → CNOT(b,a)
        if i + 4 <= len(ops) - 1:
            a, b, c, d, e = ops[i], ops[i+1], ops[i+2], ops[i+3], ops[i+4]
            if (a[0].name == "H" and b[0].name == "H" and c[0].name == "CNOT"
                    and d[0].name == "H" and e[0].name == "H"):
                ctrl, tgt = c[1][0], c[1][1]
                if a[1] == d[1] == (ctrl,) and b[1] == e[1] == (tgt,):
                    result.append((CNOT_gate, (tgt, ctrl), False))
                    i += 5
                    continue

        result.append(ops[i])
        i += 1
    return result


def optimize_ops(ops: list) -> list:
    """Optimize a flat op list via fixed-point cancellation + template rewriting."""
    ops = _normalize_ops(ops)
    prev_len = -1
    while len(ops) != prev_len:
        prev_len = len(ops)
        ops = _cancellation_pass(ops)
        ops = _apply_templates(ops)
    return ops
