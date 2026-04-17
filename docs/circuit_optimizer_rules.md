# Native Quditkit Circuit Optimizer — Optimization Rules

Gate set in scope: `H`, `X`, `Z`, `CNOT`, `SWAP`

> `CCX` (Toffoli) is not a Clifford gate and is not supported by quditkit — it never appears in quditkit circuits.

Convention: for multi-qubit gates, `op[1] = (control, target)` — e.g. `CNOT(0,1)` has control=0, target=1.

---

## 1. Self-Inverse Cancellation

Any gate `G` satisfying `G·G = I` can be cancelled against an identical copy.
The search is not limited to adjacent gates — a matching copy is accepted as long as every
intervening gate on shared qubits **commutes** with `G` (see Section 3).

| Gate | Identity |
|------|----------|
| `H(q)` | `H·H = I` |
| `X(q)` | `X·X = I` |
| `Z(q)` | `Z·Z = I` |
| `CNOT(c,t)` | `CNOT·CNOT = I` (same control & target) |
| `SWAP(a,b)` | `SWAP·SWAP = I` (same qubit pair, order-normalised) |

SWAP qubit ordering is normalised to `(min, max)` before matching so `SWAP(1,0)` and `SWAP(0,1)` are treated as identical.

---

## 2. Template Rewriting

Contiguous 3-gate windows are scanned left-to-right and replaced when a pattern matches.
The fixed-point loop means chained reductions (e.g. two overlapping 3-gate patterns) are
caught across iterations.

| Pattern | Result | Gate saving |
|---------|--------|-------------|
| `H(q) · Z(q) · H(q)` | `X(q)` | −2 gates |
| `H(q) · X(q) · H(q)` | `Z(q)` | −2 gates |
| `H(a) · H(b) · CNOT(a,b) · H(a) · H(b)` | `CNOT(b,a)` | −4 gates |
| `CNOT(a,b) · CNOT(b,a) · CNOT(a,b)` | `SWAP(a,b)` | −2 gates |

**Note on `H·CNOT·H`:** the flip identity requires H on **both** qubits (control
and target). `H(a)·CNOT(a,b)·H(a)` alone does not equal `CNOT(b,a)` — this was
verified numerically. The correct rule is the 5-gate window above.

---

## 3. Commutation Rules

Used by the cancellation pass to decide whether a gate can "look past" a blocker.
If gate `A` and gate `B` share at least one qubit, the pass checks whether they commute
before deciding to skip or stop.

| Gate A | Gate B | Commutes? | Condition |
|--------|--------|-----------|-----------|
| any | any | ✅ | No shared qubits (trivially disjoint) |
| `Z(q)` | `CNOT(c,t)` | ✅ | `q == c` (Z is on the *control*) |
| `X(q)` | `CNOT(c,t)` | ✅ | `q == t` (X is on the *target*) |
| `CNOT(c,t1)` | `CNOT(c,t2)` | ✅ | Same control, different targets |
| `CNOT(c1,t)` | `CNOT(c2,t)` | ✅ | Different controls, same target |
| `H(q)` | any on `q` | ❌ | H blocks all commutation on the same qubit |
| `Z(q)` | `CNOT(c,t)` | ❌ | `q == t` (Z is on the *target*) |
| `X(q)` | `CNOT(c,t)` | ❌ | `q == c` (X is on the *control*) |

All other overlapping-qubit pairs not listed above are treated as **non-commuting** (conservative default).

---

## 4. Fixed-Point Iteration

The two passes (cancellation, template rewriting) are applied in a loop until the circuit
length stops decreasing:

```
normalize SWAP ordering
repeat:
    run cancellation pass   (self-inverse + commutation)
    run template pass       (3-gate window rewrites)
until len(ops) unchanged
```

Termination is guaranteed because every pass either reduces the gate count or leaves it unchanged.
