# Research Brainstorm — Noise-Aware Circuit Generation

*Written 2026-05-25. Covers everything discussed after the SWAP conditioning experiments.*

---

## 1. What we established with SWAP conditioning

Two models trained to condition on `; swap_count=N` in the CLIP text prompt.

| Model | exact_found_rate (count=0) | SWAP compliance | Training |
|---|---|---|---|
| From-scratch | 0.648 | ~100% | 150 epochs, lr=3e-4 |
| Fine-tuned from baseline | 0.898 | 97% | 75 epochs, lr=3e-5 |
| Baseline (no conditioning) | 0.961 | — | — |

**Key findings:**
- Fine-tuning from baseline recovers ~30pp of the accuracy drop vs from-scratch
- Initialisation matters more than training schedule
- CLIP text conditioning is too coarse for clean numeric constraints — the suffix shifts the whole embedding vector

---

## 2. Noise simulation results

Depolarising noise model (p₁=0.001, p₂=0.01, p_swap=0.03, p_ccx=0.06).
64 unitaries, 64 samples, finetuned model.

| Requested SWAP count | Mean noisy fidelity | Mean total gates | Effective 2q (CX + 3×SWAP) |
|---|---|---|---|
| 0 | 0.952 | 4.28 | 1.54 |
| 1 | 0.936 | 4.63 | 4.58 |
| 2 | 0.919 | 5.39 | 7.15 |
| 3 | 0.898 | 6.15 | 10.40 |

**Two criticisms addressed:**
1. *"The model just substitutes CX for SWAP"* — refuted. Effective 2q (CX + 3×SWAP on real hardware) is 1.54 for swap_count=0 vs 10.40 for swap_count=3. The model finds genuinely shorter circuits.
2. *"The user isn't getting the optimal circuit"* — refuted. Oracle analysis: for all 63 unitaries, the oracle-best circuit is always at swap_count=0. The constraint is not suboptimal.

---

## 3. The simpler alternative that undermines SWAP conditioning

**Finding:** simply removing `'swap'` from the gate set string in the CLIP prompt achieves the same result without any retraining:

| Approach | exact_found_rate | mean_actual_swap | Training required |
|---|---|---|---|
| Baseline (with swap) | 0.969 | 0.976 | — |
| Baseline (swap removed from prompt) | 0.906 | 0.001 | **none** |
| Finetuned, swap_count=0 | 0.898 | 0.023 | 75 epochs |

**Implication:** SWAP conditioning adds no practical value over gate set manipulation. The CLIP gate set conditioning was already doing this implicitly.

---

## 4. CCX conditioning

The baseline uses mean_ccx=1.143 per circuit — CCX is far more expensive (≡6 CX on hardware) than SWAP (≡3 CX). Targeting CCX has a larger potential payoff.

**Results with gate set manipulation:**

| Prompt variant | exact_found | mean_ccx | eff_2q (CX + 6×CCX + 3×SWAP) |
|---|---|---|---|
| Baseline | 0.969 | 1.143 | 10.72 |
| No CCX in gate set | 0.844 | 0.551 | 8.68 |
| No CCX + no SWAP | 0.844 | 0.414 | 5.38 |

**Important difference from SWAP:** CCX compliance is imperfect — removing CCX from the prompt only reduces it from 1.14 to 0.41, not to near-zero. The model doesn't fully respect the gate set hint for CCX, presumably because CCX appears densely in training circuits.

**The tradeoff exists but has zero overlap:** Circuits with 0 CCX have eff_2q=6.25 and fidelity=0.953; circuits with 1 CCX have eff_2q=11.47 and fidelity=0.909. But among 64 evaluation unitaries, not a single one has both a 0-CCX and a 1-CCX exact circuit — the model commits to one compilation strategy per unitary.

**Why ccx_count=N conditioning is more interesting than ccx_count=0 only:**
- `ccx_count=0` just replicates what removing CCX from the gate set already does
- The interesting part is the full range: for some unitaries, is 1 CCX cheaper than the CCX-free alternative?
- If the CCX-free alternative needs 7+ CX, then 1 CCX (≡6 CX) wins
- This is a genuine per-unitary optimisation problem, not always answered by "use fewer CCX"

**But the same problem recurs:** `ccx_count=0` achieves what gate set removal already does, and for the full range, gate set manipulation (with/without CCX) already provides binary control. The only genuinely novel gate for count conditioning is CX — the one gate you can't remove from the gate set.

---

## 5. The logical boundary of gate count conditioning

| Gate | Gate set removal works? | Count conditioning genuinely novel? |
|---|---|---|
| SWAP | Yes (drops to ~0) | No |
| CCX | Partially (drops to 0.41) | No |
| CX | No (can't remove entanglement) | **Yes** |

CX count conditioning is the only case where count conditioning adds something that gate set manipulation cannot. But it's also the hardest to train for: wide range, CX appears in almost every circuit, CLIP number separation is weak.

---

## 6. Noise budget / noise_level conditioning

**The idea:** instead of per-gate counts, condition on a single scalar "noise budget" or categorical label (`noise_level=low/medium/high`). The model learns to balance gate composition to stay within the budget.

**Why this is more interesting than gate counts:**
- Single user-facing knob rather than per-gate counts
- Not reducible to gate set removal
- Captures CX/CCX/SWAP jointly
- Categorical labels (`low/medium/high`) have better CLIP separation than numbers

**Training data analysis:**
- Training set: 2.1M circuits, eff_2q ranging 0–63, mean=15.58, std=10.14
- Wide diversity exists in the data

**Key culprits:**
1. **One circuit per unitary** — the dominant problem. Low-noise circuits correspond to intrinsically simpler unitaries; high-noise circuits to complex ones. The model can't learn "compile this hard unitary at low noise" because it never sees a hard unitary with a low-noise compilation during training.
2. **Complexity confound** — "noise_level=low" would teach the model which unitary types are low-noise, not how to find alternative compilations.
3. **Label imbalance** — only 8.2% of circuits have eff_2q≤3; "low noise" is underrepresented.
4. **What you'd be training** — not "generate a lower-noise circuit for this unitary" but "generate the circuit type typically associated with a low noise budget." The result is a Pareto tradeoff between accuracy and noise, not a true noise-aware compiler.

**The deeper problem:** to train this properly, you'd need multiple compilations of the same unitary at different noise levels. That requires expensive classical circuit optimisation to generate the training data. The model then learns to approximate what the classical optimiser already does — amortised inference, not a novel capability.

---

## 7. The fundamental tension in the unitary compilation task

Classical synthesis tools (Qiskit `qs_decomposition`, `TwoQubitBasisDecomposer`, BQSKit) already solve 3-qubit unitary compilation exactly and fast. The genQC model is less accurate and not meaningfully faster for small circuits.

The natural advantage of a generative model over classical synthesis is at **larger circuit sizes**, where exact classical synthesis becomes exponentially intractable. But dataset generation for larger circuits hits the same exponential wall: the unitary matrix is 2^n × 2^n, making it infeasible to store or compute for n > ~12.

**The SRV task is on stronger ground:** there is no clean classical analogue for generating circuits with a specific Schmidt Rank Vector. Computing SRVs doesn't require the full unitary — just the entanglement structure via tensor network contraction, which is tractable for circuits with limited entanglement.

---

## 8. Gate count conditioning on the SRV task — the more promising direction

**Why the SRV task changes things:**

For the unitary task, the target unitary largely determines the gate composition — conditioning on gate count fights against the compilation constraint.

For the SRV task, many different gate compositions can achieve the same SRV. The SRV specifies entanglement structure but leaves the implementation open. Gate count conditioning is therefore an **independent axis**, not a competing constraint.

**Practical motivation:**
- A quantum algorithm designer knows the entanglement structure needed (from the algorithm's structure — VQE ansätze, communication protocols, etc.)
- They want a hardware-efficient implementation of it
- "Give me a circuit with SRV [1,2,2] using few CX gates" is a concrete, practically relevant request
- No classical tool straightforwardly does this

**Why this is more defensible:**
- Multiple valid circuit implementations per SRV → conditioning provides genuine choice between alternatives
- No classical equivalent for the combined SRV + gate composition constraint
- The "oracle always picks count=0" problem from SWAP conditioning might not apply — for some SRVs, the optimal gate composition is non-trivial

---

## 9. Proposed research question

> *To what extent can text-conditioned diffusion models for quantum circuit generation be improved and extended — through training strategy, encoder choice, and conditioning mechanisms — and where do fundamental limits emerge?*

**How each experiment fits:**

| Experiment | Level | Finding |
|---|---|---|
| Baseline reproduction | Capability baseline | Model works, slightly below paper results |
| Curriculum learning | Training strategy | Stage 2 at 95.3% vs 96.1% baseline — marginal gain; initialisation matters more than curriculum |
| CLOOB encoder | Architecture | Encoder choice has modest effect; doesn't resolve numeric conditioning problem |
| SWAP conditioning | Conditioning mechanism | CLIP too coarse for numeric constraints; fine-tuning strategy matters (warm-start essential) |
| Noise simulation | Hardware relevance | Gate composition has measurable fidelity consequences; SWAP-free circuits have 2× lower effective 2q count |
| CCX analysis | Conditioning limits | Gate set removal already achieves what count conditioning provides for excludable gates |

**The narrative arc:** we systematically examine what can be improved in the genQC pipeline, finding that training strategy matters less than expected (curriculum), encoder choice has modest effects (CLOOB), and the fundamental bottleneck for hardware-relevant conditioning is the text encoder's inability to represent structured numeric constraints — with the exception of CX count, which cannot be addressed by gate set manipulation but is the hardest to train for.

---

## 10. Open directions

- **SRV task + gate count / noise_level conditioning** — most promising, avoids the unitary compilation bottleneck, has no classical equivalent
- **CX count conditioning on unitary task** — genuinely novel but hard to train; uncertain if CLIP can separate cx_count values cleanly
- **Scaling SRV to larger qubit counts** — the natural extension where generative models have a real advantage over classical methods; bottleneck is dataset generation (SRV computation via tensor networks)
