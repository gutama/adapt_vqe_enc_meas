# Validation record

The canonical `results_summary.json` and `qubit_sweep.json` were independently rerun with:

- Python 3.12.3
- Qiskit 2.5.2
- Qiskit Nature 0.8.0
- PySCF 2.14.0
- SciPy 1.17.1
- NumPy 2.4.4

## Algebra and mapping

The JW-to-BK transform reconstructed from mapper Majorana tables is symplectic over GF(2) for the tested register sizes. Molecular Hamiltonian, pool, and gradient-commutator Pauli supports pair exactly under the transform.

For GC grouping, transferring a JW partition to BK preserves the group count exactly. For the gradient supports:

| System | Raw LF JW/BK | Transferred count | Paired `M_ind` JW | Paired `M_ind` BK |
|---|---:|---:|---:|---:|
| LiH | 66 / 67 | 66 | 23.007239 | 23.007239 |
| BeH2 | 68 / 70 | 68 | 28.590065 | 28.590065 |
| H2O | 52 / 53 | 52 | 48.893734 | 48.893734 |

The small raw LF differences arise from label-sensitive greedy ordering; the transferred partitions implement the graph isomorphism directly.

## Deterministic sorted insertion

Pauli coefficient magnitudes used by sorted insertion are rounded to 12 decimal places before ordering. This removes ULP-level process-to-process jitter from duplicate-term summation and gives stable SI group counts:

| System | JW SI groups | BK SI groups |
|---|---:|---:|
| LiH | 441 | 277 |
| BeH2 | 502 | 284 |
| H2O | 540 | 284 |

## QWC certificates

For every 8-qubit gradient support, the BK LF feasible count is below the JW clique lower bound:

| System | JW LF | JW LB | BK LF | BK LB |
|---|---:|---:|---:|---:|
| LiH | 414 | 348 | 266 | 255 |
| BeH2 | 449 | 396 | 276 | 253 |
| H2O | 467 | 420 | 295 | 263 |

The N2 sweep contains two opposite certified comparisons:

- 14 qubits: `BK LB = 215 > JW feasible = 196`.
- 16 qubits: `BK feasible = 299 < JW LB = 338`.

## Shared-shot optimization

The shared-shot solver uses the convex dual in log-parameters and screens variance entries relative to the largest matrix element. Its output is checked against primal constraints and the duality gap.

For LF QWC groups:

| System | Map | `M_ind` | `M_share` | `M_share/M_ind` |
|---|---|---:|---:|---:|
| LiH | JW | 42.79 | 23.72 | 0.55 |
| LiH | BK | 53.78 | 29.53 | 0.55 |
| BeH2 | JW | 68.91 | 25.26 | 0.37 |
| BeH2 | BK | 59.33 | 19.87 | 0.34 |
| H2O | JW | 234.52 | 56.44 | 0.24 |
| H2O | BK | 144.71 | 37.27 | 0.26 |

Across LF and sorted-insertion QWC partitions, `M_ind/M_share` ranges from 1.8x to 5.1x.

The core test suite also cross-checks the dual solver against an independent SLSQP primal solve on a synthetic variance matrix with a wide dynamic range.

## Stored-result validation

A fresh full benchmark is reduced to the manuscript-facing summary with `summarize_results.py`; validation and table generation read that summary.

Run:

```bash
python validate_results.py
```

Expected summary:

```text
NUMERICAL CLAIMS: PASS
Gradient/pool support ratio: 17.28-20.95x
8-qubit BK QWC-LF reduction: 35.7-38.5%
M_ind/M_share over QWC LF+SI: 1.8-5.1x
```
