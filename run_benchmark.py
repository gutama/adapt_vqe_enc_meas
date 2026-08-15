#!/usr/bin/env python3
"""
Reproducible benchmark suite for encoding-dependent measurement grouping in
adaptive variational quantum simulation.

Produces, for each molecule x {JW, BK}:
  (T1) ADAPT-VQE energies vs. active-space CASCI and exact qubit-Hamiltonian
       diagonalisation, with correlation energy recovered.
  (T2) QWC / general-commutativity grouping of three distinct observable
       sets -- the Hamiltonian, the raw pool generators, and the ADAPT
       gradient commutators [H, A_j] -- with a clique lower bound.
  (T3) Variance-aware independent-gradient allocation metric and optional
       shared-shot continuous optimum on the converged ADAPT state.
  (T4) Structural diagnostics and a JW/BK symplectic invariance validation.
  (T5) Propagator comparison: expm_multiply vs Lanczos vs dense vs Taylor.
  (T6) Heuristic-order sensitivity (largest-first vs lexicographic vs random).

Usage:  python run_benchmark.py [--molecules LiH BeH2 H2O] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from adapt import (MAPPERS, MOLECULES, build_problem, casci_reference,
                   constant_shift, hartree_fock_vector, prop_dense,
                   prop_expm_multiply, prop_lanczos, prop_taylor, qubit_pool,
                   run_adapt, susd_pool_fermionic, unique_terms)
from grouping import (gradient_independent_shot_metric,
                      gradient_shared_shot_metric, greedy_clique_lower_bound,
                      greedy_color, setting_reduction, shot_cost,
                      sorted_insertion, structure_stats, support_mask_stats)
from mapping_validation import (mapper_symplectic_transform, support_equivalent,
                           transform_groups)

TOL = 1e-10


# --------------------------------------------------------------------------
def commutator_terms(H_q: SparsePauliOp, pool: list[SparsePauliOp]):
    """Unique Pauli support of the full set of ADAPT gradient observables
    {[H, A_j]}.  This -- not the pool itself -- is what an ADAPT-VQE
    operator-selection step must actually measure."""
    comms, dicts = [], []
    for op in pool:
        # pool ops are stored as i*T (Hermitian); the ADAPT generator is the
        # anti-Hermitian T = -i*(i*T), so the gradient observable is
        # [H, T] = -i [H, i*T], which is Hermitian with real coefficients.
        C = (-1j * (H_q @ op - op @ H_q)).simplify(atol=TOL)
        comms.append(C)
        dicts.append({lab: complex(c).real
                      for lab, c in zip(C.paulis.to_labels(), C.coeffs)
                      if abs(c) > TOL and any(ch != "I" for ch in lab)})
    return unique_terms(comms), dicts


def group_report(labels, weights, relation, psi=None, pmat=None):
    lf = greedy_color(labels, relation=relation, order="lf")
    lex = greedy_color(labels, relation=relation, order="lex")
    si = sorted_insertion(labels, weights, relation=relation)
    rand = [len(greedy_color(labels, relation=relation, order="rand", seed=s))
            for s in range(20)]
    lb = greedy_clique_lower_bound(labels, relation=relation)

    best = min(lf, lex, si, key=len)
    rep = {
        "groups_lf": lf,
        "groups_si": si,
        "n_terms": len(labels),
        "n_groups_lf": len(lf),
        "n_groups_lex": len(lex),
        "n_groups_sorted_insertion": len(si),
        "n_groups_random_min": int(min(rand)),
        "n_groups_random_max": int(max(rand)),
        "n_groups_random_mean": float(np.mean(rand)),
        "clique_lower_bound": lb,
        "reduction_lf_pct": setting_reduction(len(labels), lf),
        "reduction_best_pct": setting_reduction(len(labels), best),
    }
    if psi is not None:
        rep["shot_metric_ungrouped"] = shot_cost(
            [[i] for i in range(len(labels))], labels, weights, psi, pmat)
        rep["shot_metric_lf"] = shot_cost(lf, labels, weights, psi, pmat)
        rep["shot_metric_sorted_insertion"] = shot_cost(si, labels, weights, psi, pmat)
    return rep


def pauli_matrix_cache(labels, n_qubits):
    return {lab: SparsePauliOp(lab).to_matrix(sparse=True).tocsr()
            for lab in labels}


# --------------------------------------------------------------------------
def propagator_study(A_sparse, psi0, thetas=(0.1, 0.5, 1.0, 2.0)):
    ref = {}
    rows = []
    for theta in thetas:
        exact = prop_dense(A_sparse, theta, psi0)
        for name, fn in [("expm_multiply", prop_expm_multiply),
                         ("lanczos_m30", lambda A, t, v: prop_lanczos(A, t, v, m=30)),
                         ("taylor_order4", lambda A, t, v: prop_taylor(A, t, v, 4))]:
            out = fn(A_sparse, theta, psi0)
            rows.append(dict(theta=float(theta), method=name,
                             norm=float(np.linalg.norm(out)),
                             norm_defect=float(abs(np.linalg.norm(out) - 1.0)),
                             err_vs_dense=float(np.linalg.norm(out - exact))))
        rows.append(dict(theta=float(theta), method="dense_expm",
                         norm=float(np.linalg.norm(exact)),
                         norm_defect=float(abs(np.linalg.norm(exact) - 1.0)),
                         err_vs_dense=0.0))
    return rows


# --------------------------------------------------------------------------
def benchmark_molecule(name, stretch=2.0, max_iter=12, verbose=True,
                       shared_shot_metric=False):
    spec = MOLECULES[name]
    problem, raw, geom, _ = build_problem(name, stretch)
    ref = casci_reference(name, stretch)
    out = {"molecule": name, "geometry": geom, "reference": ref,
           "point_group": spec["point_group"], "mappers": {}}
    validation_cache = {}

    if verbose:
        gtxt = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in geom.items())
        print(f"\n=== {name}  ({gtxt}) ===")
        print(f"    E_HF = {ref['e_hf']:.6f}   E_CASCI = {ref['e_casci']:.6f}"
              f"   E_FCI(full) = {ref['e_fci_full']:.6f}")

    for mname in ("JW", "BK"):
        t0 = time.time()
        mapper = MAPPERS[mname]()
        res = run_adapt(problem, mname, spec["occ"], spec["virt"],
                        max_iter=max_iter, verbose=verbose)
        H_q, pool, psi = res["H_qubit_op"], res["pool"], res["psi"]
        n_qubits = H_q.num_qubits

        # --- observable sets -------------------------------------------------
        h_labels, h_w = unique_terms([H_q])
        p_labels, p_w = unique_terms(pool)
        (c_labels, c_w), comm_dicts = commutator_terms(H_q, pool)

        pm_h = pauli_matrix_cache(h_labels, n_qubits)
        pm_p = pauli_matrix_cache(p_labels, n_qubits)

        entry = {
            "n_qubits": n_qubits,
            "pool_size": len(pool),
            "adapt": {k: res[k] for k in
                      ("energy", "exact", "n_ops", "state_norm")},
            "history": res["history"],
            "observables": {
                "hamiltonian": {
                    "qwc": group_report(h_labels, h_w, "qwc", psi, pm_h),
                    "gc": group_report(h_labels, h_w, "gc", psi, pm_h),
                    "structure": structure_stats(h_labels, "qwc"),
                    "masks": support_mask_stats(h_labels),
                },
                "pool_generators": {
                    "qwc": group_report(p_labels, p_w, "qwc", psi, pm_p),
                    "gc": group_report(p_labels, p_w, "gc", psi, pm_p),
                    "structure": structure_stats(p_labels, "qwc"),
                    "masks": support_mask_stats(p_labels),
                },
                "gradient_commutators": {
                    "qwc": group_report(c_labels, c_w, "qwc"),
                    "gc": group_report(c_labels, c_w, "gc"),
                    "structure": structure_stats(c_labels, "qwc"),
                    "masks": support_mask_stats(c_labels),
                },
            },
            "wall_time_s": time.time() - t0,
        }

        # ---- variance-aware ADAPT gradient metrics --------------------------
        # This metric sums independently optimized costs for each
        # gradient component.  It is useful but is *not* the exact minimum
        # shots for the whole vector because the same measurement record can
        # be reused by several commutators.
        pm_c = pauli_matrix_cache(c_labels, n_qubits)
        gc_ent = entry["observables"]["gradient_commutators"]
        ungrouped = [[i] for i in range(len(c_labels))]
        tot_u, _ = gradient_independent_shot_metric(
            ungrouped, c_labels, comm_dicts, psi, pm_c)
        tot_lf, _ = gradient_independent_shot_metric(
            gc_ent["qwc"]["groups_lf"], c_labels, comm_dicts, psi, pm_c)
        tot_si, _ = gradient_independent_shot_metric(
            gc_ent["qwc"]["groups_si"], c_labels, comm_dicts, psi, pm_c)
        tot_gc, _ = gradient_independent_shot_metric(
            gc_ent["gc"]["groups_lf"], c_labels, comm_dicts, psi, pm_c)
        entry["gradient_independent_metric"] = {
            "ungrouped": tot_u, "qwc_lf": tot_lf,
            "qwc_sorted_insertion": tot_si, "gc_lf": tot_gc,
            "reduction_qwc_lf_pct": 100.0 * (1 - tot_lf / tot_u),
            "reduction_gc_lf_pct": 100.0 * (1 - tot_gc / tot_u),
            "interpretation": "sum of per-gradient independently optimized variance metrics",
        }

        if shared_shot_metric:
            shared = {}
            for tag, groups in [
                ("ungrouped", ungrouped),
                ("qwc_lf", gc_ent["qwc"]["groups_lf"]),
                ("qwc_sorted_insertion", gc_ent["qwc"]["groups_si"]),
                ("gc_lf", gc_ent["gc"]["groups_lf"]),
            ]:
                metric, alloc, diag = gradient_shared_shot_metric(
                    groups, c_labels, comm_dicts, psi, pm_c)
                shared[tag] = {"metric": metric, "allocation": alloc,
                               "diagnostics": diag}
            entry["gradient_shared_metric"] = shared

        # correlation energy recovered, active space
        e_hf, e_cas = ref["e_hf"], ref["e_casci"]
        denom = e_hf - e_cas
        entry["adapt"]["corr_recovered_pct"] = (
            100.0 * (e_hf - res["energy"]) / denom if abs(denom) > 1e-12 else float("nan"))
        entry["adapt"]["err_vs_casci_mHa"] = 1e3 * (res["energy"] - e_cas)
        entry["adapt"]["err_vs_exact_mHa"] = 1e3 * (res["energy"] - res["exact"])

        # propagator study on the largest-gradient generator
        A0 = (-1j * pool[0].to_matrix(sparse=True)).tocsr()
        psi_hf = hartree_fock_vector(problem, mapper, n_qubits)
        entry["propagator_study"] = propagator_study(A0, psi_hf)

        out["mappers"][mname] = entry
        validation_cache[mname] = {
            "hamiltonian": h_labels,
            "pool_generators": p_labels,
            "gradient_commutators": c_labels,
            "comm_dicts": comm_dicts,
            "psi": psi,
            "pm_c": pm_c,
        }

        if verbose:
            q = entry["observables"]
            print(f"  [{mname}] pool={len(pool)}  "
                  f"E_ADAPT={res['energy']:.6f}  "
                  f"dE(CASCI)={entry['adapt']['err_vs_casci_mHa']:+.3f} mHa  "
                  f"corr={entry['adapt']['corr_recovered_pct']:.2f}%")
            for key in ("hamiltonian", "pool_generators", "gradient_commutators"):
                r = q[key]["qwc"]; rg = q[key]["gc"]
                print(f"        {key:22s} N={r['n_terms']:5d}  "
                      f"QWC_lf={r['n_groups_lf']:4d} (lb {r['clique_lower_bound']:3d}) "
                      f"QWC_rand=[{r['n_groups_random_min']}-{r['n_groups_random_max']}]  "
                      f"GC_lf={rg['n_groups_lf']:4d}  "
                      f"masks={q[key]['masks']['n_distinct_support_masks']}")
            gm = entry["gradient_independent_metric"]
            print(f"        independent gradient metric: ungrouped={gm['ungrouped']:.3f} "
                  f"QWC-LF={gm['qwc_lf']:.3f} ({gm['reduction_qwc_lf_pct']:.1f}%) "
                  f"SI={gm['qwc_sorted_insertion']:.3f} "
                  f"GC={gm['gc_lf']:.3f} ({gm['reduction_gc_lf_pct']:.1f}%)")

    # ---- cross-encoding invariance validation -------------------
    jw, bk = out["mappers"]["JW"], out["mappers"]["BK"]
    if abs(jw["adapt"]["energy"] - bk["adapt"]["energy"]) > 1e-10:
        raise AssertionError("JW/BK ADAPT energies are not representation invariant")
    if abs(jw["adapt"]["exact"] - bk["adapt"]["exact"]) > 1e-10:
        raise AssertionError("JW/BK exact energies are not representation invariant")
    if abs(jw["adapt"]["exact"] - ref["e_casci"]) > 1e-8:
        raise AssertionError("exact active-space qubit energy disagrees with CASCI")

    n_qubits = jw["n_qubits"]
    transform = mapper_symplectic_transform(
        MAPPERS["JW"](), MAPPERS["BK"](), n_qubits)
    mapping_validation = {"symplectic": True, "support_equivalence": {},
                     "paired_gc": {}}
    for key in ("hamiltonian", "pool_generators", "gradient_commutators"):
        ok = support_equivalent(validation_cache["JW"][key],
                                validation_cache["BK"][key], transform)
        mapping_validation["support_equivalence"][key] = bool(ok)
        if not ok:
            raise AssertionError(f"JW/BK Pauli support mismatch for {key}")

        jw_groups = jw["observables"][key]["gc"]["groups_lf"]
        bk_paired = transform_groups(
            jw_groups, validation_cache["JW"][key], validation_cache["BK"][key], transform)
        mapping_validation["paired_gc"][key] = {
            "n_groups_jw": len(jw_groups),
            "n_groups_bk_paired": len(bk_paired),
            "raw_label_tiebreak_bk": bk["observables"][key]["gc"]["n_groups_lf"],
        }

        if key == "gradient_commutators":
            paired_bk_metric, _ = gradient_independent_shot_metric(
                bk_paired, validation_cache["BK"][key], validation_cache["BK"]["comm_dicts"],
                validation_cache["BK"]["psi"], validation_cache["BK"]["pm_c"])
            jw_metric = jw["gradient_independent_metric"]["gc_lf"]
            mapping_validation["paired_gc"][key]["independent_metric_jw"] = jw_metric
            mapping_validation["paired_gc"][key]["independent_metric_bk_paired"] = paired_bk_metric
            if abs(jw_metric - paired_bk_metric) > 1e-8 * max(1.0, abs(jw_metric)):
                raise AssertionError("paired GC variance metric is not encoding invariant")

    out["mapping_validation"] = mapping_validation
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecules", nargs="+", default=["LiH", "BeH2", "H2O"])
    ap.add_argument("--stretch", type=float, default=2.0)
    ap.add_argument("--max-iter", type=int, default=12)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--shared-shot-metric", action="store_true",
                    help="also solve the shared-shot convex allocation for the gradient vector")
    args = ap.parse_args()

    import qiskit, qiskit_nature, pyscf, scipy
    env = dict(python=platform.python_version(), qiskit=qiskit.__version__,
               qiskit_nature=qiskit_nature.__version__,
               pyscf=pyscf.__version__, scipy=scipy.__version__,
               numpy=np.__version__, platform=platform.platform())
    print("Environment:", json.dumps(env, indent=None))

    results = {"environment": env, "stretch": args.stretch, "molecules": []}
    for name in args.molecules:
        results["molecules"].append(
            benchmark_molecule(name, args.stretch, args.max_iter,
                               shared_shot_metric=args.shared_shot_metric))

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()
                    if k not in ("groups_lf", "groups_si")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    results = strip(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
