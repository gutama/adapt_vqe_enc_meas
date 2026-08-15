#!/usr/bin/env python3
"""
Active-space scaling study.

For a sequence of active spaces (8 -> 14 qubits) this measures, for both
mappings:
  * peak resident set size of the sparse ADAPT pipeline (tracemalloc +
    /proc/self/status VmHWM),
  * the dense storage that the same operators would require,
  * unique Pauli support and QWC group counts for the Hamiltonian and
    ADAPT gradient commutators;
  * raw label-tiebreak GC greedy counts as diagnostics only.  GC is invariant
    under the JW/BK Clifford transform and these raw heuristic counts must not
    be interpreted as mapping-dependent resources.

Run:  python run_scaling.py --out scaling.json
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import time

import numpy as np

from adapt import (MAPPERS, MOLECULES, build_problem, constant_shift,
                   hartree_fock_vector, prop_expm_multiply, qubit_pool,
                   susd_pool_fermionic, unique_terms)
from grouping import greedy_color, structure_stats

CONFIGS = {
    "a": [("LiH", (1, 1), 4), ("BeH2", (2, 2), 4)],
    "b": [("LiH", (2, 2), 5), ("BeH2", (3, 3), 5)],
    "c": [("LiH", (2, 2), 6), ("BeH2", (3, 3), 6)],
    "d": [("BeH2", (3, 3), 7)],
}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def one_config(name, n_elec, n_orb, mapper_name, do_commutators=True):
    spec = dict(MOLECULES[name])
    spec["n_elec"], spec["n_orb"] = n_elec, n_orb
    n_occ = n_elec[0]
    occ = list(range(n_occ))
    virt = list(range(n_occ, n_orb))

    saved = MOLECULES[name].copy()
    MOLECULES[name].update(n_elec=n_elec, n_orb=n_orb)
    try:
        problem, _, geom, _ = build_problem(name, 2.0)
        mapper = MAPPERS[mapper_name]()
        H_q = mapper.map(problem.second_q_ops()[0]).simplify()
        n_qubits = H_q.num_qubits
        dim = 2 ** n_qubits

        t0 = time.time()
        H = H_q.to_matrix(sparse=True).tocsr()
        pool = qubit_pool(susd_pool_fermionic(occ, virt,
                                              problem.num_spatial_orbitals), mapper)
        A_sparse = [(-1j * op.to_matrix(sparse=True)).tocsr() for op in pool]
        psi = hartree_fock_vector(problem, mapper, n_qubits)

        # one exact sparse propagation step per generator, no dense matrices
        for A in A_sparse:
            psi = prop_expm_multiply(A, 0.05, psi)
        norm = float(np.linalg.norm(psi))

        sparse_bytes = (H.data.nbytes + H.indices.nbytes + H.indptr.nbytes
                        + sum(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes
                              for A in A_sparse))
        dense_bytes = (1 + len(A_sparse)) * dim * dim * 16   # complex128

        h_labels, _ = unique_terms([H_q])
        rec = dict(molecule=name, n_elec=list(n_elec), n_orb=n_orb,
                   mapper=mapper_name, n_qubits=n_qubits, dim=dim,
                   pool_size=len(pool), state_norm=norm,
                   H_terms=len(h_labels),
                   H_qwc_groups=len(greedy_color(h_labels, "qwc", "lf")),
                   H_gc_raw_lf=len(greedy_color(h_labels, "gc", "lf")),
                   H_structure=structure_stats(h_labels, "qwc"),
                   sparse_MB=sparse_bytes / 2**20,
                   dense_equivalent_MB=dense_bytes / 2**20,
                   ratio=dense_bytes / max(sparse_bytes, 1),
                   peak_rss_MB=peak_rss_mb(),
                   wall_time_s=time.time() - t0)

        if do_commutators and n_qubits <= 14:
            comms = [(-1j * (H_q @ op - op @ H_q)).simplify(atol=1e-10)
                     for op in pool]
            c_labels, _ = unique_terms(comms)
            rec["grad_terms"] = len(c_labels)
            rec["grad_qwc_groups"] = len(greedy_color(c_labels, "qwc", "lf"))
            rec["grad_gc_raw_lf"] = len(greedy_color(c_labels, "gc", "lf"))
        return rec
    finally:
        MOLECULES[name].update(saved)
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scaling.json")
    ap.add_argument("--chunk", default="a")
    ap.add_argument("--no-commutators", action="store_true")
    args = ap.parse_args()

    rows = []
    print(f"{'system':22s} {'map':4s} {'nq':>3s} {'pool':>5s} {'H':>5s} "
          f"{'Hqwc':>5s} {'HGCr':>4s} {'grad':>6s} {'gqwc':>5s} {'gGCr':>4s} "
          f"{'sparse/MB':>10s} {'dense/MB':>10s} {'ratio':>7s} {'RSS/MB':>8s}")
    for name, ne, no in CONFIGS[args.chunk]:
        for mp in ("JW", "BK"):
            r = one_config(name, ne, no, mp,
                           do_commutators=not args.no_commutators)
            rows.append(r)
            tag = f"{name}({sum(ne)}e,{no}o)"
            print(f"{tag:22s} {mp:4s} {r['n_qubits']:3d} {r['pool_size']:5d} "
                  f"{r['H_terms']:5d} {r['H_qwc_groups']:5d} {r['H_gc_raw_lf']:4d} "
                  f"{r.get('grad_terms',0):6d} {r.get('grad_qwc_groups',0):5d} "
                  f"{r.get('grad_gc_raw_lf',0):4d} {r['sparse_MB']:10.3f} "
                  f"{r['dense_equivalent_MB']:10.1f} {r['ratio']:7.1f} "
                  f"{r['peak_rss_MB']:8.1f}")

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
