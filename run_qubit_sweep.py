#!/usr/bin/env python3
"""Hamiltonian-only QWC sweep over active-space size.

The sweep tests how Jordan--Wigner (JW) and Bravyi--Kitaev (BK) change
qubit-wise commuting (QWC) partitions as the register grows.  General/full
Pauli commutation (GC) is *not* treated as an encoding-dependent resource:
JW and BK are related by an invertible Clifford/symplectic basis change, so
their GC conflict graphs are isomorphic.  We therefore validate the Pauli
bijection and carry one paired GC partition across mappings instead of
comparing label-dependent greedy tie-breaks.
"""
from __future__ import annotations

import argparse
import json

from adapt import MAPPERS, MOLECULES, build_problem, unique_terms
from grouping import greedy_color, greedy_clique_lower_bound, structure_stats
from mapping_validation import (mapper_symplectic_transform, support_equivalent,
                           transform_groups)


def sweep(name, elec_fn, orb_list, with_lb=True):
    rows = []
    saved = MOLECULES[name].copy()
    try:
        for no in orb_list:
            n_elec = elec_fn(no)
            MOLECULES[name].update(n_elec=n_elec, n_orb=no)
            problem, _, _, _ = build_problem(name, 2.0)
            rec = {"molecule": name, "n_orb": no, "n_qubits": 2 * no,
                   "n_elec": list(n_elec)}
            labels = {}
            gc_groups = {}
            for mp in ("JW", "BK"):
                H = MAPPERS[mp]().map(problem.second_q_ops()[0]).simplify()
                labs, _ = unique_terms([H])
                labels[mp] = labs
                st = structure_stats(labs, "qwc")
                gc_groups[mp] = greedy_color(labs, "gc", "lf")
                rec[mp] = {
                    "n_terms": len(labs),
                    "qwc": len(greedy_color(labs, "qwc", "lf")),
                    "gc_raw_label_tiebreak": len(gc_groups[mp]),
                    "mean_weight": st["mean_weight"],
                    "max_weight": st["max_weight"],
                    "compat_density": st["compat_density"],
                }
                if with_lb:
                    rec[mp]["qwc_lb"] = greedy_clique_lower_bound(labs, "qwc")

            T = mapper_symplectic_transform(MAPPERS["JW"](), MAPPERS["BK"](), 2 * no)
            if not support_equivalent(labels["JW"], labels["BK"], T):
                raise AssertionError("JW/BK Hamiltonian Pauli supports are not symplectically paired")
            paired_bk = transform_groups(gc_groups["JW"], labels["JW"], labels["BK"], T)
            rec["gc_validation"] = {
                "symplectic": True,
                "support_equivalent": True,
                "paired_group_count": len(paired_bk),
                "jw_raw_label_tiebreak": len(gc_groups["JW"]),
                "bk_raw_label_tiebreak": len(gc_groups["BK"]),
                "interpretation": (
                    "paired GC partition is encoding invariant; differing raw counts, if any, "
                    "are heuristic tie-break artifacts"
                ),
            }
            rec["qwc_ratio_BK_over_JW"] = rec["BK"]["qwc"] / rec["JW"]["qwc"]
            rows.append(rec)
            print(f"  N={2*no:3d} q  terms={rec['JW']['n_terms']:5d}  "
                  f"QWC: JW={rec['JW']['qwc']:4d} BK={rec['BK']['qwc']:4d} "
                  f"(BK/JW={rec['qwc_ratio_BK_over_JW']:.3f})   "
                  f"GC paired={rec['gc_validation']['paired_group_count']:3d} "
                  f"(raw JW/BK={rec['gc_validation']['jw_raw_label_tiebreak']}/"
                  f"{rec['gc_validation']['bk_raw_label_tiebreak']})   "
                  f"maxwt {rec['JW']['max_weight']}/{rec['BK']['max_weight']}  "
                  f"meanwt {rec['JW']['mean_weight']:.2f}/{rec['BK']['mean_weight']:.2f}")
    finally:
        MOLECULES[name].update(saved)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orbs", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--out", default="qubit_sweep.json")
    args = ap.parse_args()
    out = {}
    print("N2, STO-3G, 2 x Re:")
    out["N2"] = sweep("N2", lambda no: (no - 3, no - 3), args.orbs)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("Wrote", args.out)
