#!/usr/bin/env python3
"""Validate machine-readable numerical claims used by the manuscript."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    results = json.loads((ROOT / "results_summary.json").read_text())
    sweep = json.loads((ROOT / "qubit_sweep.json").read_text())["N2"]

    assert results["environment"]["python"] == "3.12.3"
    assert results["environment"]["qiskit"] == "2.5.2"
    assert results["environment"]["qiskit_nature"] == "0.8.0"
    assert results["environment"]["pyscf"] == "2.14.0"
    assert results["environment"]["scipy"] == "1.17.1"
    assert results["environment"]["numpy"] == "2.4.4"

    support_ratios = []
    qwc_reductions = []
    shared_ratios = []

    for mol in results["molecules"]:
        name = mol["molecule"]
        jw, bk = mol["mappers"]["JW"], mol["mappers"]["BK"]

        assert abs(jw["adapt"]["energy"] - bk["adapt"]["energy"]) < 1e-10, name
        assert abs(jw["adapt"]["exact"] - bk["adapt"]["exact"]) < 1e-10, name
        assert abs(jw["adapt"]["exact"] - mol["reference"]["e_casci"]) < 1e-8, name
        assert abs(jw["adapt"]["state_norm"] - 1.0) < 1e-10, name
        assert abs(bk["adapt"]["state_norm"] - 1.0) < 1e-10, name

        for key in ("hamiltonian", "pool_generators", "gradient_commutators"):
            jn = jw["observables"][key]["qwc"]["n_terms"]
            bn = bk["observables"][key]["qwc"]["n_terms"]
            assert jn == bn, (name, key, jn, bn)

        pool_terms = jw["observables"]["pool_generators"]["qwc"]["n_terms"]
        grad_terms = jw["observables"]["gradient_commutators"]["qwc"]["n_terms"]
        support_ratios.append(grad_terms / pool_terms)

        jq = jw["observables"]["gradient_commutators"]["qwc"]
        bq = bk["observables"]["gradient_commutators"]["qwc"]
        assert bq["n_groups_lf"] < jq["clique_lower_bound"], name
        qwc_reductions.append(100.0 * (1.0 - bq["n_groups_lf"] / jq["n_groups_lf"]))

        for mp in (jw, bk):
            mi = mp["gradient_independent_metric"]
            ms = mp["gradient_shared_metric"]
            for tag in ("qwc_lf", "qwc_sorted_insertion", "gc_lf"):
                assert ms[tag]["metric"] <= mi[tag] + 1e-8
                diag = ms[tag]["diagnostics"]
                assert diag["max_constraint"] <= 1.0 + 1e-5
                assert diag["relative_duality_gap"] <= 1e-5
                if tag.startswith("qwc"):
                    shared_ratios.append(mi[tag] / ms[tag]["metric"])

        mv = mol["mapping_validation"]
        assert mv["symplectic"] is True
        assert all(mv["support_equivalence"].values())
        pg = mv["paired_gc"]["gradient_commutators"]
        assert pg["n_groups_jw"] == pg["n_groups_bk_paired"]
        denom = max(1.0, abs(pg["independent_metric_jw"]))
        assert abs(pg["independent_metric_jw"] - pg["independent_metric_bk_paired"]) / denom < 1e-8

    assert 17.2 < min(support_ratios) < 17.4
    assert 20.9 < max(support_ratios) < 21.1
    assert 35.6 < min(qwc_reductions) < 35.8
    assert 38.4 < max(qwc_reductions) < 38.6
    assert 1.7 < min(shared_ratios) < 1.9
    assert 5.0 < max(shared_ratios) < 5.2

    rows = {row["n_qubits"]: row for row in sweep}
    assert rows[14]["BK"]["qwc_lb"] > rows[14]["JW"]["qwc"]
    assert rows[16]["BK"]["qwc"] < rows[16]["JW"]["qwc_lb"]
    assert rows[8]["BK"]["mean_weight"] > rows[8]["JW"]["mean_weight"]
    assert rows[8]["BK"]["qwc"] < rows[8]["JW"]["qwc"]
    assert rows[14]["BK"]["max_weight"] < rows[14]["JW"]["max_weight"]
    assert rows[14]["BK"]["qwc_lb"] > rows[14]["JW"]["qwc"]

    print("NUMERICAL CLAIMS: PASS")
    print(f"Gradient/pool support ratio: {min(support_ratios):.2f}-{max(support_ratios):.2f}x")
    print(f"8-qubit BK QWC-LF reduction: {min(qwc_reductions):.1f}-{max(qwc_reductions):.1f}%")
    print(f"M_ind/M_share over QWC LF+SI: {min(shared_ratios):.1f}-{max(shared_ratios):.1f}x")


if __name__ == "__main__":
    main()
