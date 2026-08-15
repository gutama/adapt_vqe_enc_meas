#!/usr/bin/env python3
"""Extract the compact, manuscript-facing summary from a full benchmark JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _shared_entry(entry: dict) -> dict:
    return {
        "metric": entry["metric"],
        "diagnostics": entry["diagnostics"],
    }


def compact_results(full: dict) -> dict:
    out = {
        "environment": full["environment"],
        "stretch": full["stretch"],
        "molecules": [],
    }
    for mol in full["molecules"]:
        m_out = {
            "molecule": mol["molecule"],
            "reference": {"e_casci": mol["reference"]["e_casci"]},
            "mappers": {},
            "mapping_validation": mol["mapping_validation"],
        }
        for mapper in ("JW", "BK"):
            src = mol["mappers"][mapper]
            obs = src["observables"]
            adapt = src["adapt"]
            m_out["mappers"][mapper] = {
                "n_qubits": src["n_qubits"],
                "pool_size": src["pool_size"],
                "adapt": {
                    "energy": adapt["energy"],
                    "exact": adapt["exact"],
                    "n_ops": adapt["n_ops"],
                    "state_norm": adapt["state_norm"],
                    "err_vs_casci_mHa": adapt["err_vs_casci_mHa"],
                },
                "observables": {
                    "pool_generators": {
                        "qwc": {"n_terms": obs["pool_generators"]["qwc"]["n_terms"]}
                    },
                    "gradient_commutators": {
                        "qwc": {
                            key: obs["gradient_commutators"]["qwc"][key]
                            for key in (
                                "n_terms",
                                "n_groups_lf",
                                "n_groups_sorted_insertion",
                                "clique_lower_bound",
                            )
                        }
                    },
                    "hamiltonian": {
                        "qwc": {"n_terms": obs["hamiltonian"]["qwc"]["n_terms"]}
                    },
                },
                "gradient_independent_metric": {
                    key: src["gradient_independent_metric"][key]
                    for key in ("qwc_lf", "qwc_sorted_insertion", "gc_lf")
                },
                "gradient_shared_metric": {
                    key: _shared_entry(src["gradient_shared_metric"][key])
                    for key in ("qwc_lf", "qwc_sorted_insertion", "gc_lf")
                },
            }
        out["molecules"].append(m_out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="results.json")
    parser.add_argument("--out", default="results_summary.json")
    args = parser.parse_args()

    full = json.loads(Path(args.input).read_text())
    summary = compact_results(full)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
