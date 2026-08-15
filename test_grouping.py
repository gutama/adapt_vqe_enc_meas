#!/usr/bin/env python3
"""Core regression tests for grouping, GF(2) mapping, and shot allocation."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import grouping as g


def _random_labels(rng, n_terms, n_qubits):
    alphabet = np.array(list("IXYZ"))
    out = []
    while len(out) < n_terms:
        lab = "".join(rng.choice(alphabet, size=n_qubits))
        if lab != "I" * n_qubits and lab not in out:
            out.append(lab)
    return out


def test_known_relations():
    labs = ["XX", "YY", "XI"]
    X, Z = g.pack(labs)
    qwc = g.qwc_matrix(X, Z)
    gc = g.gc_matrix(X, Z)
    assert not qwc[0, 1]
    assert gc[0, 1]
    assert qwc[0, 2] and gc[0, 2]


def test_dense_scalable_equivalence():
    rng = np.random.default_rng(7)
    for n_qubits in (2, 4, 7):
        labels = _random_labels(rng, min(48, 4 ** n_qubits - 1), n_qubits)
        weights = rng.normal(size=len(labels))
        for rel in ("qwc", "gc"):
            for order in ("lf", "lex"):
                assert len(g._greedy_color_dense(labels, rel, order)) == len(
                    g.greedy_color(labels, rel, order)
                )
            assert len(g._sorted_insertion_dense(labels, weights, rel)) == len(
                g.sorted_insertion(labels, weights, rel)
            )
            for grp in g.greedy_color(labels, rel, "lf"):
                X, Z = g.pack([labels[i] for i in grp])
                assert g.RELATIONS[rel](X, Z).all()


def test_determinism_and_validation():
    labs = ["XI", "IX", "XX", "ZI", "IZ", "ZZ", "YY"]
    assert g.greedy_color(labs, "qwc", "lf") == g.greedy_color(labs, "qwc", "lf")
    assert g.greedy_color(labs, "gc", "lex") == g.greedy_color(labs, "gc", "lex")
    for bad in ([], ["XI", "XYZ"]):
        try:
            g.pack(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid labels accepted: {bad}")


def test_shared_shot_reuse():
    labels = ["Z"]
    groups = [[0]]
    psi = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    comms = [{"Z": 1.0}, {"Z": 2.0}]
    independent, per_component = g.gradient_independent_shot_metric(
        groups, labels, comms, psi, {"Z": Z}
    )
    shared, allocation, diag = g.gradient_shared_shot_metric(
        groups, labels, comms, psi, {"Z": Z}
    )
    assert np.allclose(per_component, [1.0, 4.0], atol=1e-8)
    assert abs(independent - 5.0) < 1e-8
    assert abs(shared - 4.0) < 2e-6
    assert len(allocation) == 1
    assert diag["max_constraint"] <= 1.0 + 5e-6
    assert diag["relative_duality_gap"] <= 5e-6


def test_shared_solver_wide_dynamic_range():
    # Includes numerical-scale entries to exercise relative screening and the
    # log-dual parameterization used by the benchmark solver.
    V = np.array([
        [1.0, 1e-28, 0.2, 0.0, 2.0],
        [0.0, 2e-24, 1.5, 0.4, 0.1],
        [0.7, 0.0, 0.0, 2.3, 0.8],
        [1e-34, 0.6, 0.3, 0.2, 0.0],
    ])
    metric, n, diag = g.solve_shared_allocation(V)
    assert metric > 0
    assert diag["max_constraint"] <= 1.0 + 1e-5
    assert diag["relative_duality_gap"] <= 1e-5

    # Independent primal cross-check with SLSQP on the screened matrix.
    Vs = V.copy()
    Vs[Vs < 1e-10 * Vs.max()] = 0.0
    keep = Vs.max(axis=0) > 0
    Vs = Vs[:, keep]
    Vs = Vs[Vs.max(axis=1) > 0]

    def obj(x):
        return float(x.sum())

    cons = [
        {"type": "ineq", "fun": lambda x, row=row: 1.0 - np.sum(row / x)}
        for row in Vs
    ]
    x0 = np.full(Vs.shape[1], max(10.0, 2.0 * Vs.sum()))
    ref = minimize(obj, x0, method="SLSQP", bounds=[(1e-12, None)] * len(x0),
                   constraints=cons, options={"ftol": 1e-12, "maxiter": 20000})
    assert ref.success, ref.message
    assert abs(metric - ref.fun) / max(1.0, ref.fun) < 2e-7


def test_mapping_validation_core():
    from mapping_validation import gf2_inverse, is_symplectic, symplectic_form

    rng = np.random.default_rng(11)
    for n in (1, 2, 5):
        while True:
            A = rng.integers(0, 2, size=(n, n), dtype=np.uint8)
            try:
                Ainv = gf2_inverse(A)
                break
            except ValueError:
                continue
        T = np.block([
            [A, np.zeros((n, n), dtype=np.uint8)],
            [np.zeros((n, n), dtype=np.uint8), Ainv.T],
        ])
        assert is_symplectic(T)
        J = symplectic_form(n)
        assert np.array_equal((T @ J @ T.T) & 1, J)


def test_chemistry_integration_optional():
    try:
        from adapt import (MAPPERS, build_problem, qubit_pool,
                           susd_pool_fermionic, unique_terms,
                           prop_dense, prop_expm_multiply, prop_lanczos)
    except ModuleNotFoundError as exc:
        print(f"  SKIP chemistry integration ({exc.name} not installed)")
        return

    problem, _, _, _ = build_problem("LiH", 2.0)
    mapper = MAPPERS["JW"]()
    H = mapper.map(problem.second_q_ops()[0]).simplify()
    pool = qubit_pool(susd_pool_fermionic([0], [1, 2, 3], 4), mapper)
    comms = [(-1j * (H @ o - o @ H)).simplify(atol=1e-10) for o in pool]
    for tag, ops in (("H", [H]), ("pool", pool), ("grad", comms)):
        labs, w = unique_terms(ops)
        assert np.array_equal(w, np.round(w, 12))
        for rel in ("qwc", "gc"):
            assert len(g._greedy_color_dense(labs, rel, "lf")) == len(g.greedy_color(labs, rel, "lf"))
            assert len(g._sorted_insertion_dense(labs, w, rel)) == len(g.sorted_insertion(labs, w, rel))
        print(f"  ok chemistry {tag:5s} N={len(labs)}")

    A = (-1j * pool[0].to_matrix(sparse=True)).tocsr()
    rng = np.random.default_rng(0)
    v = rng.normal(size=A.shape[0]) + 1j * rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    for th in (0.1, 1.0, 3.0):
        ref = prop_dense(A, th, v)
        for name, out in (("expm_multiply", prop_expm_multiply(A, th, v)),
                          ("lanczos", prop_lanczos(A, th, v, m=40))):
            err = np.linalg.norm(out - ref)
            nd = abs(np.linalg.norm(out) - 1.0)
            assert err < 1e-12 and nd < 1e-12, (name, th, err, nd)


def main():
    tests = [
        test_known_relations,
        test_dense_scalable_equivalence,
        test_determinism_and_validation,
        test_shared_shot_reuse,
        test_shared_solver_wide_dynamic_range,
        test_mapping_validation_core,
        test_chemistry_integration_optional,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()
        print("  PASS")
    print("\nALL AVAILABLE TESTS PASSED")


if __name__ == "__main__":
    main()
