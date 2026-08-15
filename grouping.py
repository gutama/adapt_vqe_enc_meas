"""
Deterministic Pauli-string extraction, commutativity relations and
measurement-grouping heuristics.

Everything here is order-deterministic: no reliance on Python ``set``
iteration order (which is salted by ``PYTHONHASHSEED`` and therefore
makes clique counts irreproducible from run to run).

Relations
---------
qwc   : qubit-wise commutativity (single-qubit / tensor-product basis)
gc    : general (full) commutativity  -> requires Clifford diagonalisation

Heuristics
----------
greedy_color(order="lf")   : largest-first greedy graph colouring
greedy_color(order="lex")  : lexicographic greedy (deterministic baseline)
greedy_color(order="rand") : random order (used only to expose variance)
sorted_insertion           : Crawford et al., Quantum 5, 385 (2021)

Lower bound
-----------
greedy_clique_lower_bound  : a clique in the conflict graph gives a lower
                             bound on the chromatic number, i.e. on the
                             minimum number of measurement settings.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Symplectic (x|z) representation
# --------------------------------------------------------------------------
_XZ = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def label_to_xz(label: str) -> tuple[np.ndarray, np.ndarray]:
    """Qiskit labels are printed most-significant-qubit first; we keep that
    order consistently, which is irrelevant for commutativity."""
    x = np.fromiter((_XZ[c][0] for c in label), dtype=np.uint8, count=len(label))
    z = np.fromiter((_XZ[c][1] for c in label), dtype=np.uint8, count=len(label))
    return x, z


def pack(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Stack labels into ``(n_terms, n_qubits)`` x/z bit matrices.

    Empty observable sets are rejected explicitly instead of failing later with
    an opaque ``IndexError``.  Chemistry paths should normally never be empty,
    but this guard makes the grouping API safer for filtered operator sets.
    """
    if not labels:
        raise ValueError("cannot pack an empty Pauli-label set")
    n = len(labels[0])
    if any(len(lab) != n for lab in labels):
        raise ValueError("all Pauli labels must have the same register length")
    X = np.zeros((len(labels), n), dtype=np.uint8)
    Z = np.zeros((len(labels), n), dtype=np.uint8)
    for i, lab in enumerate(labels):
        X[i], Z[i] = label_to_xz(lab)
    return X, Z


# --------------------------------------------------------------------------
# Commutativity relations, vectorised
# --------------------------------------------------------------------------
def qwc_matrix(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Boolean (n_terms, n_terms) matrix, True where the pair is QWC.

    Two single-qubit Paulis commute qubit-wise iff they are equal or at
    least one of them is the identity.  In (x|z) form: identity is
    (0,0), so the condition ``not (support_i & support_j & different)``.
    """
    supp = (X | Z).astype(bool)                       # (m, n)
    same = (X[:, None, :] == X[None, :, :]) & (Z[:, None, :] == Z[None, :, :])
    both = supp[:, None, :] & supp[None, :, :]
    conflict = both & ~same
    return ~conflict.any(axis=2)


def gc_matrix(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Boolean matrix, True where the pair commutes in the full
    (general-commutativity) sense: symplectic product x_i.z_j + z_i.x_j = 0 mod 2."""
    s = (X.astype(np.int64) @ Z.T.astype(np.int64)
         + Z.astype(np.int64) @ X.T.astype(np.int64)) % 2
    return s == 0


RELATIONS = {"qwc": qwc_matrix, "gc": gc_matrix}


# --------------------------------------------------------------------------
# Grouping heuristics
# --------------------------------------------------------------------------
def greedy_color(labels: list[str], relation: str = "qwc",
                 order: str = "lf", seed: int | None = None) -> list[list[int]]:
    """Greedy clique cover of the compatibility graph (= greedy colouring of
    the conflict graph).  Returns a list of groups, each a list of indices
    into ``labels``.

    ``order``:
      "lf"   largest-first: descending conflict degree, ties broken
             lexicographically -> fully deterministic.
      "lex"  lexicographic order of the Pauli label.
      "rand" random permutation (only to quantify heuristic variance).
    """
    X, Z = pack(labels)
    compat = RELATIONS[relation](X, Z)
    m = len(labels)

    if order == "lf":
        degree = (~compat).sum(axis=1)
        seq = sorted(range(m), key=lambda i: (-int(degree[i]), labels[i]))
    elif order == "lex":
        seq = sorted(range(m), key=lambda i: labels[i])
    elif order == "rand":
        rng = np.random.default_rng(seed)
        seq = list(rng.permutation(m))
    else:
        raise ValueError(order)

    groups: list[list[int]] = []
    for i in seq:
        for g in groups:
            if compat[i, g].all():
                g.append(int(i))
                break
        else:
            groups.append([int(i)])
    return groups


def sorted_insertion(labels: list[str], weights: np.ndarray,
                     relation: str = "qwc") -> list[list[int]]:
    """Crawford et al. sorted-insertion: process terms in descending |coeff|
    and place each in the first compatible group.  Ties broken
    lexicographically so the result is deterministic."""
    X, Z = pack(labels)
    compat = RELATIONS[relation](X, Z)
    seq = sorted(range(len(labels)), key=lambda i: (-abs(weights[i]), labels[i]))

    groups: list[list[int]] = []
    for i in seq:
        for g in groups:
            if compat[i, g].all():
                g.append(int(i))
                break
        else:
            groups.append([int(i)])
    return groups


def greedy_clique_lower_bound(labels: list[str], relation: str = "qwc") -> int:
    """Greedy maximum clique in the *conflict* graph.  Any clique of
    mutually incompatible terms must occupy distinct measurement settings,
    so its size lower-bounds the minimum clique cover."""
    X, Z = pack(labels)
    conflict = ~RELATIONS[relation](X, Z)
    np.fill_diagonal(conflict, False)
    degree = conflict.sum(axis=1)
    order = sorted(range(len(labels)), key=lambda i: (-int(degree[i]), labels[i]))

    best = 0
    for start in order[: min(64, len(order))]:      # bounded restarts
        clique = [start]
        cand = np.flatnonzero(conflict[start])
        cand = sorted(cand, key=lambda i: (-int(degree[i]), labels[i]))
        for c in cand:
            if conflict[c, clique].all():
                clique.append(int(c))
        best = max(best, len(clique))
    return best


# --------------------------------------------------------------------------
# Cost metrics
# --------------------------------------------------------------------------
def n_settings(groups: list[list[int]]) -> int:
    return len(groups)


def setting_reduction(n_terms: int, groups: list[list[int]]) -> float:
    """Fractional reduction in the number of distinct measurement settings
    relative to measuring every Pauli term separately."""
    return 100.0 * (1.0 - len(groups) / n_terms)


def shot_cost(groups: list[list[int]], labels: list[str],
              coeffs: np.ndarray, psi: np.ndarray,
              pauli_matrices) -> float:
    """Exact optimal-allocation shot metric

        M = ( sum_g sqrt(Var_g) )^2 ,   Var_g = <O_g^2> - <O_g>^2,

    where O_g is the sum of the (coefficient-weighted) Pauli terms in group
    g, evaluated on the *actual* state ``psi``.  This is the quantity that
    multiplies 1/eps^2 in the total shot count, and it is the metric that
    actually matters -- the raw group count is only a proxy for it.

    ``pauli_matrices`` maps a label to its sparse matrix.
    """
    total = 0.0
    for g in groups:
        Opsi = np.zeros_like(psi)
        for i in g:
            Opsi += coeffs[i] * (pauli_matrices[labels[i]] @ psi)
        mean = np.vdot(psi, Opsi).real
        second = np.vdot(Opsi, Opsi).real
        var = max(second - mean ** 2, 0.0)
        total += np.sqrt(var)
    return total ** 2


# --------------------------------------------------------------------------
# Structural diagnostics -- used to explain *why* a mapping groups better
# --------------------------------------------------------------------------
def structure_stats(labels: list[str], relation: str = "qwc") -> dict:
    X, Z = pack(labels)
    compat = RELATIONS[relation](X, Z)
    m = len(labels)
    weights = np.array([sum(c != "I" for c in lab) for lab in labels])
    off_diag = compat.sum() - m
    return {
        "n_terms": m,
        "n_qubits": len(labels[0]),
        "mean_weight": float(weights.mean()),
        "max_weight": int(weights.max()),
        "mean_identity_positions": float(len(labels[0]) - weights.mean()),
        "compat_density": float(off_diag / (m * (m - 1))) if m > 1 else 1.0,
        "mean_compat_degree": float(off_diag / m),
    }


def gradient_group_variances(groups, labels, comms, psi, pauli_matrices,
                             tol: float = 1e-12) -> np.ndarray:
    """Return ``V[j,g]`` for grouped ADAPT-gradient estimators.

    ``V[j,g]`` is the exact statevector variance of the portion of
    commutator ``j`` measured in group ``g``.  This is the common primitive
    needed by both independent-per-gradient and shared-shot allocation.
    """
    V = np.zeros((len(comms), len(groups)), dtype=float)
    for j, cj in enumerate(comms):
        for gi, group in enumerate(groups):
            Opsi = np.zeros_like(psi)
            used = False
            for i in group:
                c = cj.get(labels[i], 0.0)
                if abs(c) <= tol:
                    continue
                used = True
                Opsi += c * (pauli_matrices[labels[i]] @ psi)
            if not used:
                continue
            mean = np.vdot(psi, Opsi).real
            second = np.vdot(Opsi, Opsi).real
            V[j, gi] = max(second - mean ** 2, 0.0)
    return V


def gradient_independent_shot_metric(groups, labels, comms, psi, pauli_matrices):
    """Independent-gradient allocation metric.

    For each gradient component ``j`` we optimize its group allocation as if
    it were measured in an independent experiment,

        M_j = (sum_g sqrt(V[j,g]))**2,

    then return ``sum_j M_j``.  This is a useful variance-aware proxy and an
    upper bound on a workflow that can reuse the same group shots across
    multiple gradient components.  It must *not* be described as the exact
    minimum shots for the full gradient vector.
    """
    V = gradient_group_variances(groups, labels, comms, psi, pauli_matrices)
    per_j = np.square(np.sqrt(V).sum(axis=1))
    return float(per_j.sum()), [float(x) for x in per_j]


def solve_shared_allocation(V, tol: float = 1e-10):
    """Solve the continuous shared-shot allocation from a variance matrix.

    ``V[j,g]`` is the contribution of group ``g`` to the variance of gradient
    ``j`` for one shot. The objective and constraints are those in Eq. (mshare)
    of the accompanying manuscript.
    """
    from scipy.optimize import minimize

    V = np.asarray(V, dtype=float).copy()
    if V.ndim != 2:
        raise ValueError("V must be a two-dimensional variance matrix")
    if np.any(V < 0.0):
        raise ValueError("variance entries must be non-negative")
    if V.size == 0 or not np.any(V > 0.0):
        return 0.0, [], {"success": True, "max_constraint": 0.0}

    # Cancellation can leave numerical-scale positive entries. Screen them
    # relative to the largest variance so they do not dominate the dual.
    V[V < tol * V.max()] = 0.0
    keep_g = V.max(axis=0) > 0.0
    Va = V[:, keep_g]
    keep_j = Va.max(axis=1) > 0.0
    Va = Va[keep_j]
    if Va.size == 0:
        return 0.0, [], {"success": True, "max_constraint": 0.0}

    # Log parameters lambda_j = exp(mu_j) keep all dual variables positive and
    # the square-root terms smooth even when the variance scale is very wide.
    lam0 = np.maximum(
        np.array([(np.sqrt(Va[j]).sum()) ** 2 for j in range(Va.shape[0])]),
        1e-300)

    def neg_dual(mu):
        lam = np.exp(mu)
        root = np.sqrt(lam @ Va)
        dual = 2.0 * root.sum() - lam.sum()
        grad_lam = (Va / root[None, :]).sum(axis=1) - 1.0
        return -dual, -(grad_lam * lam)

    res = minimize(neg_dual, np.log(lam0), jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-16, "gtol": 1e-12,
                            "maxiter": 50000, "maxfun": 100000})
    lam = np.exp(res.x)
    n = np.sqrt(lam @ Va)
    constraints = (Va / n[None, :]).sum(axis=1)
    primal = float(n.sum())
    dual = float(2.0 * n.sum() - lam.sum())
    gap = abs(primal - dual) / max(1.0, primal)
    if constraints.max() > 1.0 + 1e-5 or gap > 1e-5:
        raise RuntimeError(
            "shared-shot allocation failed validation: "
            f"optimizer_success={res.success}, max_constraint={constraints.max():.6g}, "
            f"relative_duality_gap={gap:.3e}"
        )
    return primal, [float(x) for x in n], {
        "success": True,
        "optimizer_success": bool(res.success),
        "max_constraint": float(constraints.max()),
        "relative_duality_gap": float(gap),
        "active_gradients": int(Va.shape[0]),
        "active_groups": int(Va.shape[1]),
    }


def gradient_shared_shot_metric(groups, labels, comms, psi, pauli_matrices,
                                tol: float = 1e-10):
    """Continuous optimum when group shots are shared by all gradients."""
    V = gradient_group_variances(groups, labels, comms, psi, pauli_matrices)
    return solve_shared_allocation(V, tol=tol)



def support_mask_stats(labels):
    """Diagnostics on the *pattern* of identity positions.

    QWC compatibility is driven not only by how many identity slots a
    Pauli string has, but by whether different strings put their
    identities in the *same* places.  This diagnostic measures support-mask
    reuse without assigning a causal mechanism to a particular encoding.
    """
    n = len(labels[0])
    masks = {}
    for lab in labels:
        key = tuple(c != "I" for c in lab)
        masks[key] = masks.get(key, 0) + 1
    counts = np.array(sorted(masks.values(), reverse=True))
    per_qubit = np.array([sum(lab[q] != "I" for lab in labels) / len(labels)
                          for q in range(n)])
    return {
        "n_distinct_support_masks": int(len(masks)),
        "mask_reuse_ratio": float(len(labels) / len(masks)),
        "largest_mask_class": int(counts[0]),
        "qubit_occupancy_min": float(per_qubit.min()),
        "qubit_occupancy_max": float(per_qubit.max()),
        "qubit_occupancy_std": float(per_qubit.std()),
    }


# --------------------------------------------------------------------------
# Memory-bounded implementations (no O(m^2) matrix materialisation)
# --------------------------------------------------------------------------
def _conflict_degrees(X, Z, relation, block=512):
    """Conflict degree per term, computed in row blocks so that peak memory
    is O(block * m) rather than O(m^2 * n_qubits)."""
    m = X.shape[0]
    deg = np.zeros(m, dtype=np.int64)
    supp = (X | Z).astype(bool)
    for s in range(0, m, block):
        e = min(s + block, m)
        if relation == "qwc":
            same = ((X[s:e, None, :] == X[None, :, :])
                    & (Z[s:e, None, :] == Z[None, :, :]))
            both = supp[s:e, None, :] & supp[None, :, :]
            compat = ~(both & ~same).any(axis=2)
        else:
            sp = (X[s:e].astype(np.int64) @ Z.T.astype(np.int64)
                  + Z[s:e].astype(np.int64) @ X.T.astype(np.int64)) % 2
            compat = sp == 0
        deg[s:e] = (~compat).sum(axis=1)
    return deg


def _order_indices(labels, X, Z, relation, order, seed):
    m = len(labels)
    if order == "lf":
        deg = _conflict_degrees(X, Z, relation)
        return sorted(range(m), key=lambda i: (-int(deg[i]), labels[i]))
    if order == "lex":
        return sorted(range(m), key=lambda i: labels[i])
    if order == "rand":
        return list(np.random.default_rng(seed).permutation(m))
    raise ValueError(order)


def greedy_color_scalable(labels, relation="qwc", order="lf", seed=None,
                          weights=None):
    """Greedy clique cover that never builds the full pairwise matrix.

    For QWC each group is summarised by its tensor-product measurement
    basis: a per-qubit assignment in {free, X, Y, Z}.  A term joins the
    group iff, at every qubit in its support, the slot is free or already
    carries the same single-qubit Pauli.  This makes the membership test
    O(n_qubits) instead of O(|group| * n_qubits).

    For general commutativity no such summary exists, so membership is
    tested against the stored members with a vectorised symplectic
    product.
    """
    X, Z = pack(labels)
    m, n = X.shape
    if weights is not None:
        seq = sorted(range(m), key=lambda i: (-abs(weights[i]), labels[i]))
    else:
        seq = _order_indices(labels, X, Z, relation, order, seed)

    groups: list[list[int]] = []
    if relation == "qwc":
        # slot codes: 0 free, 1 X, 2 Y, 3 Z
        code = np.zeros((m,), dtype=np.uint8)
        codes = (X.astype(np.uint8) * 1 + Z.astype(np.uint8) * 2)  # X=1,Z=2,Y=3
        slots: list[np.ndarray] = []
        for i in seq:
            ci = codes[i]
            supp_i = ci != 0
            for gi, slot in enumerate(slots):
                clash = supp_i & (slot != 0) & (slot != ci)
                if not clash.any():
                    slot[supp_i] = ci[supp_i]
                    groups[gi].append(int(i))
                    break
            else:
                new = np.zeros(n, dtype=np.uint8)
                new[supp_i] = ci[supp_i]
                slots.append(new)
                groups.append([int(i)])
        del code
    else:
        gx: list[np.ndarray] = []
        gz: list[np.ndarray] = []
        for i in seq:
            xi, zi = X[i].astype(np.int64), Z[i].astype(np.int64)
            for gi in range(len(groups)):
                sp = (gx[gi] @ zi + gz[gi] @ xi) % 2
                if not sp.any():
                    gx[gi] = np.vstack([gx[gi], xi])
                    gz[gi] = np.vstack([gz[gi], zi])
                    groups[gi].append(int(i))
                    break
            else:
                gx.append(xi[None, :])
                gz.append(zi[None, :])
                groups.append([int(i)])
    return groups


def clique_lower_bound_scalable(labels, relation="qwc", restarts=32):
    """Greedy maximum clique in the conflict graph, memory-bounded."""
    X, Z = pack(labels)
    m = X.shape[0]
    deg = _conflict_degrees(X, Z, relation)
    order = sorted(range(m), key=lambda i: (-int(deg[i]), labels[i]))
    supp = (X | Z).astype(bool)

    def conflicts_with(i, js):
        js = np.asarray(js, dtype=np.int64)
        if relation == "qwc":
            same = (X[i][None, :] == X[js]) & (Z[i][None, :] == Z[js])
            both = supp[i][None, :] & supp[js]
            return (both & ~same).any(axis=1)
        sp = (X[js].astype(np.int64) @ Z[i].astype(np.int64)
              + Z[js].astype(np.int64) @ X[i].astype(np.int64)) % 2
        return sp != 0

    best = 0
    for start in order[:restarts]:
        cand = np.flatnonzero(conflicts_with(start, np.arange(m)))
        cand = sorted(cand, key=lambda i: (-int(deg[i]), labels[i]))
        clique = [start]
        for c in cand:
            if conflicts_with(int(c), clique).all():
                clique.append(int(c))
        best = max(best, len(clique))
    return best


# --------------------------------------------------------------------------
# Public API: dispatch to the memory-bounded implementations.  The dense
# versions above are retained as reference implementations and are checked
# against these in tests/test_grouping.py.
# --------------------------------------------------------------------------
_greedy_color_dense = greedy_color
_sorted_insertion_dense = sorted_insertion
_clique_lb_dense = greedy_clique_lower_bound


def greedy_color(labels, relation="qwc", order="lf", seed=None):      # noqa: F811
    return greedy_color_scalable(labels, relation, order, seed)


def sorted_insertion(labels, weights, relation="qwc"):                # noqa: F811
    return greedy_color_scalable(labels, relation, weights=np.asarray(weights))


def greedy_clique_lower_bound(labels, relation="qwc"):                # noqa: F811
    return clique_lower_bound_scalable(labels, relation)


def structure_stats(labels, relation="qwc"):                          # noqa: F811
    X, Z = pack(labels)
    m = X.shape[0]
    deg = _conflict_degrees(X, Z, relation)
    weights = np.array([sum(c != "I" for c in lab) for lab in labels])
    compat_deg = (m - 1) - deg
    return {
        "n_terms": m,
        "n_qubits": X.shape[1],
        "mean_weight": float(weights.mean()),
        "max_weight": int(weights.max()),
        "mean_identity_positions": float(X.shape[1] - weights.mean()),
        "compat_density": float(compat_deg.sum() / (m * (m - 1))) if m > 1 else 1.0,
        "mean_compat_degree": float(compat_deg.mean()),
    }
