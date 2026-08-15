"""Audits for encoding invariance between linear fermion-to-qubit mappings.

Jordan--Wigner and Bravyi--Kitaev are linear encodings of the Fock basis.
At the Pauli level they are related by a Clifford/symplectic transformation.
Consequently full Pauli commutation is preserved exactly.  Qubit-wise
commutativity need not be preserved because an entangling Clifford changes the
tensor-product support pattern.

This module reconstructs the binary symplectic transformation directly from a
Qiskit Nature mapper's Majorana Pauli table.  It is intentionally independent
of any hard-coded BK convention, so it also catches mapper-version changes.
"""
from __future__ import annotations

import numpy as np

from grouping import label_to_xz

_INV_XZ = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def _vec(label: str) -> np.ndarray:
    x, z = label_to_xz(label)
    return np.concatenate([x, z]).astype(np.uint8)


def _label(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.uint8) & 1
    n = len(v) // 2
    x, z = v[:n], v[n:]
    return "".join(_INV_XZ[(int(a), int(b))] for a, b in zip(x, z))


def gf2_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of a square binary matrix over GF(2)."""
    a = (np.asarray(a, dtype=np.uint8) & 1).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("GF(2) inverse requires a square matrix")
    n = a.shape[0]
    aug = np.concatenate([a, np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        piv = np.flatnonzero(aug[col:, col])
        if len(piv) == 0:
            raise ValueError("matrix is singular over GF(2)")
        piv = int(piv[0] + col)
        if piv != col:
            aug[[col, piv]] = aug[[piv, col]]
        rows = np.flatnonzero(aug[:, col])
        rows = rows[rows != col]
        for r in rows:
            aug[r] ^= aug[col]
    return aug[:, n:]


def symplectic_form(n_qubits: int) -> np.ndarray:
    eye = np.eye(n_qubits, dtype=np.uint8)
    zero = np.zeros_like(eye)
    return np.block([[zero, eye], [eye, zero]])


def is_symplectic(t: np.ndarray) -> bool:
    t = np.asarray(t, dtype=np.uint8) & 1
    if t.ndim != 2 or t.shape[0] != t.shape[1] or t.shape[0] % 2:
        return False
    n = t.shape[0] // 2
    j = symplectic_form(n)
    return np.array_equal((t @ j @ t.T) & 1, j)


def mapper_majorana_labels(mapper, register_length: int) -> list[str]:
    """Return the 2N Majorana Pauli representatives of a Qiskit mapper."""
    table = mapper.pauli_table(register_length)
    labels: list[str] = []
    for p_re, p_im in table:
        labels.extend([p_re.to_label(), p_im.to_label()])
    return labels


def mapper_symplectic_transform(source_mapper, target_mapper,
                                 register_length: int) -> np.ndarray:
    """Reconstruct ``v_target = v_source @ T`` over GF(2)."""
    src = np.vstack([_vec(l) for l in mapper_majorana_labels(
        source_mapper, register_length)])
    dst = np.vstack([_vec(l) for l in mapper_majorana_labels(
        target_mapper, register_length)])
    t = (gf2_inverse(src) @ dst) & 1
    if not is_symplectic(t):
        raise AssertionError("reconstructed mapper transform is not symplectic")
    if not np.array_equal((src @ t) & 1, dst):
        raise AssertionError("mapper transform does not reproduce Majorana table")
    return t


def transform_label(label: str, transform: np.ndarray) -> str:
    """Transform a Pauli label modulo its overall phase."""
    return _label((_vec(label) @ (np.asarray(transform, dtype=np.uint8) & 1)) & 1)


def transform_labels(labels: list[str], transform: np.ndarray) -> list[str]:
    return [transform_label(lab, transform) for lab in labels]


def support_equivalent(source_labels: list[str], target_labels: list[str],
                       transform: np.ndarray) -> bool:
    return set(transform_labels(source_labels, transform)) == set(target_labels)


def transform_groups(groups: list[list[int]], source_labels: list[str],
                     target_labels: list[str], transform: np.ndarray) -> list[list[int]]:
    """Carry a grouping across an encoding using the Pauli bijection."""
    target_index = {lab: i for i, lab in enumerate(target_labels)}
    if len(target_index) != len(target_labels):
        raise ValueError("target labels are not unique")
    out = []
    for group in groups:
        mapped = []
        for i in group:
            lab = transform_label(source_labels[i], transform)
            if lab not in target_index:
                raise KeyError(f"mapped label {lab} absent from target support")
            mapped.append(target_index[lab])
        out.append(mapped)
    return out


def commutation_signature(labels: list[str]) -> np.ndarray:
    """Packed upper-triangle GC signature, useful in regression tests."""
    from grouping import gc_matrix, pack
    x, z = pack(labels)
    mat = gc_matrix(x, z)
    iu = np.triu_indices(len(labels), 1)
    return mat[iu]
