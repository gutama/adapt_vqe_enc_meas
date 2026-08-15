"""
Active-space chemistry, spin-adapted operator pools, matrix-exponential
propagators, and a genuine ADAPT-VQE driver.

Design notes
------------
* Geometries are generated from a reference bond length ``Re`` and a
  stretch factor, so "2 x Re" is enforced rather than asserted.
* The reference energy is CASCI in the *same* active space that the
  qubit Hamiltonian describes -- this is the only fair comparison.
* ``expm_multiply`` (SciPy) implements Al-Mohy & Higham's scaled
  truncated-Taylor algorithm, *not* an Arnoldi/Lanczos Krylov
  projection.  A Lanczos propagator is provided separately so the two
  can be compared rather than conflated.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply, eigsh

from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import BravyiKitaevMapper, JordanWignerMapper
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

MAPPERS = {"JW": JordanWignerMapper, "BK": BravyiKitaevMapper}

# --------------------------------------------------------------------------
# Molecular specifications.  Re values are equilibrium bond lengths (Ang).
# --------------------------------------------------------------------------
MOLECULES = {
    "LiH": dict(Re=1.595, n_elec=(1, 1), n_orb=4, occ=[0], virt=[1, 2, 3],
                point_group=r"C_{\infty v}", charge=0, spin=0),
    "BeH2": dict(Re=1.332, n_elec=(2, 2), n_orb=4, occ=[0, 1], virt=[2, 3],
                 point_group=r"D_{\infty h}", charge=0, spin=0),
    "N2": dict(Re=1.0977, n_elec=(3, 3), n_orb=4, occ=[0, 1, 2], virt=[3],
               point_group=r"D_{\infty h}", charge=0, spin=0),
    "H2O": dict(Re=0.9584, angle=104.45, n_elec=(2, 2), n_orb=4,
                occ=[0, 1], virt=[2, 3], point_group=r"C_{2v}",
                charge=0, spin=0),
}


def geometry(name: str, stretch: float = 2.0) -> tuple[str, dict]:
    """Return a PySCF atom string and the realised internal coordinates."""
    spec = MOLECULES[name]
    R = spec["Re"] * stretch
    if name == "LiH":
        atom = f"Li 0.0 0.0 0.0; H 0.0 0.0 {R:.6f}"
        geom = {"R_LiH": R}
    elif name == "BeH2":
        atom = f"Be 0.0 0.0 0.0; H 0.0 0.0 {R:.6f}; H 0.0 0.0 {-R:.6f}"
        geom = {"R_BeH": R}
    elif name == "N2":
        atom = f"N 0.0 0.0 0.0; N 0.0 0.0 {R:.6f}"
        geom = {"R_NN": R}
    elif name == "H2O":
        half = np.deg2rad(spec["angle"]) / 2.0
        y, z = R * np.sin(half), R * np.cos(half)
        atom = (f"O 0.0 0.0 0.0; "
                f"H 0.0 {y:.6f} {z:.6f}; H 0.0 {-y:.6f} {z:.6f}")
        geom = {"R_OH": R, "angle_HOH": spec["angle"]}
    else:
        raise ValueError(name)
    geom["stretch"] = stretch
    geom["Re"] = spec["Re"]
    return atom, geom


# --------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------
def build_problem(name: str, stretch: float = 2.0, basis: str = "sto3g"):
    spec = MOLECULES[name]
    atom, geom = geometry(name, stretch)
    driver = PySCFDriver(atom=atom, basis=basis,
                         charge=spec["charge"], spin=spec["spin"])
    raw = driver.run()
    transformer = ActiveSpaceTransformer(
        num_electrons=spec["n_elec"], num_spatial_orbitals=spec["n_orb"])
    problem = transformer.transform(raw)
    return problem, raw, geom, spec


def constant_shift(problem) -> float:
    """Sum of all scalar constants (nuclear repulsion + inactive-space
    energy) attached to the reduced electronic Hamiltonian."""
    return float(sum(problem.hamiltonian.constants.values()))


def casci_reference(name: str, stretch: float = 2.0, basis: str = "sto3g"):
    """Independent CASCI/FCI reference computed directly with PySCF."""
    from pyscf import gto, scf, mcscf, fci

    spec = MOLECULES[name]
    atom, _ = geometry(name, stretch)
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom",
                charge=spec["charge"], spin=spec["spin"], verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    n_e = sum(spec["n_elec"])
    mc = mcscf.CASCI(mf, spec["n_orb"], n_e)
    mc.verbose = 0
    e_cas = mc.kernel()[0]
    e_fci = fci.FCI(mf).kernel()[0]
    return dict(e_hf=float(mf.e_tot), e_casci=float(e_cas),
                e_fci_full=float(e_fci), e_nuc=float(mol.energy_nuc()))


# --------------------------------------------------------------------------
# Spin-complemented singles / opposite-spin doubles pool
# --------------------------------------------------------------------------
def susd_pool_fermionic(occ, virt, n_spatial) -> list[FermionicOp]:
    """Spin-complemented singles and opposite-spin doubles pool.

    The singles combine alpha and beta complements. The doubles contain the
    two opposite-spin complements; same-spin alpha-alpha and beta-beta terms
    are not part of this pool.
    """
    ops = []
    for i in occ:
        for a in virt:
            ib, ab = i + n_spatial, a + n_spatial
            ops.append(FermionicOp(
                {f"+_{a} -_{i}": 1.0, f"+_{i} -_{a}": -1.0,
                 f"+_{ab} -_{ib}": 1.0, f"+_{ib} -_{ab}": -1.0},
                num_spin_orbitals=2 * n_spatial))
    for ii, i in enumerate(occ):
        for j in occ[ii:]:
            for aa, a in enumerate(virt):
                for b in virt[aa:]:
                    ib, jb = i + n_spatial, j + n_spatial
                    ab, bb = a + n_spatial, b + n_spatial
                    ops.append(FermionicOp(
                        {f"+_{a} +_{bb} -_{jb} -_{i}": 1.0,
                         f"+_{i} +_{jb} -_{bb} -_{a}": -1.0,
                         f"+_{ab} +_{b} -_{j} -_{ib}": 1.0,
                         f"+_{ib} +_{j} -_{b} -_{ab}": -1.0},
                        num_spin_orbitals=2 * n_spatial))
    return ops


def canonical_key(op: SparsePauliOp, tol: float = 1e-10) -> tuple:
    """Phase- and order-invariant fingerprint of a SparsePauliOp.

    Sorting the (label, coefficient) pairs removes term-ordering
    ambiguity; dividing by the leading coefficient removes an arbitrary
    global scale/phase.  Without this, structurally identical pool
    operators survive de-duplication and inflate the term count.
    """
    items = [(lab, complex(c))
             for lab, c in zip(op.paulis.to_labels(), op.coeffs)
             if abs(c) > tol]
    if not items:
        return ()
    items.sort(key=lambda t: t[0])
    lead = items[0][1]
    return tuple((lab, complex(np.round(c / lead, 10))) for lab, c in items)


def qubit_pool(pool_ferm, mapper, tol: float = 1e-10):
    """Map, simplify, multiply by i (anti-Hermitian -> Hermitian generator
    convention used for gradients), and de-duplicate canonically."""
    out, seen = [], set()
    for op in pool_ferm:
        q = (mapper.map(op).simplify() * 1j).simplify()
        q = SparsePauliOp(q.paulis, q.coeffs).simplify(atol=tol)
        key = canonical_key(q, tol)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def unique_terms(ops, tol: float = 1e-10):
    """Deterministically ordered unique non-identity Pauli labels across a
    list of operators, with the max |coefficient| observed for each."""
    acc: dict[str, float] = {}
    for op in ops:
        for lab, c in zip(op.paulis.to_labels(), op.coeffs):
            if abs(c) <= tol or all(ch == "I" for ch in lab):
                continue
            acc[lab] = max(acc.get(lab, 0.0), abs(complex(c)))
    labels = sorted(acc)
    # SparsePauliOp.simplify() sums duplicate-Pauli coefficients in a
    # hash-dependent order, so |c| can differ in its last bits between
    # processes.  Sorted insertion breaks ties on these magnitudes, so the
    # jitter changes the partition.  Quantise to make the ordering canonical.
    return labels, np.round(np.array([acc[l] for l in labels]), 12)


# --------------------------------------------------------------------------
# Propagators
# --------------------------------------------------------------------------
def prop_expm_multiply(A_sparse, theta, psi):
    """SciPy: Al-Mohy--Higham scaled truncated Taylor series."""
    return expm_multiply(theta * A_sparse, psi)


def prop_lanczos(A_sparse, theta, psi, m: int = 30, tol: float = 1e-14):
    """Genuine Krylov (Lanczos) propagator for anti-Hermitian A.

    Writes A = i B with B = -i A Hermitian, builds the Krylov space
    K_m(B, psi) by Lanczos, and evaluates exp(i theta B) in that space.
    Unitarity is exact up to the orthogonality of the Lanczos basis.
    """
    beta = np.linalg.norm(psi)
    if beta < tol:
        return psi.copy()
    V = np.zeros((m + 1, psi.size), dtype=complex)
    alpha = np.zeros(m, dtype=float)
    betas = np.zeros(m, dtype=float)
    V[0] = psi / beta
    k_used = m
    for k in range(m):
        w = -1j * (A_sparse @ V[k])                  # B v
        alpha[k] = np.vdot(V[k], w).real
        w = w - alpha[k] * V[k] - (betas[k - 1] * V[k - 1] if k > 0 else 0)
        for j in range(k + 1):                        # full reorthogonalisation
            w -= np.vdot(V[j], w) * V[j]
        nb = np.linalg.norm(w)
        if nb < tol:
            k_used = k + 1
            break
        betas[k] = nb
        V[k + 1] = w / nb
    else:
        k_used = m
    T = np.diag(alpha[:k_used]) \
        + np.diag(betas[:k_used - 1], 1) + np.diag(betas[:k_used - 1], -1)
    w_, S = np.linalg.eigh(T)
    e1 = np.zeros(k_used, dtype=complex)
    e1[0] = beta
    coef = S @ (np.exp(1j * theta * w_) * (S.conj().T @ e1))
    return coef @ V[:k_used]


def prop_dense(A_sparse, theta, psi):
    from scipy.linalg import expm
    return expm(theta * A_sparse.toarray()) @ psi


def prop_taylor(A_sparse, theta, psi, order: int = 4):
    """Naive fixed-order Taylor series -- included only to demonstrate the
    norm drift that motivates a proper matrix-function algorithm."""
    out = psi.copy()
    term = psi.copy()
    for k in range(1, order + 1):
        term = (theta / k) * (A_sparse @ term)
        out = out + term
    return out


# --------------------------------------------------------------------------
# ADAPT-VQE
# --------------------------------------------------------------------------
def hartree_fock_vector(problem, mapper, n_qubits):
    from qiskit.quantum_info import Statevector
    from qiskit_nature.second_q.circuit.library import HartreeFock
    circ = HartreeFock(problem.num_spatial_orbitals,
                       problem.num_particles, mapper)
    return np.asarray(Statevector.from_instruction(circ).data, dtype=complex)


def _apply_chain(gens, thetas, psi0, propagator):
    for A, t in zip(gens, thetas):
        psi0 = propagator(A, t, psi0)
    return psi0


def energy_and_grad(gens, thetas, psi0, H_sparse, propagator):
    """Energy and analytic gradient by adjoint (reverse-mode) differentiation."""
    states = [psi0]
    psi = psi0
    for A, t in zip(gens, thetas):
        psi = propagator(A, t, psi)
        states.append(psi)
    E = np.vdot(psi, H_sparse @ psi).real

    phi = H_sparse @ psi
    grad = np.zeros(len(thetas))
    for k in range(len(thetas) - 1, -1, -1):
        A, t = gens[k], thetas[k]
        psi_k = states[k + 1]
        grad[k] = 2.0 * np.vdot(phi, A @ psi_k).real
        phi = propagator(A, -t, phi)
    return E, grad


def run_adapt(problem, mapper_name, occ, virt, max_iter=12, grad_tol=1e-4,
              propagator=prop_expm_multiply, verbose=False):
    """Standard ADAPT-VQE: select the pool operator with the largest
    |<psi|[H, A]>|, append it, then re-optimise *all* parameters."""
    from scipy.optimize import minimize

    mapper = MAPPERS[mapper_name]()
    H_q = mapper.map(problem.second_q_ops()[0]).simplify()
    H = H_q.to_matrix(sparse=True).tocsr()
    shift = constant_shift(problem)

    pool = qubit_pool(susd_pool_fermionic(occ, virt,
                                          problem.num_spatial_orbitals), mapper)
    # Anti-Hermitian generators A = -i * (i*T) restores the original T.
    A_sparse = [(-1j * op.to_matrix(sparse=True)).tocsr() for op in pool]

    psi0 = hartree_fock_vector(problem, mapper, H_q.num_qubits)
    gens, thetas = [], []
    history = []

    for it in range(max_iter):
        psi = _apply_chain(gens, thetas, psi0, propagator)
        Hpsi = H @ psi
        grads = np.array([2.0 * np.vdot(Hpsi, A @ psi).real for A in A_sparse])
        j = int(np.argmax(np.abs(grads)))
        gmax = float(abs(grads[j]))
        norm = float(np.linalg.norm(psi))
        if abs(norm - 1.0) > 5e-10:
            raise AssertionError(
                f"state norm drifted to {norm:.16g} at ADAPT iteration {it}"
            )
        E_cur = float(np.vdot(psi, Hpsi).real) + shift
        history.append(dict(iteration=it, energy=E_cur, grad_max=gmax,
                            state_norm=norm, selected=j))
        if verbose:
            print(f"    it={it:2d}  E={E_cur:.9f}  |g|max={gmax:.3e}  "
                  f"||psi||={norm:.15f}")
        if gmax < grad_tol:
            break

        gens.append(A_sparse[j])
        thetas.append(0.0)
        x0 = np.array(thetas)

        def fun(x):
            E, g = energy_and_grad(gens, x, psi0, H, propagator)
            return E, g

        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=500, ftol=1e-14, gtol=1e-10))
        jac_inf = float(np.linalg.norm(np.asarray(res.jac), ord=np.inf))
        if (not res.success) and jac_inf > 1e-7:
            raise RuntimeError(
                f"L-BFGS-B failed at ADAPT iteration {it}: {res.message}; "
                f"||grad||_inf={jac_inf:.3e}"
            )
        thetas = list(res.x)
        history[-1]["optimizer_success"] = bool(res.success)
        history[-1]["optimizer_grad_inf"] = jac_inf
        history[-1]["optimizer_message"] = str(res.message)

    psi = _apply_chain(gens, thetas, psi0, propagator)
    final_norm = float(np.linalg.norm(psi))
    if abs(final_norm - 1.0) > 5e-10:
        raise AssertionError(f"final state norm drifted to {final_norm:.16g}")
    E_final = float(np.vdot(psi, H @ psi).real) + shift

    # exact diagonalisation of the same qubit Hamiltonian
    n = H.shape[0]
    if n <= 4096:
        e_exact = float(np.linalg.eigvalsh(H.toarray())[0]) + shift
    else:
        e_exact = float(eigsh(H, k=1, which="SA",
                              return_eigenvectors=False)[0]) + shift

    return dict(energy=E_final, exact=e_exact, n_ops=len(gens),
                state_norm=final_norm,
                history=history, psi=psi, H=H, pool=pool,
                H_qubit_op=H_q, shift=shift)
