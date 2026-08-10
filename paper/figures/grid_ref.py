"""
Pure-numpy reference for the enhanced (grid) algorithm, copied from
maps_to_grid/enhanced/grid.ipynb so the paper figures can be regenerated from
the saved dynamical maps. The dilated-unitary propagation here is exactly the
operation the Qiskit circuit runs (AerSimulator statevector == matrix-vector
product), so these trajectories are the quantum-grid results.
"""
import numpy as np
from scipy.linalg import sqrtm


def vec(rho):
    return np.asarray(rho).flatten(order="F")


def unvec(v, d):
    return np.asarray(v).reshape(d, d, order="F")


def choi_from_map(L, d):
    C = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            C += np.kron(Eij, unvec(L @ vec(Eij), d))
    return C


def kraus_from_choi(C, d, tol=1e-10):
    Ch = 0.5 * (C + C.conj().T)
    evals, evecs = np.linalg.eigh(Ch)
    kraus = [np.sqrt(evals[k]) * unvec(evecs[:, k], d)
             for k in range(len(evals)) if evals[k] > tol]
    return kraus, float(evals.min())


def enhanced_kraus(L_dt, d, tol=1e-10):
    return kraus_from_choi(choi_from_map(L_dt, d), d, tol)


def transfer_tensors(maps, K):
    T = []
    for n in range(1, K + 1):
        Tn = np.array(maps[n], dtype=complex, copy=True)
        for m in range(1, n):
            Tn -= T[m - 1] @ maps[n - m]
        T.append(Tn)
    return T


def companion_propagator(T):
    K, D = len(T), T[0].shape[0]
    E = np.zeros((K * D, K * D), dtype=complex)
    for m, Tm in enumerate(T):
        E[:D, m * D:(m + 1) * D] = Tm
    for r in range(1, K):
        E[r * D:(r + 1) * D, (r - 1) * D:r * D] = np.eye(D)
    return E


def sz_nagy_dilation(E, s=None):
    if s is None:
        s = np.linalg.norm(E, 2) * (1.0 + 1e-12)
    Es = E / s
    I = np.eye(Es.shape[0])
    B = sqrtm(I - Es @ Es.conj().T)
    C = sqrtm(I - Es.conj().T @ Es)
    B, C = 0.5 * (B + B.conj().T), 0.5 * (C + C.conj().T)
    return np.block([[Es, B], [C, -Es.conj().T]]), float(s)


def propagate_dilated(E, x0, n_steps):
    """The grid: dilate E/s into one unitary, then re-apply it, reading the
    top block out and feeding it back in (== the reused Qiskit circuit)."""
    U, s = sz_nagy_dilation(E)
    n = E.shape[0]
    x = np.asarray(x0, complex)
    traj = [x.copy()]
    for _ in range(n_steps):
        nrm = np.linalg.norm(x)
        y = U @ np.concatenate([x / nrm, np.zeros(n)])
        x = y[:n] * (s * nrm)
        traj.append(x.copy())
    return np.array(traj), U, s


def pops_direct(maps, rho0):
    """Exact benchmark: propagate the dynamical map itself at every time."""
    d = rho0.shape[0]
    v0 = vec(np.asarray(rho0, complex))
    return np.array([np.real(np.diag(unvec(L @ v0, d))) for L in maps])


def pops_enhanced_markov(L_dt, d, rho0, n_steps):
    """Variant 1 (semigroup): one-step Kraus M_k(dt), re-applied."""
    kraus, _ = enhanced_kraus(L_dt, d)
    rho = np.asarray(rho0, complex)
    traj = [rho.copy()]
    for _ in range(n_steps):
        rho = sum(M @ rho @ M.conj().T for M in kraus)
        traj.append(rho)
    return np.real(np.einsum("tii->ti", np.array(traj)))


def pops_enhanced_memory(maps, rho0, K, n_steps):
    """Variant 2 (memory): transfer tensors + companion operator on the grid."""
    D = maps.shape[1]
    d = int(round(np.sqrt(D)))
    T = transfer_tensors(maps, K)
    E = companion_propagator(T)
    v0 = vec(np.asarray(rho0, complex))
    hist = [maps[n] @ v0 for n in range(min(K, n_steps + 1))]
    X0 = np.concatenate(hist[::-1])
    Xs, U, s = propagate_dilated(E, X0, max(n_steps + 1 - K, 0))
    states = hist + [X[:D] for X in Xs[1:]]
    traj = np.array([unvec(v, d) for v in states[:n_steps + 1]])
    Tn = np.array([np.linalg.norm(Tm, 2) for Tm in T])
    return np.real(np.einsum("tii->ti", traj)), Tn


def traj_and_E_memory(maps, rho0, K, n_steps):
    """Return the full density-matrix trajectory and the companion operator E,
    for the physicality and stability diagnostics."""
    D = maps.shape[1]
    d = int(round(np.sqrt(D)))
    T = transfer_tensors(maps, K)
    E = companion_propagator(T)
    v0 = vec(np.asarray(rho0, complex))
    hist = [maps[n] @ v0 for n in range(min(K, n_steps + 1))]
    Xs, _, _ = propagate_dilated(E, np.concatenate(hist[::-1]),
                                 max(n_steps + 1 - K, 0))
    states = hist + [X[:D] for X in Xs[1:]]
    rho = np.array([unvec(v, d) for v in states[:n_steps + 1]])
    return rho, E


def traj_markov(L_dt, d, rho0, n_steps):
    kraus, _ = enhanced_kraus(L_dt, d)
    rho = np.asarray(rho0, complex)
    traj = [rho.copy()]
    for _ in range(n_steps):
        rho = sum(M @ rho @ M.conj().T for M in kraus)
        traj.append(rho)
    return np.array(traj)


def load_maps(path):
    z = np.load(path, allow_pickle=True)
    return dict(t_fs=z["t_fs"], maps=z["maps"], H=z["H"],
                d=int(z["d"]), label=str(z["label"]))
