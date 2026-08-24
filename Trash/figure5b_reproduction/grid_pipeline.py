"""
The 'grid' structure used throughout the Figures folder (cf. calc_rho_M_k in
figure6.21.py and thesis Sec. 6.1.1): for every time t on the grid, take the
map matrix L(t), build the Choi matrix, eigendecompose it into Kraus
operators M_k(t), and propagate rho(0) via the operator sum
    rho(t) = sum_k M_k rho(0) M_k^dagger .
Column-stacking convention: vec(rho)[i + d*j] = rho[i, j].
"""

import numpy as np


def vec(rho):
    return rho.flatten(order='F')


def unvec(v, d):
    return v.reshape(d, d, order='F')


def choi_from_map(L, d):
    """Choi matrix  C = sum_ij E_ij (x) eps(E_ij)  (paper eq. 13)."""
    C = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            C += np.kron(Eij, unvec(L @ vec(Eij), d))
    return C


def kraus_from_choi(C, d, tol=1e-10):
    """Eigendecomposition of the Choi matrix (paper eqs. 14-15).
    Returns (kraus_list, min_eigenvalue)."""
    Ch = 0.5 * (C + C.conj().T)
    evals, evecs = np.linalg.eigh(Ch)
    kraus = []
    for k in range(len(evals)):
        if evals[k] > tol:
            kraus.append(np.sqrt(evals[k]) * unvec(evecs[:, k], d))
    return kraus, evals.min()


def propagate_grid(maps, rho0):
    """Apply the grid pipeline at every time point.
    Returns (rho_direct, rho_kraus, n_kraus, min_choi_eig, trace_err)."""
    d = rho0.shape[0]
    rho_dir, rho_kr, nk, min_eig, tr_err = [], [], [], [], []
    one = vec(np.eye(d))
    for L in maps:
        rho_dir.append(unvec(L @ vec(rho0), d))
        C = choi_from_map(L, d)
        kraus, me = kraus_from_choi(C, d)
        rho_kr.append(sum(M @ rho0 @ M.conj().T for M in kraus))
        nk.append(len(kraus))
        min_eig.append(me)
        tr_err.append(np.abs(one.conj() @ L - one.conj()).max())
    return (np.array(rho_dir), np.array(rho_kr), np.array(nk),
            np.array(min_eig), np.array(tr_err))
