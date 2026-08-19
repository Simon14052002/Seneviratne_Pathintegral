"""
The 'grid' (NORMAL / full pipeline): turn a saved dynamical map L(t) into
Kraus operators and propagate -- fresh Choi/Kraus decomposition at EVERY
saved time point. This is the direct, per-timestep read-out; see ../enhanced/
for the "compute Kraus once, reuse it" algorithm.

A *map file* is an .npz with the standard fields
    t_fs   : (nt,)          time grid in femtoseconds
    maps   : (nt, D, D)     the linear maps L(t_k), D = d^2  (column-stacking)
    H      : (d, d)         system Hamiltonian (reference only)
    d      : ()             system dimension
    label  : ()             human-readable name

Every *_map.py file in this folder produces such a file (in data/). This
module reads it back, builds the Kraus operators M_k(t) at each time (Choi
eigendecomposition), and propagates a state by the operator sum
    rho(t) = sum_k M_k(t) rho0 M_k(t)^dag.

Column-stacking convention everywhere:  vec(rho)[i + d*j] = rho[i, j].
"""

import numpy as np


# ---------------------------------------------------------------- vectorization
def vec(rho):
    return np.asarray(rho).flatten(order="F")


def unvec(v, d):
    return np.asarray(v).reshape(d, d, order="F")


# ---------------------------------------------------------------- Choi / Kraus
def choi_from_map(L, d):
    """Choi matrix  C = sum_ij E_ij (x) L(E_ij)  (Seneviratne eq. 13)."""
    C = np.zeros((d * d, d * d), dtype=complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d), dtype=complex)
            Eij[i, j] = 1.0
            C += np.kron(Eij, unvec(L @ vec(Eij), d))
    return C


def kraus_from_choi(C, d, tol=1e-10):
    """Kraus operators from the Choi eigendecomposition (eqs. 14-15).
    Returns (kraus_list, min_eigenvalue)."""
    Ch = 0.5 * (C + C.conj().T)
    evals, evecs = np.linalg.eigh(Ch)
    kraus = [np.sqrt(evals[k]) * unvec(evecs[:, k], d)
             for k in range(len(evals)) if evals[k] > tol]
    return kraus, float(evals.min())


def kraus_from_map(L, d, tol=1e-10):
    """Kraus operators M_k of a single map matrix L (D x D)."""
    return kraus_from_choi(choi_from_map(L, d), d, tol)


# ---------------------------------------------------------------- save / load
def save_maps(path, t_fs, maps, H, label):
    H = np.asarray(H)
    np.savez(path, t_fs=np.asarray(t_fs, float),
             maps=np.asarray(maps, complex), H=H, d=H.shape[0],
             label=str(label))


def load_maps(path):
    """Returns dict with t_fs, maps, H, d, label."""
    z = np.load(path, allow_pickle=True)
    return dict(t_fs=z["t_fs"], maps=z["maps"], H=z["H"],
                d=int(z["d"]), label=str(z["label"]))


# ---------------------------------------------------------------- propagation
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


def populations_via_kraus(mapfile_or_dict, rho0):
    """Read a map file (or loaded dict), build Kraus operators at EVERY
    saved time, propagate rho0. Returns dict: t_fs, pops (nt,d) from the
    operator sum, plus CP diagnostics."""
    data = (load_maps(mapfile_or_dict) if isinstance(mapfile_or_dict, str)
            else mapfile_or_dict)
    d = data["d"]
    rho0 = np.asarray(rho0, complex)
    _, rho_kr, nk, min_eig, tr_err = propagate_grid(data["maps"], rho0)
    return dict(t_fs=data["t_fs"],
                pops=np.real(np.einsum("tii->ti", rho_kr)),
                rho=rho_kr, n_kraus=nk, min_choi_eig=min_eig,
                trace_err=tr_err, label=data["label"])
