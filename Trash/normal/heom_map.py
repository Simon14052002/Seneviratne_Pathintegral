"""
HEOM dynamical map  L(t) = [ exp(L_HEOM t) ]_00   (qHEOM, Batista et al.,
JCTC 2025, arXiv:2411.12049, Eq. 33): the reduced map is the top-left
(zeroth-ADO) block of the extended HEOM propagator, with the auxiliary density
operators (ADOs) initialised to zero.

NORMAL-pipeline copy: produces data/heom_maps.npz, one map per saved time,
consumed by ./grid.py's populations_via_kraus (fresh Choi/Kraus at every
time). For the "compute once, reuse forever" enhanced algorithm --
including the ADOs-as-memory-register one-step propagator -- see
../enhanced/heom_map.py.

Provides:
  build_heom_maps(...)     -> t_fs, maps  (via one eigendecomposition of the
                               constant HEOM generator; all times for free)
  standard_heom_pops(...)  -> populations from a direct qutip HEOMSolver.run
                               (the 'standard implementation' reference)
  compute_and_save(...)    -> writes data/heom_maps.npz in grid.py's format
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # let 'from grid import ...' work from any cwd

import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzPadeBath

# physical constants inlined (energies in cm^-1, time tau = 2*pi*c*t in cm) so
# this module has no cross-file import for the IDE's auto-organiser to rewrite
_C_CM = 2.99792458e10                    # speed of light [cm/s]
FS_TO_CM = 1e-15 * 2 * np.pi * _C_CM     # tau[cm] per t[fs]
KB_CM = 0.6950348004                     # Boltzmann constant [cm^-1/K]

# 4-site FMO Hamiltonian (Seneviratne eq. 30), cm^-1  -- same as the others
H_FMO = np.array([[12375.0, -87.7,   5.5,  -5.9],
                  [-87.7, 12495.0,  30.8,   8.2],
                  [5.5,      30.8, 12175.0, -53.4],
                  [-5.9,      8.2, -53.4, 12285.0]])
H_FMO = H_FMO - np.trace(H_FMO) / 4 * np.eye(4)
LAM_FMO, GAM_FMO, T_FMO = 35.0, 106.18, 300.0


def _make_solver(H, lam, gamma, T_K, Nk, depth):
    d = H.shape[0]
    baths = [DrudeLorentzPadeBath(qt.basis(d, i) * qt.basis(d, i).dag(), lam=lam, gamma=gamma, T=KB_CM * T_K, Nk=Nk)
             for i in range(d)]
    return HEOMSolver(qt.Qobj(H), baths, max_depth=depth)


def build_heom_maps(t_fs, H=H_FMO, lam=LAM_FMO, gamma=GAM_FMO, T_K=T_FMO, Nk=1, depth=3, verbose=True):

    """Reduced HEOM maps L(t) on the grid t_fs, via the block-exponential."""
    d = H.shape[0]
    solver = _make_solver(H, lam, gamma, T_K, Nk, depth)

    if verbose:
        print(f"  HEOM: {solver._n_ados} ADOs, generator "
              f"{solver._n_ados * d * d} x {solver._n_ados * d * d}; "
              f"eigendecomposing (one-time)...", flush=True)

    A = solver.rhs(0).full()                       # constant generator
    w, V = np.linalg.eig(A)

    Vinv = np.linalg.inv(V)
    Vtop = V[:d * d, :]     # zeroth-ADO block projectors
    Vleft = Vinv[:, :d * d]

    maps = np.array([(Vtop * np.exp(w * (t * FS_TO_CM))[None, :]) @ Vleft
                     for t in t_fs])
    return maps


def standard_heom_pops(t_fs, rho0, H=H_FMO, lam=LAM_FMO, gamma=GAM_FMO, T_K=T_FMO, Nk=1, depth=3):
    """Reference dynamics from a direct HEOMSolver.run of the physical state."""

    solver = _make_solver(H, lam, gamma, T_K, Nk, depth)
    states = solver.run(qt.Qobj(np.asarray(rho0, complex)), np.asarray(t_fs) * FS_TO_CM).states

    return np.array([np.real(np.asarray(s.full()).diagonal()) for s in states])


def compute_and_save(outpath="data/heom_maps.npz", t_fs=None, Nk=1, depth=3):
    from grid import save_maps

    if t_fs is None:
        t_fs = np.arange(0.0, 1001.0, 10.0)

    maps = build_heom_maps(t_fs, Nk=Nk, depth=depth)
    save_maps(outpath, t_fs, maps, H_FMO, f"HEOM (Nk={Nk}, depth={depth})")
    print("saved", outpath, "  shape", maps.shape)


if __name__ == "__main__":
    compute_and_save()
