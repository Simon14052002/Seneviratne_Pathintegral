"""
Lindblad (secular Redfield / GKSL) dynamical map  L(t) = exp(L_Lindblad * t),
using the jump-operator construction from the Master's thesis (S. Mader,
Sec. 6; `calc_L_list` in figure6.21.py).  Each site couples to a Drude bath;
the jump operators are built in the system eigenbasis with rates set by the
one-sided bath spectrum  2 Re tau(omega)  (detailed balance).

NORMAL-pipeline copy: produces data/lindblad_maps.npz, one map per saved
time, consumed by ./grid.py's populations_via_kraus. For the enhanced
algorithm see ../enhanced/lindblad_map.py.

Provides:
  calc_L_list(...)             -> the GKSL jump operators L_k (thesis method)
  lindblad_superop(H, L_list)  -> the Liouvillian L (column-stacking)
  build_lindblad_maps(...)     -> t_fs, maps  (L(t) = expm(L * t))
  standard_lindblad_pops(...)  -> populations from a direct qutip mesolve
  compute_and_save(...)        -> writes data/lindblad_maps.npz

Note: because L(t)=exp(L t) is exactly what mesolve integrates, the Kraus
dynamics from this map and the 'standard' mesolve agree to machine precision;
the physically interesting comparison is Lindblad vs the exact HEOM / path
integral (Lindblad is the Markovian approximation of the same bath).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # let 'from grid import ...' work from any cwd

import numpy as np
from scipy.linalg import expm

# physical constants inlined (energies in cm^-1, time tau = 2*pi*c*t in cm) so
# this module has no cross-file import for the IDE's auto-organiser to rewrite
_C_CM = 2.99792458e10
FS_TO_CM = 1e-15 * 2 * np.pi * _C_CM
KB_CM = 0.6950348004

# 4-site FMO Hamiltonian (Seneviratne eq. 30), cm^-1  -- same as the others
H_FMO = np.array([[12375.0, -87.7,   5.5,  -5.9],
                  [-87.7, 12495.0,  30.8,   8.2],
                  [5.5,      30.8, 12175.0, -53.4],
                  [-5.9,      8.2, -53.4, 12285.0]])
H_FMO = H_FMO - np.trace(H_FMO) / 4 * np.eye(4)
LAM_FMO, GAM_FMO, T_FMO = 35.0, 106.18, 300.0


def calc_L_list(H, lam=LAM_FMO, gamma=GAM_FMO, T_K=T_FMO):
    """GKSL jump operators, thesis construction (figure6.21.py:calc_L_list).
    One local site-projector coupling per site; rate 2 Re tau(omega)."""
    kBT = KB_CM * T_K
    d = H.shape[0]

    def Re_tau(omega):
        if abs(omega) < 1e-12:
            return 2 * kBT * lam / gamma                       # cm^-1
        J = 2 * lam * omega * gamma / (omega ** 2 + gamma ** 2)
        n = 1.0 / (np.exp(omega / kBT) - 1.0)
        return J * (n + 1.0)

    eks, V = np.linalg.eigh(H)                                 # V[:,M] = |v_M>
    # Bohr frequencies: all ordered pairs (M != N), plus 0 (pure dephasing)
    w_diff = [eks[M] - eks[N] for M in range(d) for N in range(d)
              if M != N] + [0.0]

    L_list = []
    for m in range(d):                                        # site projector
        for w in w_diff:
            Lmw = np.zeros((d, d), dtype=complex)
            for M in range(d):
                for N in range(d):
                    if abs((eks[M] - eks[N]) - w) < 1e-9:
                        amp = np.conj(V[m, M]) * V[m, N]       # <v_M|m><m|v_N>
                        Lmw += amp * np.outer(V[:, M], np.conj(V[:, N]))
            gam = 2 * Re_tau(w)
            if gam > 0:
                L_list.append(np.sqrt(gam) * Lmw)
    return L_list


def lindblad_superop(H, L_list):
    """Liouvillian L (D x D, D=d^2) in the column-stacking convention:
    vec(A rho B) = (B^T (x) A) vec(rho)."""
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))            # -i[H, .]
    for Lk in L_list:
        LdL = Lk.conj().T @ Lk
        Lsup += (np.kron(Lk.conj(), Lk)
                 - 0.5 * np.kron(I, LdL)
                 - 0.5 * np.kron(LdL.T, I))
    return Lsup


def build_lindblad_maps(t_fs, H=H_FMO, lam=LAM_FMO, gamma=GAM_FMO, T_K=T_FMO):
    """Lindblad maps L(t) = exp(L * t) on the grid t_fs."""
    Lsup = lindblad_superop(H, calc_L_list(H, lam, gamma, T_K))
    return np.array([expm(Lsup * (t * FS_TO_CM)) for t in t_fs])


def standard_lindblad_pops(t_fs, rho0, H=H_FMO, lam=LAM_FMO, gamma=GAM_FMO,
                           T_K=T_FMO):
    """Reference dynamics from a direct qutip mesolve with the same L_list."""
    import qutip as qt
    L_list = calc_L_list(H, lam, gamma, T_K)
    res = qt.mesolve(qt.Qobj(H), qt.Qobj(np.asarray(rho0, complex)),
                     np.asarray(t_fs) * FS_TO_CM,
                     c_ops=[qt.Qobj(Lk) for Lk in L_list])
    return np.array([np.real(np.asarray(s.full()).diagonal())
                     for s in res.states])


def compute_and_save(outpath="data/lindblad_maps.npz", t_fs=None):
    from grid import save_maps
    if t_fs is None:
        t_fs = np.arange(0.0, 1001.0, 10.0)
    maps = build_lindblad_maps(t_fs)
    save_maps(outpath, t_fs, maps, H_FMO, "Lindblad (thesis, secular Redfield)")
    print("saved", outpath, "  shape", maps.shape)


if __name__ == "__main__":
    compute_and_save()
