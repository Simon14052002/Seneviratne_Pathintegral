import numpy as np
import qutip as qt
from scipy.linalg import expm
from qutip.solver.heom import HEOMSolver, DrudeLorentzPadeBath
from scipy.sparse.linalg import expm_multiply


def _make_solver(H, lam, gamma, T_K, KB_CM, Nk, depth):
    d = H.shape[0]
    baths = [DrudeLorentzPadeBath(qt.basis(d, i) * qt.basis(d, i).dag(), lam=lam, gamma=gamma, T=KB_CM * T_K, Nk=Nk)
             for i in range(d)]
    return HEOMSolver(qt.Qobj(H), baths, max_depth=depth)



def heom_onestep_propagator(dt_fs, *, H, lam, gamma, T_K, KB_CM, FS_TO_CM,
                            Nk=1, depth=3, verbose=True):
    """ONE-step propagator P = exp(L_HEOM * dt) on the extended system+ADO
    space, computed once. Iterating it and reading the zeroth-ADO block
    (the first d^2 entries, with the ADOs initialised to zero) is exact for
    arbitrarily long times. Returns (P, n_ados)."""
    d = H.shape[0]
    solver = _make_solver(H, lam, gamma, T_K, KB_CM, Nk, depth)

    if verbose:
        print(f"  one-step P: {solver._n_ados} ADOs, "
              f"{solver._n_ados * d * d} x {solver._n_ados * d * d} "
              f"(one-time matrix exponential)...", flush=True)

    # Holen der vollen Superoperator-Matrix aus dem Solver
    A = solver.rhs(0).full()

    # Direktes Berechnen von exp(A * dt) mittels Padé-Approximation
    dt_cm = dt_fs * FS_TO_CM
    P = expm(A * dt_cm)

    return P, solver._n_ados

def heom_shorttime_maps(P, K, d):
    """The reduced maps L(0), L(dt), ..., L(K dt) extracted from the ONE
    propagator P -- no further HEOM solve, just K matrix-vector products.

    Column j of L(m dt) is the zeroth-ADO block reached after m applications
    of P from the initial condition (system = basis element E_j, all ADOs
    zero), which is exactly qHEOM Eq. 33 evaluated at t = m dt.

    These K maps are what grid.ipynb turns into the transfer tensors and the
    one fixed companion operator that the enhanced algorithm re-applies."""
    D = d * d
    X = np.zeros((P.shape[0], D), dtype=complex)
    X[:D, :] = np.eye(D)                       # ADOs start at zero
    maps = [np.eye(D, dtype=complex)]
    for _ in range(K):
        X = P @ X
        maps.append(X[:D, :].copy())    # Wir hängen nur die ersten D Zeilen (=rho_S) an, die restlichen Zeilen enthalten ADOs
    return np.array(maps)



def transfer_tensors(maps, K):
    """Transfer tensors T_1..T_K from the short-time maps L_1..L_K only."""
    T = []
    for n in range(1, K + 1):                              # maps[0] = I, T[0] = T1
        Tn = np.array(maps[n], dtype=complex, copy=True)   # Set: Ti = maps[i] = Li -> T1 = L1,...
        for m in range(1, n):                              # n=2: T2 = L2 - T[0] @ maps[1] = T1 @ L1
            Tn -= T[m - 1] @ maps[n - m]                   # n=3: T3 = L3 - T[0] @ maps[2] - T[1] @ maps[1] = L3 - T1 @ L2 - T2 @ L1
        T.append(Tn)
    return T


def companion_propagator(T):
    """The ONE fixed one-step operator E on the enlarged (K*D-dim) register:
    first block row = (T_1 ... T_K), lower rows shift the history window."""
    K, D = len(T), T[0].shape[0]
    E = np.zeros((K * D, K * D), dtype=complex)
    for m, Tm in enumerate(T):
        E[:D, m * D:(m + 1) * D] = Tm
    for r in range(1, K):
        E[r * D:(r + 1) * D, (r - 1) * D:r * D] = np.eye(D)
    return E


def calculate_heom_propagator(dt_fs, HEOM_depth=3, K_heom=25, N=4, **MODEL):
    P, n_ado = heom_onestep_propagator(dt_fs, **MODEL, depth=HEOM_depth, verbose=False)

    maps_heom = heom_shorttime_maps(P, K_heom, N) 
    maps = maps_heom["maps"] if isinstance(maps_heom, dict) else np.asarray(maps_heom)

    T = transfer_tensors(maps, K_heom)
    E = companion_propagator(T)  

    return E, K_heom, maps_heom 