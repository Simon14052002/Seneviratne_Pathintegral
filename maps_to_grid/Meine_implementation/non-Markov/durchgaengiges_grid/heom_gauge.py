"""
Non-Markovian HEOM dynamics on ONE continuous quantum grid
==========================================================

The point of this module is *where the work happens*.  Every construction
below is performed once and its cost is independent of the number of time
steps: the classical computer never propagates the state, not even for a
single step.  All propagation -- for 100, 1000 or 10000 steps -- happens on
the grid by re-applying one fixed unitary.

Why the obvious route fails
---------------------------
The HEOM hierarchy is a *closed linear* system on the extended space of the
system density matrix plus all auxiliary density operators (ADOs),

    d/dtau x(tau) = A x(tau),        x = (vec rho_S, vec rho_1, ...),

so the exact one-step propagator P = exp(A dt) needs no memory kernel at all:
the ADOs carry the full non-Markovian memory and P^n is exact for every n.
This is what the transfer-tensor/companion construction throws away -- it
re-derives an approximate memory kernel from K classically computed short-time
maps, and everything beyond K dt is lost.

P cannot be put on a grid directly, though.  A Sz.-Nagy dilation embeds P/s
with s >= ||P||_2 into a unitary, and the post-selection probabilities
multiply, p_n = ||P^n x||^2 / (s^(2n) ||x||^2).  For the 4-site FMO model
||P||_2 = 32.39 while the spectral radius is exactly 1, so s^(2n) = 32^200
kills the circuit even though the dynamics itself is perfectly bounded.

The fix: a metric that makes A dissipative
------------------------------------------
||P||_2 = 32 is a statement about the *Euclidean* inner product, not about the
physics.  A is power-bounded, so by the Lyapunov/Sz.-Nagy similarity theorem
there is a positive definite W with

    (A - delta)^dag W + W (A - delta) <= 0                             (LYAP)

for every delta > 0 (delta > 0 is needed only because A has the steady state
as an exact zero eigenvalue).  In the gauge  x~ = W^(1/2) x  the generator
A~ = W^(1/2) A W^(-1/2) satisfies A~ + A~^dag <= 2 delta, hence

    ||P~||_2 = ||W^(1/2) exp(A dt) W^(-1/2)||_2 <= exp(delta dt),

i.e. P~ is a contraction up to a factor we choose.  The gauge is an exact
similarity transform, so the trajectory is *not* approximated at all.

Two things follow for free:

  * Sz.-Nagy with s = exp(delta dt) ~ 1 gives an O(1) success probability --
    what is left of it is the genuine physical decay of ||x(t)||, not an
    artefact of the representation.
  * Any Krylov compression H_m = Q_m^dag P~ Q_m of a contraction is again a
    contraction, ||H_m||_2 <= ||P~||_2, because Q_m is an isometry.  Arnoldi
    applied to the raw P instead produces ||H_m|| ~ 32 and a trajectory that
    diverges; in the gauge it is unconditionally stable for any m.

Pipeline
--------
    heom_generator      A          (sparse, from qutip, one call)
    contraction_gauge   W^(1/2)    (one Lyapunov solve, O(n^3), no dynamics)
    gauged_propagator   P~         (one matrix exponential)
    arnoldi             H_m, Q_m   (m matrix-vector products)
    dilate              U          (one unitary, the grid's only gate)
"""

import numpy as np
import qutip as qt
from scipy.linalg import expm, solve_continuous_lyapunov, eigh, sqrtm
from qutip.solver.heom import HEOMSolver, DrudeLorentzPadeBath


# --------------------------------------------------------------------------
# 1.  the HEOM generator
# --------------------------------------------------------------------------

def make_solver(*, H, lam, gamma, T_K, KB_CM, Nk=1, depth=3):
    """qutip HEOMSolver with one Drude-Lorentz Pade bath per site."""
    d = H.shape[0]
    baths = [DrudeLorentzPadeBath(qt.basis(d, i) * qt.basis(d, i).dag(),
                                  lam=lam, gamma=gamma, T=KB_CM * T_K, Nk=Nk)
             for i in range(d)]
    return HEOMSolver(qt.Qobj(H), baths, max_depth=depth)


def heom_generator(*, H, lam, gamma, T_K, KB_CM, FS_TO_CM, Nk=1, depth=3):
    """The full HEOM generator A on the system+ADO space.

    Returns (A, info).  A is n x n with n = n_ados * d^2; the first d^2
    coordinates are vec(rho_S) in column-stacking order.  FS_TO_CM is accepted
    so that the same MODEL dict can be splatted in everywhere.
    """
    solver = make_solver(H=H, lam=lam, gamma=gamma, T_K=T_K, KB_CM=KB_CM,
                         Nk=Nk, depth=depth)
    A = solver.rhs(0).full()
    d = H.shape[0]
    return A, dict(d=d, n_ados=solver._n_ados, n=A.shape[0], solver=solver)


# --------------------------------------------------------------------------
# 2.  the contraction gauge
# --------------------------------------------------------------------------

def contraction_gauge(A, *, delta, Q=None):
    """Solve (LYAP) and return the gauge matrices (Wh, Wih) = (W^1/2, W^-1/2).

    delta > 0 is a spectral shift in cm^-1.  It is the *only* knob: it bounds
    ||P~||_2 <= exp(delta*dt) from above and cond(W) ~ 1/delta from below, so
    it trades the residual post-selection loss exp(2*delta*T) over the whole
    run against the conditioning of the classical basis change.
    """
    n = A.shape[0]
    if Q is None:
        Q = np.eye(n, dtype=complex)
    Ad = A - delta * np.eye(n)
    # scipy solves  a x + x a^H = q; with a = Ad^dag this is Ad^dag W + W Ad = -Q
    W = solve_continuous_lyapunov(Ad.conj().T, -Q)
    W = 0.5 * (W + W.conj().T)
    w, V = eigh(W)
    if w.min() <= 0:
        raise np.linalg.LinAlgError(
            f"Lyapunov solution not positive definite (min eig {w.min():.3e}); "
            "increase delta or check that A is power bounded.")
    Wh = (V * np.sqrt(w)) @ V.conj().T
    Wih = (V / np.sqrt(w)) @ V.conj().T
    info = dict(delta=delta, eig_min=float(w.min()), eig_max=float(w.max()),
                cond=float(w.max() / w.min()))
    return Wh, Wih, info


def gauged_propagator(A, dt_cm, Wh, Wih):
    """P~ = W^(1/2) exp(A dt) W^(-1/2).  Exactly similar to P, but contractive."""
    P = expm(A * dt_cm)
    return Wh @ P @ Wih, P


# --------------------------------------------------------------------------
# 3.  Krylov compression  (contractive by construction in the gauge)
# --------------------------------------------------------------------------

def arnoldi(P, x0, m):
    """Orthonormal Krylov basis Q_m of span{x0, P x0, ...} and H_m = Q_m^dag P Q_m.

    Reorthogonalised twice; at m ~ 100 in a badly conditioned gauge the classical
    Gram-Schmidt loses orthogonality otherwise.
    """
    n = P.shape[0]
    Qb = np.zeros((n, m + 1), complex)
    H = np.zeros((m + 1, m), complex)
    Qb[:, 0] = x0 / np.linalg.norm(x0)
    mm = m
    for j in range(m):
        w = P @ Qb[:, j]
        for _ in range(2):
            h = Qb[:, :j + 1].conj().T @ w
            w -= Qb[:, :j + 1] @ h
            H[:j + 1, j] += h
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-13:
            mm = j + 1
            break
        Qb[:, j + 1] = w / H[j + 1, j]
    return H[:mm, :mm], Qb[:, :mm]


def safe_norm2(M):
    """Spectral norm, or nan if the SVD cannot converge.

    At the conditioning where the gauge breaks down (cond(W) ~ 1e19 for the
    strongly coupled model) LAPACK's SVD itself fails, so the validity check
    must not depend on it succeeding."""
    try:
        return float(np.linalg.norm(M, 2))
    except np.linalg.LinAlgError:
        return float('nan')


def dilate(M, s=None):
    """Sz.-Nagy: M/s as the top-left block of a unitary one qubit larger.

        U = [[ M/s ,  sqrt(I - M M^dag/s^2) ],
             [ sqrt(I - M^dag M/s^2) , -(M/s)^dag ]]

    Unitarity follows from the intertwining relation M^dag f(M M^dag) =
    f(M^dag M) M^dag.  In the gauge s = ||M||_2 is essentially 1.
    """
    n = M.shape[0]
    if s is None:
        s = float(np.linalg.norm(M, 2)) * (1 + 1e-12)
    Ms = M / s
    I = np.eye(n)
    B = sqrtm(I - Ms @ Ms.conj().T)
    C = sqrtm(I - Ms.conj().T @ Ms)
    B, C = 0.5 * (B + B.conj().T), 0.5 * (C + C.conj().T)
    return np.block([[Ms, B], [C, -Ms.conj().T]]), float(s)


# --------------------------------------------------------------------------
# 4.  the whole one-off construction
# --------------------------------------------------------------------------

def build_grid(dt_fs, *, rho0, delta=2.0, m=64, depth=3, Nk=1, verbose=True,
               **MODEL):
    """Everything the grid needs, from the model parameters alone.

    Returns a dict with
        U       the single unitary the circuit re-applies      (2*mp x 2*mp)
        s       its Sz.-Nagy scale, ~ exp(delta*dt)
        m, mp   Krylov dimension and its padding to a power of two
        y0      the initial Krylov coordinates (= ||x~0|| e_1)
        R       readout matrix, vec(rho_S)(t) = R @ y(t)       (d^2 x m)
        Hm, Qm, Wh, Wih, A, P      the classical objects, for diagnostics
    """
    d = MODEL['H'].shape[0]
    D = d * d
    dt_cm = dt_fs * MODEL['FS_TO_CM']

    A, info = heom_generator(depth=depth, Nk=Nk, **MODEL)
    if verbose:
        print(f"  A: {info['n']} x {info['n']} ({info['n_ados']} ADOs)", flush=True)

    Wh, Wih, ginfo = contraction_gauge(A, delta=delta)
    Pt, P = gauged_propagator(A, dt_cm, Wh, Wih)

    # Free validity check.  The gauge GUARANTEES ||P~||_2 <= exp(delta*dt); the
    # bound is saturated to six digits when the arithmetic holds up.  Exceeding
    # it means W^(1/2) and W^(-1/2) have lost precision -- forming W already
    # squares its condition number, so a strongly transient generator (large
    # ||P||_2) can push cond(W) past what float64 carries.  The dynamics is then
    # still right, but the post-selection probability collapses.
    bound = float(np.exp(delta * dt_cm))
    norm_Pt = safe_norm2(Pt)
    ginfo.update(bound=bound, norm_P=safe_norm2(P), norm_Pt=norm_Pt,
                 trustworthy=bool(norm_Pt <= bound * (1 + 1e-6)))
    if verbose:
        print(f"  gauge delta={delta} cm^-1: cond(W)={ginfo['cond']:.2e}, "
              f"||P||={ginfo['norm_P']:.4f} -> ||P~||={norm_Pt:.6f} "
              f"(bound {bound:.6f})", flush=True)
    if not ginfo['trustworthy']:
        import warnings
        warnings.warn(
            f"gauge lost precision: ||P~||_2 = {norm_Pt:.4g} exceeds the "
            f"guaranteed bound exp(delta*dt) = {bound:.6g} by a factor "
            f"{norm_Pt / bound:.3g}. cond(W) = {ginfo['cond']:.2e} is too large "
            f"for float64. Raise delta, shrink dt, or reduce the transient "
            f"growth of the generator; the post-selection probability from this "
            f"grid is not usable.", RuntimeWarning, stacklevel=2)

    x0 = np.zeros(info['n'], complex)
    x0[:D] = np.asarray(rho0, complex).flatten(order='F')
    xt0 = Wh @ x0

    Hm, Qm = arnoldi(Pt, xt0, m)
    m = Hm.shape[0]
    mp = 2 ** int(np.ceil(np.log2(m)))
    Apad = np.zeros((mp, mp), complex)
    Apad[:m, :m] = Hm
    U, s = dilate(Apad)
    if verbose:
        print(f"  Krylov m={m} -> {int(np.log2(mp)) + 1} qubits, "
              f"||H_m||={np.linalg.norm(Hm, 2):.6f}, s={s:.6f}, "
              f"s^100={s ** 100:.4f}", flush=True)

    y0 = np.zeros(m, complex)
    y0[0] = np.linalg.norm(xt0)
    R = (Wih @ Qm)[:D, :]

    return dict(U=U, s=s, m=m, mp=mp, n_qubits=int(np.log2(mp)) + 1,
                y0=y0, R=R, Hm=Hm, Qm=Qm, Wh=Wh, Wih=Wih, A=A, P=P, Pt=Pt,
                d=d, D=D, dt_fs=dt_fs, dt_cm=dt_cm, gauge=ginfo, heom=info)


# --------------------------------------------------------------------------
# 5.  readout
# --------------------------------------------------------------------------

def pops_from_y(y, R, d):
    """Populations from Krylov coordinates: vec(rho) = R y, then take the diagonal."""
    return np.real(np.diag((R @ y).reshape(d, d, order='F')))


def reference_trajectory(grid, n_steps, exact=False):
    """Classical yardstick.  exact=True iterates the full gauged propagator P~
    (no Krylov truncation); otherwise it iterates H_m, which is what the grid
    does.  Used only to validate the circuit -- the grid never needs it."""
    d, R = grid['d'], grid['R']
    if exact:
        x = grid['Qm'] @ grid['y0']          # == x~0, since Qm[:,0] = x~0/||x~0||
        Rf = (grid['Wih'])[:grid['D'], :]
        out = [np.real(np.diag((Rf @ x).reshape(d, d, order='F')))]
        for _ in range(n_steps):
            x = grid['Pt'] @ x
            out.append(np.real(np.diag((Rf @ x).reshape(d, d, order='F'))))
        return np.array(out)
    y = grid['y0'].copy()
    out = [pops_from_y(y, R, d)]
    for _ in range(n_steps):
        y = grid['Hm'] @ y
        out.append(pops_from_y(y, R, d))
    return np.array(out)


def qutip_reference(t_fs, *, rho0, depth=3, Nk=1, **MODEL):
    """The only benchmark used in this folder: qutip's own HEOMSolver."""
    solver = make_solver(H=MODEL['H'], lam=MODEL['lam'], gamma=MODEL['gamma'],
                         T_K=MODEL['T_K'], KB_CM=MODEL['KB_CM'], Nk=Nk, depth=depth)
    res = solver.run(qt.Qobj(np.asarray(rho0)), np.asarray(t_fs) * MODEL['FS_TO_CM'])
    return np.array([np.real(st.diag()) for st in res.states])


# --------------------------------------------------------------------------
# 6.  caching  (the one-off construction takes a couple of minutes)
# --------------------------------------------------------------------------

_SMALL = ('U', 's', 'm', 'mp', 'n_qubits', 'y0', 'R', 'Hm', 'd', 'D',
          'dt_fs', 'dt_cm')


def save_grid(path, grid):
    """Store only what the circuit and the readout need -- a few hundred kB,
    instead of the 2640 x 2640 objects used to build them."""
    small = {k: np.asarray(grid[k]) for k in _SMALL}
    small['gauge'] = np.array(grid['gauge'], dtype=object)
    small['heom'] = np.array({k: v for k, v in grid['heom'].items()
                              if k != 'solver'}, dtype=object)
    np.savez_compressed(path, **small)


def load_grid(path):
    z = np.load(path, allow_pickle=True)
    g = {k: z[k] for k in z.files}
    for k in ('s', 'dt_fs', 'dt_cm'):
        g[k] = float(g[k])
    for k in ('m', 'mp', 'n_qubits', 'd', 'D'):
        g[k] = int(g[k])
    for k in ('gauge', 'heom'):
        g[k] = g[k].item()
    return g


# --------------------------------------------------------------------------
# 7.  the baseline this folder replaces
# --------------------------------------------------------------------------

def companion_baseline(P, K, d, rho0, n_steps):
    """Transfer-tensor / companion propagator, for comparison only.

    The reduced maps L_n = (P^n)[:D, :D] are read off the exact propagator,
    turned into transfer tensors T_n = L_n - sum_{m<n} T_m L_{n-m} (Cerrillo &
    Cao, PRL 112, 110401 (2014)) and packed into the block companion operator

        E = [[T_1 T_2 ... T_K],
             [ I   0  ...  0 ],
             [ 0   I  ...  0 ], ...]

    which acts on a K-step history window.  Two costs are structural: the first
    K states must be produced classically to fill the window, and every
    correlation reaching further back than K dt is discarded.  Both disappear in
    the ADO formulation, where the memory is carried by the ADOs themselves.
    """
    D = d * d
    X = np.zeros((P.shape[0], D), dtype=complex)
    X[:D, :] = np.eye(D)
    maps = [np.eye(D, dtype=complex)]
    for _ in range(K):
        X = P @ X
        maps.append(X[:D, :].copy())

    T = []
    for n in range(1, K + 1):
        Tn = np.array(maps[n], dtype=complex, copy=True)
        for mm in range(1, n):
            Tn -= T[mm - 1] @ maps[n - mm]
        T.append(Tn)

    E = np.zeros((K * D, K * D), dtype=complex)
    for mm, Tm in enumerate(T):
        E[:D, mm * D:(mm + 1) * D] = Tm
    for r in range(1, K):
        E[r * D:(r + 1) * D, (r - 1) * D:r * D] = np.eye(D)

    v0 = np.asarray(rho0, complex).flatten(order='F')
    hist = [maps[n] @ v0 for n in range(min(K, n_steps + 1))]   # classical steps!
    pops = np.zeros((n_steps + 1, d))
    for i, h in enumerate(hist):
        pops[i] = np.real(np.diag(h.reshape(d, d, order='F')))
    X = np.concatenate(hist[::-1])
    for t in range(K, n_steps + 1):
        X = E @ X
        pops[t] = np.real(np.diag(X[:D].reshape(d, d, order='F')))
    return pops, E


def post_selection_curve(M, x0, n_steps, s):
    """p_n = ||M^n x||^2 / (s^(2n) ||x||^2): what a step-wise Sz.-Nagy dilation
    of M with scale s actually costs.  This is the whole story of the gauge --
    the same physical trajectory, read with s = ||P||_2 = 32 or with s ~ 1."""
    x = np.asarray(x0, complex)
    n0 = np.linalg.norm(x)
    out = [1.0]
    for n in range(1, n_steps + 1):
        x = M @ x
        with np.errstate(over='ignore', under='ignore'):
            out.append(float((np.linalg.norm(x) / (s ** n * n0)) ** 2))
    return np.array(out)


def regrid(grid, m, *, rho0=None, verbose=False):
    """Re-compress an existing gauge at a different Krylov dimension.

    The expensive part of build_grid -- the Lyapunov solve and the matrix
    exponential -- depends only on the model, not on m and not on the initial
    state, so a scan over either reuses it and costs only m matrix-vector
    products per point.  Pass rho0 to rebuild the Krylov basis for a different
    initial state; the gauge W is unchanged.
    """
    if rho0 is None:
        xt0 = grid['Qm'][:, 0] * np.linalg.norm(grid['y0'])
    else:
        x0 = np.zeros(grid['Wh'].shape[0], complex)
        x0[:grid['D']] = np.asarray(rho0, complex).flatten(order='F')
        xt0 = grid['Wh'] @ x0
    Hm, Qm = arnoldi(grid['Pt'], xt0, m)
    m = Hm.shape[0]
    mp = 2 ** int(np.ceil(np.log2(m)))
    Apad = np.zeros((mp, mp), complex)
    Apad[:m, :m] = Hm
    U, s = dilate(Apad)
    y0 = np.zeros(m, complex)
    y0[0] = np.linalg.norm(xt0)
    out = dict(grid)
    out.update(U=U, s=s, m=m, mp=mp, n_qubits=int(np.log2(mp)) + 1,
               y0=y0, R=(grid['Wih'] @ Qm)[:grid['D'], :], Hm=Hm, Qm=Qm)
    if verbose:
        print(f"  m={m} -> {out['n_qubits']} qubits, "
              f"||H_m||={np.linalg.norm(Hm, 2):.6f}", flush=True)
    return out
