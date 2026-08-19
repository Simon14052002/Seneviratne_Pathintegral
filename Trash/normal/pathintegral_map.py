"""
Exact non-Markovian linear map  eps_{t,0}  (and its matrix representation
L(t), eq. 12 of the paper) for a multi-state system bilinearly coupled to
independent harmonic baths, computed from Feynman's path integral with the
(discretized) Feynman-Vernon influence functional.

This reimplements the strategy of
  Seneviratne, Walters & Wang, ACS Omega 2024, 9, 9666  (Sections II & III),
where the exact superoperator is generated with a tensor-network path
integral (TNPI, Bose & Walters, arXiv:2106.12523).  Here the augmented
path-amplitude tensor is stored as a matrix product state and propagated
TEMPO-style (Strathearn et al., Nat. Commun. 9, 3322 (2018)), with the
initial forward-backward index kept open so that every step directly yields
the full d^2 x d^2 map matrix L(t_k):

    vec(rho(t_k)) = L(t_k) vec(rho(0))        (column-stacking convention)

Model assumptions (as in the paper's FMO example):
  * each system state |j><j| couples to its own bath (identical spectral
    densities), i.e. coupling operators are the site projectors, which are
    all diagonal in the site basis;
  * factorized initial condition rho(0) x exp(-beta H_bath)/Z.

Units follow the Figures-folder conventions: energies in cm^-1, hbar = 1,
time variable tau = 2*pi*c*t so that phases are E[cm^-1]*tau[cm].

This is the NORMAL-pipeline copy of the path-integral map: it produces
data/pathintegral_maps.npz consumed by ./grid.py's populations_via_kraus
(fresh Choi/Kraus decomposition at every saved time). For the "compute once,
reuse forever" enhanced algorithm see ../enhanced/pathintegral_map.py.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # let 'from grid import ...' work from any cwd

import numpy as np
from scipy.linalg import expm, svd

C_CM = 2.99792458e10                  # speed of light [cm/s]
FS_TO_CM = 1e-15 * 2 * np.pi * C_CM   # tau[cm] per t[fs]   (= 1.8836516e-4)
KB_CM = 0.6950348004                  # Boltzmann constant [cm^-1 / K]


# --------------------------------------------------------------------------
# Bath: Drude spectral density  J(w) = 2*lam*gamma*w / (w^2 + gamma^2)
# --------------------------------------------------------------------------

def drude_expansion(lam, gamma, beta, n_mats=10000):
    """Exponential decomposition of the bath response function (eq. 10),
    C(tau) = (1/pi) int dw J(w) [coth(beta w/2) cos(w tau) - i sin(w tau)]
           = sum_m a_m exp(-nu_m tau)   for tau >= 0  (Matsubara series).

    Zerlegt die komplexe Bad-Korrelationsfunktion C(tau) in eine Summe von
    abfallenden Exponentialfunktionen. Das ist mathematisch noetig, um die
    Integrale fuer das Influence Functional (eta) analytisch loesen zu koennen.
    """
    n = np.arange(1, n_mats + 1)
    nu = 2 * np.pi * n / beta
    a = (4 * lam * gamma / beta) * nu / (nu ** 2 - gamma ** 2)
    a0 = lam * gamma * (1.0 / np.tan(beta * gamma / 2) - 1j)
    return np.concatenate([[a0], a]), np.concatenate([[gamma], nu])


def bath_correlation(tau, a, nu):
    """C(tau) for tau >= 0 from the exponential decomposition."""
    tau = np.atleast_1d(tau)
    return np.sum(a[None, :] * np.exp(-nu[None, :] * tau[:, None]), axis=1)


def eta_coefficients(a, nu, dtau, kmax):
    """
    Parameters:
        a: Die Amplituden (Gewichte) $a_m$ der exponentiellen Zerlegung aus der Matsubara-Expansion.
        nu: Die Dämpfungsraten (Frequenzen) nu_m der Bad-Moden.
        dtau: Die diskrete Zeitscheiben-Länge Delta_tau (Grid-Größe) in den passenden physikalischen Einheiten (hier in Zentimetern laut Konvention)
        kmax: Die maximale Gedächtnislänge K (Fenstergröße)

    Discretized influence-functional coefficients (QUAPI/TEMPO form):
       eta[0] = int_0^dt ds int_0^s  ds' C(s - s')
       eta[k] = int_0^dt ds int_0^dt ds' C(k*dt + s - s'),  k >= 1
    evaluated analytically for C(tau) = sum_m a_m exp(-nu_m tau).

    Berechnet die eta-Werte fuer die nicht-lokalen Zeit-Tiles.
    eta[0] ist die rein lokale Bad-Wirkung am selben Zeitschritt.
    eta[k] ist das "Gedaechtnis" ueber k Zeitschritte hinweg (Kopplung
    zwischen alpha_m und alpha_{m-k}).
    """
    eta = np.empty(kmax + 1, dtype=complex)
    x = nu * dtau
    eta[0] = np.sum(a * (x + np.expm1(-x)) / nu ** 2)   # np.expm1(x) berechnet e^x - 1 mit einer Taylorreihe um keine Nachkommastellen zu verlieren für sehr kleine x
    base = a * np.expm1(-x) ** 2 / nu ** 2      # a_m (1 - e^{-nu dt})^2 / nu^2
    for k in range(1, kmax + 1):
        eta[k] = np.sum(base * np.exp(-nu * (k - 1) * dtau))
    return eta


# --------------------------------------------------------------------------
# Influence-functional matrices in the forward-backward (FB) index
#   alpha = s_plus + d * s_minus   (column-stacking of the density matrix)
# For site-projector couplings to identical independent baths the sum over
# baths collapses to Kronecker deltas between path labels:
#   Phi_D(a, a') = -[ eta_D d(s+,s'+) - eta_D* d(s+,s'-)
#                    - eta_D d(s-,s'+) + eta_D* d(s-,s'-) ]
# --------------------------------------------------------------------------

def influence_matrices(eta, d=4):
    """
    Wandelt die abstrakten eta-Skalare in reale Einfluss-Tensoren I_D um.
    Da wir im Liouville-Raum sind, kombiniert alpha den Vorwaerts- und
    Rueckwaertspfad (s_plus, s_minus). Die Funktion gibt eine Liste von
    Matrizen zurueck, wobei mats[k] der Tensor I_k(alpha_k, alpha_{k-k'}) ist.
    """
    D = d * d   # Gesamtdimension des Liouville-Raums (z.B. 4*4 = 16)

    # sp (s_plus) extrahiert den Vorwärts-Zustand (Ket) für jeden der D Super-Indizes.
    # Beispiel für d=4: [0, 1, 2, 3, 0, 1, 2, 3, ...] -> Modulo-Operation
    sp = np.arange(D) % d

    # sm (s_minus) extrahiert den Rückwärts-Zustand (Bra) für jeden der D Super-Indizes.
    # Beispiel für d=4: [0, 0, 0, 0, 1, 1, 1, 1, ...] -> Ganzzahl-Division
    sm = np.arange(D) // d

    # --- HIER STECKEN DIE DELTA-FUNKTIONEN (Kronecker-Deltas) ---
    # Durch das Broadcasting [:, None] == [None, :] entsteht jeweils eine D x D Matrix.
    # Ein Eintrag (i, j) ist 1.0 wenn die Bedingung wahr ist, sonst 0.0.
    dpp = (sp[:, None] == sp[None, :]).astype(float)     # dpp = delta(s+_später, s+_früher) -> Kopplung Vorwärts mit Vorwärts
    dpm = (sp[:, None] == sm[None, :]).astype(float)     # dpm = delta(s+_später, s-_früher) -> Kopplung Vorwärts mit Rückwärts
    dmp = (sm[:, None] == sp[None, :]).astype(float)     # dmp = delta(s-_später, s+_früher) -> Kopplung Rückwärts mit Vorwärts
    dmm = (sm[:, None] == sm[None, :]).astype(float)     # dmm = delta(s-_später, s-_früher) -> Kopplung Rückwärts mit Rückwärts

    mats = []
    # Schleife über alle eta-Koeffizienten (für jeden zeitlichen Abstand k)
    for e in eta:
        # phi berechnet den Exponenten der Feynman-Vernon-Formel. Hier werden die Delta-Matrizen mit den komplexen eta-Werten skaliert.
        phi = -(e * dpp - np.conj(e) * dpm - e * dmp + np.conj(e) * dmm)
        mats.append(np.exp(phi)) # I = np.exp(Phi). np.exp wird elementweise auf die D x D Matrix angewendet. Form: I[alpha_später, alpha_früher]

    return mats


# --------------------------------------------------------------------------
# MPS helpers
# --------------------------------------------------------------------------

_rng = np.random.default_rng(12345)


def _truncate_rank(s, eps, chi_max, total2):
    """Number of singular values to keep so that the discarded Frobenius
    weight (incl. any uncomputed tail, total2 - sum s^2) stays <= eps^2.

    Implementiert das Eckart-Young Kriterium zum Finden der effektiven
    Bond-Dimension chi. Wirft alle kleinsten Singulaerwerte weg, solange
    die Summe ihrer Quadrate (der Fehler) unter dem Schwellenwert eps liegt.
    Der unberechnete 'tail' (aus der rSVD) wird fairerweise mitgezaehlt.
    """
    s2 = s ** 2
    tail = max(total2 - s2.sum(), 0.0)
    csum = np.cumsum(s2[::-1]) + tail
    ndiscard = np.searchsorted(csum, eps ** 2 * total2, side='right')
    keep = max(1, s.size - ndiscard)
    keep = min(keep, int(np.sum(s > 1e-14 * s[0])) or 1)
    return min(keep, chi_max)  # Hartes Limit bei chi_max


def _full_svd(M):
    """Klassische, exakte Singulaerwertzerlegung als Fallback."""
    try:
        return svd(M, full_matrices=False, lapack_driver='gesdd')
    except np.linalg.LinAlgError:
        return svd(M, full_matrices=False, lapack_driver='gesvd')


def _svd_split(M, eps, chi_max, rank_hint=None):
    """Truncated SVD of M with relative Frobenius-norm cutoff eps.
    Uses a randomized range-finder (one power iteration) when M is large
    and a modest rank is expected; falls back to exact LAPACK SVD.

    Das ist die Halko-Martinsson-Tropp rSVD Implementierung.
    """
    m, n = M.shape
    total2 = np.linalg.norm(M) ** 2  # Exakte Gesamtmasse fuer den Tail-Term

    # Prueft, ob die Matrix gross genug ist, dass sich rSVD ueberhaupt lohnt
    if eps > 0 and min(m, n) > 512 and rank_hint is not None:
        Mh = M.conj().T
        k = min(int(1.3 * rank_hint) + 64, min(m, n), chi_max + 32) # Oversampling

        while k < 0.4 * min(m, n):
            # 1. Skizze (Random Gauss Matrix G)
            G = (_rng.standard_normal((n, k))
                 + 1j * _rng.standard_normal((n, k)))

            # 2. Power Iteration (Verstaerkt dominante Sigma-Werte)
            Y = M @ (Mh @ (M @ G))            # one power iteration

            # 3. Range-Finder (Findet die Orthonormalbasis Q)
            Q, _ = np.linalg.qr(Y)

            # 4. Projektion auf kleinen Unterraum B und exakte SVD von B
            B = Q.conj().T @ M
            Ub, s, Vh = _full_svd(B)
            keep = _truncate_rank(s, eps, chi_max, total2)

            # Adaptivitaet: War unsere Schaetzung 'k' gut genug?
            if keep < k - 8 or k >= chi_max:   # tail captured -> accept
                return (Q @ Ub)[:, :keep], s[:keep], Vh[:keep]

            # Wenn nicht, verdopple die Skizzen-Groesse und versuche es erneut
            k = min(2 * k, min(m, n))          # sketch too small: retry

    # Fallback: Matrix ist zu klein fuer rSVD, mache exakte SVD
    U, s, Vh = _full_svd(M)
    keep = _truncate_rank(s, eps, chi_max, total2) if s.size else 1
    return U[:, :keep], s[:keep], Vh[:keep]


def _compress(mps, eps, chi_max):
    """Left-to-right QR sweep, then right-to-left truncating SVD sweep.

    Nutzt die Gauge-Freiheit (XX^-1 Trick), um den Tensor global zu optimieren.
    """
    n = len(mps)
    # 1. QR-Sweep (Links nach Rechts):
    # Macht alle linken Tensoren isometrisch und schiebt Gewichte/Muell als R nach rechts.
    for i in range(n - 1):
        l, p, r = mps[i].shape
        Q, R = np.linalg.qr(mps[i].reshape(l * p, r))
        mps[i] = Q.reshape(l, p, -1)
        mps[i + 1] = np.tensordot(R, mps[i + 1], axes=([1], [0]))

    # 2. SVD-Sweep (Rechts nach Links):
    # Da links nun alles isometrisch ist, ist der lokale Fehler bei der SVD-Trunkierung
    # garantiert identisch mit dem globalen Fehler der gesamten Wellenfunktion.
    for i in range(n - 1, 0, -1):
        l, p, r = mps[i].shape
        U, s, Vh = _svd_split(mps[i].reshape(l, p * r), eps, chi_max)
        mps[i] = Vh.reshape(-1, p, r)
        mps[i - 1] = np.tensordot(mps[i - 1], U * s[None, :], axes=([2], [0]))
    return mps


# --------------------------------------------------------------------------
# TEMPO propagation of the map
# --------------------------------------------------------------------------

class PathIntegralMap:
    """Computes L(t_k) = matrix of eps_{t_k,0} on the grid t_k = k*dt_fs."""

    def __init__(self, H_cm, dt_fs, kmax, lam_cm, gamma_cm, T_K,
                 n_mats=200000, eps=1e-8, chi_max=256):
        self.d = H_cm.shape[0]
        self.D = self.d ** 2
        self.kmax = int(kmax)
        self.eps = eps
        self.chi_max = chi_max

        dtau = dt_fs * FS_TO_CM
        beta = 1.0 / (KB_CM * T_K)
        a, nu = drude_expansion(lam_cm, gamma_cm, beta, n_mats)
        self.eta = eta_coefficients(a, nu, dtau, self.kmax)
        self.Imats = influence_matrices(self.eta, self.d)
        self.I0 = np.diag(self.Imats[0]).copy()   # same-slice factor (vector)

        Uh = expm(-1j * H_cm * dtau / 2)
        self.Ph = np.kron(Uh.conj(), Uh)          # half-step FB propagator
        self.P = self.Ph @ self.Ph                # full-step FB propagator

    # -- read-out: contract MPS, apply trailing half propagator ------------
    def _readout(self, mps):
        """
        Zieht das gesamte Netzwerk zusammen (kontrahiert alle internen Bonds c),
        sodass nur noch alpha_0 (Anfang) und alpha_last (Ende) uebrig bleiben.
        Am Ende wird der End-Halbschritt Ph multipliziert, was exakt der
        Liouville-Propagator-Matrix (16x16) fuer diesen Zeitschritt entspricht.
        """
        env = None
        for site in mps[:-1]:
            v = site.sum(axis=1)                  # sum physical index
            env = v if env is None else env @ v
        G = mps[-1][:, :, 0]                      # (chi_l, alpha_last)
        Lpre = G if env is None else env @ G      # (alpha_0, alpha_last)
        return self.Ph @ Lpre.T                   # L[beta, alpha_0]

    # -- one propagation step ----------------------------------------------
    # Gauge invariant: on entry the MPS is right-canonical with the
    # orthogonality centre at site 0 (this is what the final SVD sweep of
    # the previous step leaves behind).  The influence factors
    # I_D(alpha_new, alpha_old) and the creation of the new slice are then
    # applied in a single left-to-right zip-up pass, so every truncation
    # happens at a proper mixed-canonical cut.  The value l of the new
    # slice index is carried along the pass and closed at the right end.
    def _step(self, mps):
        """
        Die Herzstueck-Funktion ("Zip-Up" Algorithmus).
        Fuegt den neuen Zeitschritt alpha_k hinzu und baut die neuen
        Tensoren auf. Dies geschieht nach dem "Ziehharmonika"-Prinzip:
        Wir multiplizieren die Matrizen und trunkieren sofort per SVD auf chi,
        noch BEVOR die naechste Operation den Zustandsraum sprengen kann.
        So existiert das 16^K Bond niemals im Arbeitsspeicher.
        """
        D = self.D
        n_old = len(mps)                          # distances D = n_old .. 1
        # combined nearest factor: W[b, a] = I_1[b,a] P[b,a] I0[b]
        # Dies ist das W(alpha_k, c_{k-1}) Gewicht fuer den allerkuerzesten Pfad-Schritt
        W = self.Imats[1] * self.P * self.I0[:, None]

        if n_old == 1:
            # Spezialfall: Der allererste Schritt.
            G = mps[0][:, :, 0]                   # (w, a)
            Y = G[:, :, None] * W.T[None, :, :]   # (w, a, b)
            w, a_, b_ = Y.shape
            U, s, Vh = _svd_split(Y.reshape(w * a_, b_), self.eps,
                                  self.chi_max)
            mps = [U.reshape(w, a_, -1),
                   (s[:, None] * Vh).reshape(-1, b_, 1)]
        else:
            # Zip-up von Links nach Rechts durch das Gedaechtnis-Fenster
            # left end (oldest slice, distance n_old)
            G = mps[0]                            # (w, a, x)
            F = self.Imats[n_old]
            Y = G[:, :, None, :] * F.T[None, :, :, None]     # (w, a, l, x)
            w, a_, l_, x_ = Y.shape
            U, s, Vh = _svd_split(Y.reshape(w * a_, l_ * x_),
                                  self.eps, self.chi_max, rank_hint=x_)
            new_mps = [U.reshape(w, a_, -1)]
            C = (s[:, None] * Vh).reshape(-1, l_, x_)        # (c, l, x)

            # middle sites (distances n_old-1 .. 2)
            # Hier greift das Verschieben des "l" Index durch das gesamte Netz
            for pos in range(1, n_old - 1):
                F = self.Imats[n_old - pos]
                G = mps[pos]                      # (x, a, y)
                Y = np.tensordot(C, G, axes=([2], [0]))      # (c, l, a, y)
                Y *= F[None, :, :, None]                     # F[l, a]
                c_, l_, a_, y_ = Y.shape
                M = Y.transpose(0, 2, 1, 3).reshape(c_ * a_, l_ * y_)
                # Sofortiges Beschneiden (Truncation) zurueck auf effektives chi
                U, s, Vh = _svd_split(M, self.eps, self.chi_max,
                                      rank_hint=y_)
                new_mps.append(U.reshape(c_, a_, -1))
                C = (s[:, None] * Vh).reshape(-1, l_, y_)

            # last old site (distance 1) + new slice b  (l == b)
            # Abschluss am rechten Rand, wo die Indikator-Logik des Schieberegisters greift
            G = mps[-1][:, :, 0]                  # (x, a)
            Y = np.tensordot(C, G, axes=([2], [0]))          # (c, b, a)
            Y *= W[None, :, :]                               # W[b, a]
            c_, b_, a_ = Y.shape
            M = Y.transpose(0, 2, 1).reshape(c_ * a_, b_)
            U, s, Vh = _svd_split(M, self.eps, self.chi_max)
            new_mps.append(U.reshape(c_, a_, -1))
            new_mps.append((s[:, None] * Vh).reshape(-1, b_, 1))
            mps = new_mps

        # restore right-canonical gauge (centre back to site 0), truncating
        # at proper cuts (left part is left-canonical from the pass above)
        # SVD Sweep rueckwaerts, um Fehler gloabal minimal zu halten und
        # das System fuer den naechsten Schritt zu praeparieren.
        for i in range(len(mps) - 1, 0, -1):
            l, p, r = mps[i].shape
            U, s, Vh = _svd_split(mps[i].reshape(l, p * r),
                                  self.eps, self.chi_max)
            mps[i] = Vh.reshape(-1, p, r)
            mps[i - 1] = np.tensordot(mps[i - 1], U * s[None, :],
                                      axes=([2], [0]))

        # memory truncation: marginalize slices older than kmax
        # Hier implementieren wir den "Cutoff". Das Gedaechtnis reicht nur
        # kmax Schritte zurueck. Der aelteste Zustand (alpha_{k-K}) wird
        # ueber die Funktion .sum() physikalisch "vergessen" (auskontrahiert).
        if len(mps) > self.kmax:
            v = mps[0].sum(axis=1)                # (alpha_0 bond, y)
            mps[1] = np.tensordot(v, mps[1], axes=([1], [0]))
            mps.pop(0)
        return mps

    # -- driver --------------------------------------------------------------
    def run(self, n_steps, verbose=True, callback=None):
        """Returns [L(0), L(dt), ..., L(n_steps*dt)].
        callback(k, maps) is invoked after every step (checkpointing).

        Hauptschleife, die das System iterativ vorwaerts in der Zeit propagiert.
        """
        D = self.D
        maps = [np.eye(D, dtype=complex)]
        # first slice: A_1[alpha_0, a_1] = I0[a_1] * Ph[a_1, alpha_0]
        # Start-Halbschritt wird in den allerersten Tensor absorbiert
        T = (self.I0[:, None] * self.Ph).T
        mps = [T[:, :, None].copy()]
        maps.append(self._readout(mps))
        for k in range(2, n_steps + 1):
            mps = self._step(mps)
            maps.append(self._readout(mps))
            if verbose and (k % 10 == 0 or k == n_steps):
                chi = max(t.shape[2] for t in mps)
                print(f"  step {k:4d}/{n_steps}   max bond = {chi}", flush=True)
            if callback is not None:
                callback(k, maps)
        return maps


# --------------------------------------------------------------------------
# Brute-force path sum (same discretization) for validation on few steps
# --------------------------------------------------------------------------

def brute_force_maps(H_cm, dt_fs, n_steps, kmax, lam_cm, gamma_cm, T_K,
                     n_mats=200000):
    d = H_cm.shape[0]
    D = d * d
    dtau = dt_fs * FS_TO_CM
    beta = 1.0 / (KB_CM * T_K)
    a, nu = drude_expansion(lam_cm, gamma_cm, beta, n_mats)
    eta = eta_coefficients(a, nu, dtau, kmax)
    Imats = influence_matrices(eta, d)
    I0 = np.diag(Imats[0]).copy()
    Uh = expm(-1j * H_cm * dtau / 2)
    Ph = np.kron(Uh.conj(), Uh)
    P = Ph @ Ph

    # R has axes [alpha_0, a_1, ..., a_k]
    R = (I0[:, None] * Ph).T                     # (alpha0, a1)
    maps = [np.eye(D, dtype=complex)]
    maps.append(Ph @ R.T)
    for k in range(1, n_steps):
        R = R[..., None] * P.T[..., :, :].reshape((1,) * k + (D, D))
        R *= I0.reshape((1,) * (k + 1) + (D,))
        for delta in range(1, min(k, kmax) + 1):
            shape = [1] * (k + 2)
            shape[k + 1 - delta] = D
            shape[k + 1] = D
            IF = Imats[delta].T.reshape(shape)   # [a_old, ..., a_new]
            R = R * IF
        S = R.reshape(D, -1, D).sum(axis=1)      # sum over interior slices
        maps.append(Ph @ S.T)
    return maps


# --------------------------------------------------------------------------
# Driver: compute the TEMPO maps on a grid and save them in the standard
# map-file format used by grid.py.  WARNING: this is the expensive one
# (hours); the data file data/pathintegral_maps.npz is already provided
# (copied from the earlier production run), so you normally never run this.
# --------------------------------------------------------------------------

# 4-site FMO pathway Hamiltonian (Seneviratne eq. 30), cm^-1
H_FMO = np.array([[12375.0, -87.7,   5.5,  -5.9],
                  [-87.7, 12495.0,  30.8,   8.2],
                  [5.5,      30.8, 12175.0, -53.4],
                  [-5.9,      8.2, -53.4, 12285.0]])
H_FMO = H_FMO - np.trace(H_FMO) / 4 * np.eye(4)
LAM_FMO, GAM_FMO, T_FMO = 35.0, 106.18, 300.0


def compute_and_save(outpath="data/pathintegral_maps.npz",
                     dt_fs=10.0, t_total=1000.0, t_mem_fs=250.0,
                     eps=1e-6, chi_max=512):
    """(Re)compute the FULL-trajectory (0..t_total) path-integral maps and
    save. Expensive (hours)."""
    from grid import save_maps
    n_steps = int(round(t_total / dt_fs))
    kmax = int(round(t_mem_fs / dt_fs))
    pm = PathIntegralMap(H_FMO, dt_fs, kmax, LAM_FMO, GAM_FMO, T_FMO,
                         eps=eps, chi_max=chi_max)
    maps = pm.run(n_steps, verbose=True)
    t_fs = np.arange(n_steps + 1) * dt_fs
    save_maps(outpath, t_fs, np.array(maps), H_FMO, "path integral (TEMPO)")
    print("saved", outpath)


if __name__ == "__main__":
    compute_and_save()
