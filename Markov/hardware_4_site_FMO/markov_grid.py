"""
Markovsche (Lindblad-) Dynamik auf EINEM durchgaengigen Quantengitter
====================================================================

Das ist der klassische Teil; er kennt keinen Quantenrechner.  Aus den
Modellparametern entsteht genau EIN Gatter, das der Schaltkreis dann n-mal
wiederholt -- unabhaengig davon, ob 10, 100 oder 1000 Zeitschritte gerechnet
werden.  Der klassische Rechner propagiert den Zustand nie, nicht einmal fuer
einen einzigen Schritt.

Das Gegenstueck dazu ist
`non_markov/hardware_4_site_FMO_with_compression/heom_gauge.py`.  Die Pipeline
ist Zeile fuer Zeile dieselbe; ausgetauscht ist nur der Generator -- dort der
HEOM-Hierarchie-Generator auf System + ADOs (n = 720), hier der blanke
Lindblad-Liouvillian auf vec(rho_S) (n = d^2 = 16).  Damit laesst sich der
markovsche gegen den nicht-markovschen Lauf auf denselben Achsen vergleichen,
und die Unterschiede liegen wirklich an der Physik und nicht am Verfahren.

Warum ueberhaupt eine Eichung, wenn Lindblad doch kontraktiv ist
---------------------------------------------------------------
Bei HEOM ist der Grund dramatisch: ||P||_2 = 32.39 bei Spektralradius 1, also
toetet s^(2n) = 32^200 jede Nachselektion.  Beim Lindblad-Kanal ist die Lage
viel harmloser -- gemessen fuer das 4-Site-FMO mit lam = 35, gamma = 50,
dt = 20 fs:

    ohne Eichung        ||P||_2 = 1.00430
    Eichung delta = 2   ||P~||_2 = 1.00176   (Schranke exp(delta dt) = 1.00756)

Der transiente Zuwachs ist also klein, aber nicht null: vec(rho) ist in der
Hilbert-Schmidt-Norm gemessen, und ein CPTP-Kanal ist darin NICHT automatisch
eine Kontraktion (kontraktiv ist er in der Spurnorm).  1.0043^n waechst bei
n = 25 auf 1.11, und genau dieser Faktor ginge als s^t in die Skala ein.  Die
Eichung drueckt ihn auf 1.045 und kostet nichts, weil sie eine exakte
Aehnlichkeitstransformation ist -- die Trajektorie wird an keiner Stelle
genaehert.

delta > 0 wird gebraucht, weil A den stationaeren Zustand als exakten
Eigenwert 0 hat; fuer delta = 0 existiert kein positiv definites W.

Pipeline
--------
    lindblad_generator   A          (16 x 16, ein Kronecker-Produkt)
    contraction_gauge    W^(1/2)    (ein Lyapunov-Loeser, keine Dynamik)
    gauged_propagator    P~         (ein Matrixexponential)
    arnoldi              H_m, Q_m   (m Matrix-Vektor-Produkte)
    dilate               U          (eine Unitaere -- das einzige Gatter)

Alles zusammen laeuft in Sekundenbruchteilen und faellt EINMAL an.
"""

import numpy as np
from scipy.linalg import (expm, solve_continuous_lyapunov, eigh, sqrtm,
                          schur, get_lapack_funcs)

__all__ = ['FS_TO_CM', 'KB_CM', 'H_FMO_FULL', 'make_model', 'selbsttest',
           'calc_L_list', 'lindblad_superop', 'lindblad_generator',
           'lindblad_onestep_map',
           'lyapunov_schur', 'contraction_gauge', 'gauged_propagator',
           'arnoldi', 'safe_norm2', 'dilate',
           'build_markov_grid', 'regrid',
           'pops_from_y', 'reference_trajectory', 'krylov_fehler_je_site',
           'qutip_reference', 'qutip_reference_rho']

_C_CM = 2.99792458e10                    # Lichtgeschwindigkeit [cm/s]
FS_TO_CM = 1e-15 * 2 * np.pi * _C_CM     # tau[cm] je t[fs]
KB_CM = 0.6950348004                     # Boltzmann [cm^-1/K]

# Read et al., Biophys. J. 95, 847 (2008); Pfad 1 -> 2 -> 3 -> 4 des FMO.
# Bitgleich mit dem H im nicht-markovschen Hardwareordner.
H_FMO_FULL = np.array([[12375.0, -87.7,   5.5,  -5.9],
                       [-87.7, 12495.0,  30.8,   8.2],
                       [5.5,      30.8, 12175.0, -53.4],
                       [-5.9,      8.2, -53.4, 12285.0]])


def make_model(n_sites=4, *, lam=35.0, gamma=50.0, T_K=300.0, H=None):
    """Modell und Anfangszustand -- gleiche Signatur wie `hardware.make_model`
    im nicht-markovschen Ordner, damit beide Laeufe dieselben Zahlen sehen.

    lam = 35 cm^-1 und gamma = 50 cm^-1 sind genau die Werte des
    Hardware-Laufs in
    non_markov/hardware_4_site_FMO_with_compression/main_hardware.ipynb
    (dort steht LAM = 35.0, gamma bleibt auf der Voreinstellung 50.0).

    Ohne eigenes `H` wird der fuehrende n_sites x n_sites-Block des
    FMO-Hamiltonians genommen und spurlos gemacht; die Anregung startet auf
    Site 1.  Anders als bei HEOM haengt hier NICHTS ausser der Zahl der
    Ablesekreise von n_sites ab: der Liouvillian ist d^2 x d^2, und der
    Krylov-Raum hat ohnehin nur m Dimensionen.

    Returns (MODEL, rho0).
    """
    if H is None:
        if not 2 <= n_sites <= 4:
            raise ValueError("ohne eigenes H sind nur n_sites = 2, 3, 4 "
                             "moeglich (Bloecke des FMO-Hamiltonians)")
        H = H_FMO_FULL[:n_sites, :n_sites].copy()
    H = np.asarray(H, float)
    n_sites = H.shape[0]
    H = H - np.trace(H) / n_sites * np.eye(n_sites)
    model = dict(H=H, lam=lam, gamma=gamma, T_K=T_K,
                 KB_CM=KB_CM, FS_TO_CM=FS_TO_CM)
    rho0 = np.diag([1.0] + [0.0] * (n_sites - 1)).astype(complex)
    return model, rho0


# --------------------------------------------------------------------------
# 1.  der Lindblad-Generator
# --------------------------------------------------------------------------

def calc_L_list(H, lam, gamma, T_K, KB_CM):
    """GKSL-Sprungoperatoren, Konstruktion aus der Masterarbeit (Sec. 6,
    `calc_L_list` in Sources/Figures/figure 6.21/figure6.21.py).

    Je Site eine lokale Projektorkopplung |m><m| an ein Drude-Bad.  Gebaut wird
    in der Eigenbasis von H: fuer jede Bohr-Frequenz omega = E_M - E_N wird der
    Anteil von |m><m| gesammelt, der genau mit dieser Frequenz oszilliert
    (Sekularnaeherung); omega = 0 gibt die reine Dephasierung.  Die Rate
    2 Re tau(omega) traegt das detaillierte Gleichgewicht ueber den
    Bose-Faktor.

    Dasselbe Bad wie im HEOM-Lauf -- Drude-Lorentz mit denselben lam und gamma.
    Der Unterschied ist die Naeherung, nicht das Modell: hier Born-Markov +
    Sekular, dort die exakte Hierarchie.
    """
    kBT = KB_CM * T_K
    d = H.shape[0]

    def Re_tau(omega):
        if abs(omega) < 1e-12:
            return 2 * kBT * lam / gamma                       # cm^-1
        J = 2 * lam * omega * gamma / (omega ** 2 + gamma ** 2)
        n = 1.0 / (np.exp(omega / kBT) - 1.0)
        return J * (n + 1.0)

    eks, V = np.linalg.eigh(H)                                 # V[:,M] = |v_M>
    w_diff = [eks[M] - eks[N] for M in range(d) for N in range(d)
              if M != N] + [0.0]

    L_list = []
    for m in range(d):
        for w in w_diff:
            Lmw = np.zeros((d, d), dtype=complex)
            for M in range(d):
                for N in range(d):
                    if abs((eks[M] - eks[N]) - w) < 1e-9:
                        amp = np.conj(V[m, M]) * V[m, N]       # <v_M|m><m|v_N>
                        Lmw += amp * np.outer(V[:, M], np.conj(V[:, N]))
            # VORZEICHENKONVENTION -- hier lag ein Fehler, siehe Test
            # `pruefe_detailliertes_gleichgewicht` unten.
            #
            # Lmw ist aus Termen |v_M><v_N| mit E_M - E_N = w gebaut, bildet
            # also |v_N> auf |v_M> mit E_M = E_N + w ab: positives w HEBT die
            # Energie an.  Die Rate dafuer ist die ABSORPTION aus dem Bad,
            # J(w) n(w).  Re_tau(w) = J(w) (n(w)+1) ist dagegen die EMISSION
            # und gehoert zu energiesenkenden Operatoren.
            #
            # In der ueblichen Konvention (Breuer-Petruccione) senkt A(omega)
            # mit omega > 0 die Energie, und genau dort steht (n+1).  Dieses
            # Modul baut die Operatoren mit dem umgekehrten Vorzeichen, also
            # muss die Rate bei -w ausgewertet werden.  Wegen
            #     J(-w) (n(-w)+1) = J(w) n(w)
            # liefert Re_tau(-w) exakt die Absorptionsrate.
            #
            # Mit Re_tau(+w) relaxierte der Generator in die ANTI-Boltzmann-
            # Verteilung: das Verhaeltnis rauf/runter war exp(+w/kT) = 6.59
            # statt exp(-w/kT) = 0.15, und der hoechste Eigenzustand bekam
            # 54.5 % statt 7.4 % der Population.
            gam = 2 * Re_tau(-w)
            if gam > 0:
                L_list.append(np.sqrt(gam) * Lmw)
    return L_list


def selbsttest(model=None, rho0=None, dt_fs=20.0, n_steps=10, streng=True,
               verbose=True):
    """Zehn physikalische Pflichtpruefungen des Generators.  Wirft bei Verstoss.

    Hintergrund: in einer frueheren Fassung bekamen in `calc_L_list` die
    energieANHEBENDEN Operatoren die EMISSIONsrate J(w)(n(w)+1) statt der
    Absorptionsrate J(w)n(w).  Das Verhaeltnis rauf/runter war dadurch
    exp(+w/kT) statt exp(-w/kT), und der Generator relaxierte exakt in die
    ANTI-Boltzmann-Verteilung -- der hoechste Eigenzustand bekam 54.5 % statt
    7.4 %.  Der Fehler war an den Populationskurven nicht ohne Weiteres zu
    sehen, weil der Algorithmus den (falschen) Generator korrekt reproduzierte
    und auch die qutip-Referenz dieselben Sprungoperatoren benutzte.

    Pruefung 1 und 2 haetten ihn sofort gefunden.  Deshalb vor jedem
    Hardwarelauf aufrufen.
    """
    import numpy as np
    from scipy.linalg import expm
    if model is None:
        model, rho0 = make_model(4)
    if rho0 is None:
        rho0 = np.diag([1.0] + [0.0] * (model['H'].shape[0] - 1))
    H = model['H']; d = H.shape[0]
    kBT = model['KB_CM'] * model['T_K']
    eks, V = np.linalg.eigh(H)
    A, info = lindblad_generator(**model)
    P = expm(A * dt_fs * model['FS_TO_CM'])
    fehler = []

    def pruefe(name, bedingung, zusatz=''):
        if verbose:
            print(f"  [{'OK  ' if bedingung else 'FEHLER'}] {name}{zusatz}")
        if not bedingung:
            fehler.append(name)

    # 1 stationaerer Zustand == Boltzmann
    w_, vr = np.linalg.eig(A)
    rho = vr[:, np.argmin(np.abs(w_))].reshape(d, d, order='F')
    rho = rho / np.trace(rho)
    pop = np.real(np.diag(V.conj().T @ rho @ V))
    bz = np.exp(-(eks - eks.min()) / kBT); bz /= bz.sum()
    ab = np.abs(pop - bz).max()
    pruefe('stationaerer Zustand = Boltzmann', ab < 1e-8, f' (Abstand {ab:.2e})')

    # 2 detailliertes Gleichgewicht der Raten
    L_list = info['L_list']
    om = eks[-1] - eks[0]
    lam, gam_b = model['lam'], model['gamma']
    def J(o): return 2 * lam * o * gam_b / (o ** 2 + gam_b ** 2)
    def nb(o): return 1.0 / (np.exp(o / kBT) - 1.0)
    rauf, runter = J(om) * nb(om), J(om) * (nb(om) + 1.0)
    pruefe('detailliertes Gleichgewicht', abs(rauf / runter - np.exp(-om / kBT)) < 1e-9,
           f' (rauf/runter {rauf/runter:.5f} = exp(-w/kT) {np.exp(-om/kBT):.5f})')

    # 3 Spurerhaltung
    tv = np.eye(d).reshape(-1, order='F')
    m3 = np.abs(tv @ A).max()
    pruefe('Spurerhaltung', m3 < 1e-8, f' (max|1^T A| = {m3:.2e})')

    # 4 Hermitizitaet erhalten
    rng = np.random.default_rng(0)
    X = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d)); X = X + X.conj().T
    Y = (P @ X.reshape(-1, order='F')).reshape(d, d, order='F')
    m4 = np.abs(Y - Y.conj().T).max()
    pruefe('hermitesch erhalten', m4 < 1e-8, f' ({m4:.2e})')

    # 5 vollstaendige Positivitaet (Choi)
    Choi = np.zeros((d * d, d * d), complex)
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d)); E[i, j] = 1
            Choi[i*d:(i+1)*d, j*d:(j+1)*d] = \
                (P @ E.reshape(-1, order='F')).reshape(d, d, order='F')
    mn = np.linalg.eigvalsh((Choi + Choi.conj().T) / 2).min()
    pruefe('vollstaendig positiv', mn > -1e-8, f' (min Choi-Eigenwert {mn:.2e})')

    # 6 gegen qutip mesolve
    t = np.arange(0.0, n_steps * dt_fs + 0.5, dt_fs)
    ref = qutip_reference_rho(t, rho0=rho0, **model)
    y = np.asarray(rho0, complex).reshape(-1, order='F')
    e6 = 0.0
    for k in range(1, len(t)):
        y = P @ y
        e6 = max(e6, np.abs(y.reshape(d, d, order='F') - ref[k]).max())
    pruefe('exp(A dt)^n == qutip mesolve', e6 < 1e-6, f' (max {e6:.2e})')

    # 7 Sprungoperatoren wohlgeformt
    S = sum(Lk.conj().T @ Lk for Lk in L_list)
    pruefe('sum L^dag L hermitesch', np.abs(S - S.conj().T).max() < 1e-8)

    # 8 Dephasierungsgrenzwert
    lim = 2 * kBT * lam / gam_b
    nah = J(1e-6) * (nb(1e-6) + 1.0)
    pruefe('Dephasierungsrate stetig bei w->0', abs(lim - nah) / lim < 1e-4,
           f' ({lim:.4f} vs {nah:.4f})')

    # 9 keine negativen Raten
    neg = sum(1 for Lk in L_list if not np.isfinite(Lk).all())
    pruefe('alle Sprungoperatoren endlich', neg == 0, f' ({neg} fehlerhaft)')

    # 10 Anfangszustand physikalisch
    r0a = np.asarray(rho0, complex)
    pruefe('rho0 Spur 1 und positiv',
           abs(np.trace(r0a) - 1) < 1e-12
           and np.linalg.eigvalsh((r0a + r0a.conj().T) / 2).min() > -1e-12)

    if fehler:
        msg = f"Selbsttest fehlgeschlagen: {fehler}"
        if streng:
            raise AssertionError(msg)
        print('  ' + msg)
    elif verbose:
        print(f"  alle 10 Pruefungen bestanden")
    return not fehler


def lindblad_superop(H, L_list):
    """Liouvillian L (D x D, D = d^2) in SPALTENSTAPELUNG:
    vec(A rho B) = (B^T (x) A) vec(rho).

    Die Konvention muss zu `R` und zu `reshape(order='F')` in `pops_from_y`
    passen -- sie ist dieselbe wie im nicht-markovschen Ordner.
    """
    d = H.shape[0]
    I = np.eye(d)
    Lsup = -1j * (np.kron(I, H) - np.kron(H.T, I))            # -i[H, .]
    for Lk in L_list:
        LdL = Lk.conj().T @ Lk
        Lsup += (np.kron(Lk.conj(), Lk)
                 - 0.5 * np.kron(I, LdL)
                 - 0.5 * np.kron(LdL.T, I))
    return Lsup


def lindblad_generator(*, H, lam, gamma, T_K, KB_CM, FS_TO_CM):
    """Der Generator A auf vec(rho_S), also A = L_Lindblad.

    Gegenstueck zu `heom_gauge.heom_generator`.  FS_TO_CM wird nur
    entgegengenommen, damit sich dasselbe MODEL-Dictionary ueberall
    hineinsplatten laesst.  Returns (A, info).
    """
    d = H.shape[0]
    L_list = calc_L_list(H, lam, gamma, T_K, KB_CM)
    A = lindblad_superop(H, L_list)
    return A, dict(d=d, n=A.shape[0], n_jump=len(L_list), L_list=L_list)


def lindblad_onestep_map(dt_fs, *, H, lam, gamma, T_K, KB_CM, FS_TO_CM):
    """L(dt) = exp(L dt) -- die Einschrittabbildung, EIN Matrixexponential.

    Wird vom Gitter selbst nicht gebraucht (dort steht die geeichte Variante
    `gauged_propagator`), ist aber die kompakteste klassische Referenz: weil
    GKSL eine Halbgruppe ist, gilt L(n dt) = L(dt)^n exakt.
    """
    A, _ = lindblad_generator(H=H, lam=lam, gamma=gamma, T_K=T_K,
                              KB_CM=KB_CM, FS_TO_CM=FS_TO_CM)
    return expm(A * (dt_fs * FS_TO_CM))


# --------------------------------------------------------------------------
# 2.  die Kontraktionseichung
# --------------------------------------------------------------------------

def lyapunov_schur(A):
    """Schur-Zerlegung von A^dagger, wiederverwendbar fuer beliebige delta.

    Bartels-Stewart loest (LYAP), indem es a = A^dag - delta I auf Schur-Form
    bringt und dann eine Dreiecksgleichung loest.  Der Shift delta I aendert
    die Schur-VEKTOREN nicht, nur die Diagonale von T.  Fuer einen Sweep ueber
    viele delta rechnet man die Zerlegung deshalb genau einmal.

    Returns (T, Z) mit A^dagger = Z T Z^dagger.
    """
    T, Z = schur(A.conj().T, output='complex')
    return T, Z


def contraction_gauge(A, *, delta, Q=None, schur_factor=None):
    """Loest (LYAP) und gibt die Eichmatrizen (Wh, Wih) = (W^1/2, W^-1/2).

    delta > 0 ist ein spektraler Shift in cm^-1 und der EINZIGE Knopf: er
    beschraenkt ||P~||_2 <= exp(delta*dt) nach oben und cond(W) ~ 1/delta nach
    unten, handelt also den Nachselektionsverlust ueber den ganzen Lauf gegen
    die Kondition des klassischen Basiswechsels.

    Beim 16x16-Liouvillian ist beides unkritisch -- cond(W) = 2.7e2 bei
    delta = 2 gegen 1e16, wo `eigh` zusammenbricht.  Die ADO-Umskalierung, die
    im HEOM-Fall noetig ist, entfaellt hier ersatzlos: es gibt keine ADOs, und
    alle 16 Koordinaten sind Eintraege derselben Dichtematrix, also von
    vergleichbarer Groesse.
    """
    n = A.shape[0]
    if Q is None:
        Q = np.eye(n, dtype=complex)
    if schur_factor is None:
        Ad = A - delta * np.eye(n)
        # scipy loest  a x + x a^dag = q; mit a = Ad^dag ist das Ad^dag W + W Ad = -Q
        W = solve_continuous_lyapunov(Ad.conj().T, -Q)
    else:
        T, Z = schur_factor
        Td = T - delta * np.eye(n)
        Qp = Z.conj().T @ Q @ Z
        trsyl, = get_lapack_funcs(('trsyl',), (Td, Qp))
        Y, scl, inf = trsyl(Td, Td, -Qp, trana='N', tranb='C', isgn=1)
        if inf < 0:
            raise np.linalg.LinAlgError(f"trsyl: ungueltiges Argument {-inf}")
        W = Z @ (Y / scl) @ Z.conj().T
    W = 0.5 * (W + W.conj().T)
    w, V = eigh(W)
    if w.min() <= 0:
        raise np.linalg.LinAlgError(
            f"Lyapunov-Loesung nicht positiv definit (min eig {w.min():.3e}); "
            "delta erhoehen.  Fuer einen echten Lindblad-Generator kann das "
            "nur an delta = 0 liegen -- A ist power-bounded, hat aber den "
            "stationaeren Zustand als exakten Eigenwert 0.")
    Wh = (V * np.sqrt(w)) @ V.conj().T
    Wih = (V / np.sqrt(w)) @ V.conj().T
    info = dict(delta=delta, eig_min=float(w.min()), eig_max=float(w.max()),
                cond=float(w.max() / w.min()))
    return Wh, Wih, info


def gauged_propagator(A, dt_cm, Wh, Wih):
    """P~ = W^(1/2) exp(A dt) W^(-1/2).  Exakt aehnlich zu P, aber kontraktiv."""
    P = expm(A * dt_cm)
    return Wh @ P @ Wih, P


# --------------------------------------------------------------------------
# 3.  Krylov-Kompression und Sz.-Nagy-Dilatation
# --------------------------------------------------------------------------

def arnoldi(P, x0, m):
    """Orthonormale Krylov-Basis Q_m von span{x0, P x0, ...} und
    H_m = Q_m^dag P Q_m.  Zweimal reorthogonalisiert.

    Der Krylov-Raum K_m(P~, x~0) enthaelt die ersten m Zeitschritte EXAKT;
    darueber hinaus extrapoliert er.  Weil Q_m eine Isometrie ist, erbt H_m
    die Kontraktivitaet von P~ -- die Kompression kann also nicht divergieren.
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


def safe_norm2(A):
    """Spektralnorm, oder nan, wenn die SVD nicht konvergiert."""
    try:
        return float(np.linalg.norm(A, 2))
    except np.linalg.LinAlgError:
        return float('nan')


def dilate(M, s=None):
    """Sz.-Nagy: M/s als linker oberer Block einer Unitaeren, ein Qubit groesser.

        U = [[ M/s ,  sqrt(I - M M^dag/s^2) ],
             [ sqrt(I - M^dag M/s^2) , -(M/s)^dag ]]

    Die Unitaritaet folgt aus der Verflechtungsrelation
    M^dag f(M M^dag) = f(M^dag M) M^dag.  In der Eichung ist s ~ 1.
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
# 4.  die ganze einmalige Konstruktion
# --------------------------------------------------------------------------

def build_markov_grid(dt_fs=20.0, *, rho0=None, delta=2.0, m=4, model=None,
                      n_sites=4, verbose=True):
    """Alles, was das Gitter braucht, allein aus den Modellparametern.

    m = 4 -> 3 Qubits (2 Krylov + 1 Ancilla), m = 8 -> 4 Qubits.  Weil
    D = d^2 = 16 ist, ist m = 16 bereits die volle, unkomprimierte Rechnung --
    im Gegensatz zum HEOM-Fall, wo n = 720 waere.

    Returns ein Dictionary mit
        U       die eine Unitaere, die der Schaltkreis wiederholt (2mp x 2mp)
        s       ihre Sz.-Nagy-Skala, ~ exp(delta dt)
        m, mp   Krylov-Dimension und ihre Polsterung auf eine Zweierpotenz
        y0      die Anfangskoordinaten (= ||x~0|| e_1)
        R       Ablesematrix, vec(rho_S)(t) = R @ y(t)          (d^2 x m)
        Hm, Qm, Wh, Wih, A, P, Pt   die klassischen Objekte zur Diagnose
        model, rho0, d, D, dt_fs, dt_cm, n_sites
    """
    if model is None:
        model, rho0_default = make_model(n_sites)
        if rho0 is None:
            rho0 = rho0_default
    elif rho0 is None:
        rho0 = np.diag([1.0] + [0.0] * (model['H'].shape[0] - 1)).astype(complex)

    d = model['H'].shape[0]
    D = d * d
    dt_cm = dt_fs * model['FS_TO_CM']
    if m > D:
        raise ValueError(f"m = {m} ist groesser als der ganze Raum D = {D}; "
                         f"m = {D} ist bereits die unkomprimierte Rechnung.")

    A, info = lindblad_generator(**model)
    Wh, Wih, gauge_info = contraction_gauge(A, delta=delta)
    Pt, P = gauged_propagator(A, dt_cm, Wh, Wih)
    norm_Pt = safe_norm2(Pt)

    # Kostenlose Gueltigkeitspruefung.  Die Eichung GARANTIERT
    # ||P~||_2 <= exp(delta dt); wird das ueberschritten, haben W^(1/2) und
    # W^(-1/2) Stellen verloren.  Beim 16x16-Liouvillian passiert das nicht,
    # die Pruefung bleibt trotzdem drin -- sie kostet nichts und faengt ein
    # falsch uebergebenes Modell ab.
    bound = float(np.exp(delta * dt_cm))
    gauge_info.update(bound=bound, norm_P=safe_norm2(P), norm_Pt=norm_Pt,
                      trustworthy=bool(norm_Pt <= bound * (1 + 1e-6)))
    if not gauge_info['trustworthy']:
        import warnings
        warnings.warn(
            f"Eichung hat Genauigkeit verloren: ||P~||_2 = {norm_Pt:.4g} "
            f"ueberschreitet exp(delta dt) = {bound:.6g}.  "
            f"cond(W) = {gauge_info['cond']:.2e}.  delta erhoehen oder dt "
            f"verkleinern.", RuntimeWarning, stacklevel=2)

    x0 = np.zeros(D, complex)
    x0[:] = np.asarray(rho0, complex).flatten(order='F')
    xt0 = Wh @ x0

    Hm, Qm = arnoldi(Pt, xt0, m)
    m = Hm.shape[0]
    mp = 2 ** int(np.ceil(np.log2(m)))
    Apad = np.zeros((mp, mp), complex)
    Apad[:m, :m] = Hm
    U, s = dilate(Apad)

    y0 = np.zeros(m, complex)
    y0[0] = np.linalg.norm(xt0)
    R = (Wih @ Qm)[:D, :]

    grid = dict(U=U, s=s, m=m, mp=mp, n_qubits=int(np.log2(mp)) + 1,
                y0=y0, R=R, Hm=Hm, Qm=Qm, Wh=Wh, Wih=Wih, A=A, P=P, Pt=Pt,
                d=d, D=D, dt_fs=dt_fs, dt_cm=dt_cm, gauge=gauge_info,
                lindblad=info, model=model, rho0=rho0, n_sites=d, delta=delta)

    if verbose:
        print(f"  {d} Sites, lam = {model['lam']:.1f}, "
              f"gamma = {model['gamma']:.1f} cm^-1, T = {model['T_K']:.0f} K, "
              f"dt = {dt_fs:.0f} fs")
        print(f"  A: {info['n']} x {info['n']} "
              f"({info['n_jump']} Sprungoperatoren)")
        print(f"  Eichung delta = {delta} cm^-1: cond(W) = "
              f"{gauge_info['cond']:.2e}, ||P|| = {gauge_info['norm_P']:.5f} "
              f"-> ||P~|| = {norm_Pt:.6f} (Schranke {bound:.6f})")
        print(f"  Krylov m = {m} -> gepolstert {mp}, "
              f"{grid['n_qubits']} Qubits, ||H_m|| = "
              f"{np.linalg.norm(Hm, 2):.6f}, s = {s:.6f}, "
              f"s^25 = {s ** 25:.4f}")
    return grid


def regrid(grid, m, *, rho0=None, verbose=False):
    """Dieselbe Eichung bei anderer Krylov-Dimension neu komprimieren.

    Der teure Teil von `build_markov_grid` -- Lyapunov-Loeser und
    Matrixexponential -- haengt weder von m noch vom Anfangszustand ab.  Ein
    Sweep ueber m kostet deshalb nur m Matrix-Vektor-Produkte je Punkt.
    """
    if rho0 is None:
        xt0 = grid['Qm'][:, 0] * np.linalg.norm(grid['y0'])
    else:
        x0 = np.asarray(rho0, complex).flatten(order='F')
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
    if rho0 is not None:
        out['rho0'] = rho0
    if verbose:
        print(f"  m = {m} -> {out['n_qubits']} Qubits, "
              f"||H_m|| = {np.linalg.norm(Hm, 2):.6f}", flush=True)
    return out


# --------------------------------------------------------------------------
# 5.  Ablesen und klassische Referenzen
# --------------------------------------------------------------------------

def pops_from_y(y, R, d):
    """Populationen aus Krylov-Koordinaten: vec(rho) = R y, dann die Diagonale."""
    return np.real(np.diag((R @ y).reshape(d, d, order='F')))


def rho_from_y(y, R, d):
    """Die volle Dichtematrix aus Krylov-Koordinaten."""
    return (R @ y).reshape(d, d, order='F')


def reference_trajectory(grid, n_steps, exact=False):
    """Klassischer Massstab.  exact=True iteriert den vollen geeichten
    Propagator P~ (keine Krylov-Abschneidung), sonst H_m -- und H_m ist genau
    das, was das Gitter tut.  Die Differenz der beiden IST der algorithmische
    Fehler der Kompression.  Returns (n_steps+1, d, d) komplex.
    """
    d, D = grid['d'], grid['D']
    if exact:
        x = grid['Qm'] @ grid['y0']          # == x~0, da Qm[:,0] = x~0/||x~0||
        Rf = grid['Wih'][:D, :]
        out = [(Rf @ x).reshape(d, d, order='F')]
        for _ in range(n_steps):
            x = grid['Pt'] @ x
            out.append((Rf @ x).reshape(d, d, order='F'))
        return np.array(out)
    y = grid['y0'].copy()
    out = [rho_from_y(y, grid['R'], d)]
    for _ in range(n_steps):
        y = grid['Hm'] @ y                   # OHNE s -- H_m ist der unskalierte
        out.append(rho_from_y(y, grid['R'], d))   # Propagator; s^t hebt sich in
    return np.array(out)                          # der Auswertung wieder auf


def krylov_fehler_je_site(grid, n_steps, *, spurnormiert=True):
    """Der Krylov-Abschneidefehler JE SITE und Zeitschritt, vorzeichenbehaftet.

    `reference_trajectory` liefert die Trajektorien, die Zelle 3b als Maximum
    ueber alle Sites zusammenfasst.  Diese Funktion behaelt die Aufloesung nach
    Site UND das Vorzeichen -- und erst daran sieht man die eigentliche
    Struktur: die Abschneidung verschiebt Population gerichtet zwischen den
    Sites, statt sie gleichmaessig zu verrauschen.

    Gemessen bei m = 4, dt = 20 fs, t = 10 (spurnormiert):

        Site 1   -0.0056      Site 3   -0.0469
        Site 2   -0.0037      Site 4   +0.0562

    Site 4 wird also zu hoch, Site 3 zu tief, waehrend 1 und 2 kaum betroffen
    sind.  Grund: K_m enthaelt P^t y0 exakt nur fuer t <= m-1, und gerade der
    langsame Aufbau der schwach besetzten Zustaende 3 und 4 lebt von den
    Richtungen, die der Unterraum nicht mehr darstellt.

    `spurnormiert=True` vergleicht so, wie das Notebook plottet -- die
    Spurnormierung ist die Ablesung, die p_t nicht braucht.

    Returns ein dict mit
        t_fs         (n_steps+1,)        Zeiten
        pop_krylov   (n_steps+1, d)      was das Gitter liefert
        pop_exakt    (n_steps+1, d)      geeichter Propagator ohne Kompression
        pop_qutip    (n_steps+1, d)      qutip mesolve
        fehler       (n_steps+1, d)      pop_krylov - pop_qutip
        fehler_exakt (n_steps+1, d)      pop_exakt  - pop_qutip  (~1e-9)
    """
    d = grid['d']
    t_fs = np.arange(n_steps + 1) * grid['dt_fs']
    kry = reference_trajectory(grid, n_steps)
    exa = reference_trajectory(grid, n_steps, exact=True)
    ref = qutip_reference_rho(t_fs, rho0=grid['rho0'], **grid['model'])

    def diag(arr):
        p = np.real(np.einsum('tii->ti', arr))
        if spurnormiert:
            sp = p.sum(axis=1, keepdims=True)
            p = np.divide(p, sp, out=np.zeros_like(p), where=np.abs(sp) > 1e-30)
        return p

    pk, pe, pq = diag(kry), diag(exa), diag(ref)
    return dict(t_fs=t_fs, pop_krylov=pk, pop_exakt=pe, pop_qutip=pq,
                fehler=pk - pq, fehler_exakt=pe - pq)


def post_selection_curve(grid, n_steps):
    """p_t = ||H_m^t y0||^2 / (s^(2t) ||y0||^2) -- die ideale
    Nachselektionswahrscheinlichkeit, ohne jedes Geraeterauschen.  Sie ist der
    Massstab, an dem sich das gemessene p_t einordnen laesst."""
    y = grid['y0'].copy()
    n0 = np.linalg.norm(y)
    M = grid['Hm'] / grid['s']
    out = [1.0]
    for _ in range(n_steps):
        y = M @ y
        out.append(float((np.linalg.norm(y) / n0) ** 2))
    return np.array(out)


def qutip_reference_rho(t_fs, *, rho0, H, lam, gamma, T_K, KB_CM, FS_TO_CM):
    """BENCHMARK: die gewoehnliche Rechnung mit qutips `mesolve` -- kein
    Gitter, keine Kompression, keine Dilatation.  Dieselben Sprungoperatoren.

    ACHTUNG, Grenze dieses Vergleichs: die Sprungoperatoren kommen aus unserem
    eigenen `calc_L_list`, qutip steuert nur den ODE-Integrator bei.  Geprueft
    wird damit, ob exp(A dt)^n dieselbe DGL loest wie der Integrator -- NICHT,
    ob A der richtige Generator ist.  Genau deshalb blieb der Vorzeichenfehler
    in `calc_L_list` hier unsichtbar (Abweichung 3.6e-8, also unauffaellig).
    Fuer die Physik ist `selbsttest` zustaendig, insbesondere Pruefung 1
    (stationaerer Zustand = Boltzmann) und 2 (detailliertes Gleichgewicht).
    Returns (len(t_fs), d, d) komplex, damit auch die Kohaerenzen eine
    Referenz haben."""
    import qutip as qt
    L_list = calc_L_list(H, lam, gamma, T_K, KB_CM)
    res = qt.mesolve(qt.Qobj(H), qt.Qobj(np.asarray(rho0, complex)),
                     np.asarray(t_fs, float) * FS_TO_CM,
                     c_ops=[qt.Qobj(Lk) for Lk in L_list])
    return np.array([np.asarray(s.full()) for s in res.states])


def qutip_reference(t_fs, **kw):
    """Nur die Populationen aus `qutip_reference_rho`."""
    return np.real(np.einsum('tii->ti', qutip_reference_rho(t_fs, **kw)))
