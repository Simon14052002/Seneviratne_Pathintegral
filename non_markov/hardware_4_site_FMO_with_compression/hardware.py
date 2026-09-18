"""
4-Site-FMO auf echter IBM-Hardware -- durchgaengiges Gitter, 3 Qubits.

Warum das geht, obwohl kritik.ipynb das Gegenteil nahelegt
---------------------------------------------------------
Die unkomprimierte Variante lebt im vollen HEOM-Raum (n = 720 -> 11 Qubits) und
braucht (23/48) 4^11 ~ 2e6 CNOTs je Zeitschritt.  Das ist auf keiner heutigen
Maschine lauffaehig.  Die KOMPRIMIERTE Variante nicht: der Krylov-Raum
K_m(P~, x~0) enthaelt genau die ersten m Zeitschritte, und ein Hardware-Lauf ist
durch die Kohaerenzzeit ohnehin auf ~5-8 Schritte begrenzt.  Gemessen gegen
qutip (4-Site-FMO, Tiefe 2, ADO-skaliert, delta = 0.02):

    m    Qubits  CNOTs/Schritt   Fehler t=1   t=3       t=6
    4    3       31              1.1e-10      1.9e-10   2.0e-03
    8    4       123             1.1e-10      1.9e-10   1.9e-09
    128  8       31400           --           --        --

m = 4 gibt 3 Qubits -- dieselbe Zahl, die Seneviratne et al. (ACS Omega 2024,
9, 9666) fuer ihre Kraus-Operatoren brauchen.  Der Unterschied: dort wird die
Abbildung fuer JEDEN Zeitpunkt klassisch neu gerechnet (TNPI -> Choi -> Kraus ->
eigener Schaltkreis), hier laeuft EIN durchgaengiger Schaltkreis, und der
klassische Aufwand faellt einmal an.

Was auf Hardware anders ist als in Aer
--------------------------------------
    - kein `save_statevector`; alles kommt aus Zaehlraten
    - jeder Schaltkreis muss in die ISA der Maschine transpiliert werden
    - je Auslesezeit und je Ablesezeile ein eigener Schaltkreis, weil die
      Messung den Zustand zerstoert
    - Mid-Circuit-Messung mit Reset je Zeitschritt (dynamischer Schaltkreis)

Ablauf
------
    grid  = build_hardware_grid(m=4)
    circs = hardware_circuits(grid, times=[1,2,3,4,5,6])
    report_cost(grid, circs)          # VOR dem Absenden: Gatter, Treue, QPU-Zeit
    verify_in_aer(grid, circs)        # kostenlos gegenpruefen
    res   = run_on_backend(circs, backend_name='ibm_pittsburgh', shots=4096)
    rho, err, info = analyze(grid, circs, res)
"""

import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate, DiagonalGate

# heom_gauge.py / gauge_circuit.py liegen im Schwesterordner -- ABER eine
# lokale Kopie hat Vorrang.  Ohne diese Abfrage schob die Schleife unten den
# Schwesterordner an sys.path[0] und verdeckte damit die Datei, die direkt
# neben diesem Modul liegt.  Das war lange folgenlos, weil sich die beiden
# Kopien nur in der Voreinstellung von `scale_ados` unterscheiden und
# `build_hardware_grid` sie ohnehin explizit auf True setzt -- aber jede
# Aenderung an der lokalen Datei waere wirkungslos geblieben.
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, 'heom_gauge.py')):
    if _HERE in sys.path:
        sys.path.remove(_HERE)
    sys.path.insert(0, _HERE)          # lokale Kopie gewinnt, Suche entfaellt
else:
    for _up in range(1, 4):
        _root = os.path.abspath(os.path.join(_HERE, *(['..'] * _up)))
        for _c in (_root, os.path.join(_root, 'with_compression')):
            if os.path.exists(os.path.join(_c, 'heom_gauge.py')):
                if _c not in sys.path:
                    sys.path.insert(0, _c)
                break
        else:
            continue
        break

from heom_gauge import (build_grid, regrid, qutip_reference, qutip_reference_rho,
                        krylov_fehler_je_site)

__all__ = ['MODEL', 'RHO0', 'H_FMO_FULL', 'make_model', 'isa_report',
           'build_hardware_grid', 'hardware_circuits', 'report_cost',
           'verify_in_aer', 'run_on_backend', 'analyze',
           'qutip_reference', 'qutip_reference_rho', 'krylov_fehler_je_site']

_C_CM = 2.99792458e10
FS_TO_CM = 1e-15 * 2 * np.pi * _C_CM
KB_CM = 0.6950348004

# Read et al., Biophys. J. 95, 847 (2008); Pfad 1 -> 2 -> 3 -> 4 des FMO.
H_FMO_FULL = np.array([[12375.0, -87.7,   5.5,  -5.9],
                       [-87.7, 12495.0,  30.8,   8.2],
                       [5.5,      30.8, 12175.0, -53.4],
                       [-5.9,      8.2, -53.4, 12285.0]])


def make_model(n_sites=4, *, lam=3.0, gamma=50.0, T_K=300.0, H=None):
    """Modell und Anfangszustand fuer n_sites Sites.

    Ohne eigenes `H` wird der fuehrende n_sites x n_sites-Block des
    FMO-Hamiltonians genommen und spurlos gemacht.  Die Anregung startet
    auf Site 1.

    Die Krylov-Dimension haengt NICHT von n_sites ab, sondern von der Zahl der
    Zeitschritte.  Gemessen bei m = 4 (3 Qubits, 31 CZ je Schritt), Tiefe 2:

        n_sites   ADOs    n     Fehler t=1   t=3        t=6
        2         15      60    1.2e-11      5.8e-09    2.7e-03
        3         28      252   4.5e-11      3.1e-10    3.5e-03
        4         45      720   1.1e-10      1.9e-10    2.0e-03

    Der volle 4-Site-Fall kostet auf der Maschine also genauso viel wie der
    Dimer -- die Kompression frisst den Unterschied auf.

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
    rho0 = np.diag([1.0] + [0.0] * (n_sites - 1))
    return model, rho0


N_SITES = 4
MODEL, RHO0 = make_model(N_SITES)       # Voreinstellung: volles 4-Site-FMO

# Heron-Basisgatter; fuer die Offline-Schaetzung ohne Backend
HERON_BASIS = ['cz', 'rz', 'sx', 'x', 'measure', 'reset']


# --------------------------------------------------------------------------
# 1.  das kleine Gitter
# --------------------------------------------------------------------------

def build_hardware_grid(*, n_sites=4, m=4, dt_fs=10.0, depth=2, Nk=1,
                        delta=10.0, m_build=64, model=None, rho0=None,
                        verbose=True):
    """Das durchgaengige Gitter, auf m Krylov-Dimensionen komprimiert.

    m = 4 -> 3 Qubits (2 System + 1 Ancilla), m = 8 -> 4 Qubits.  Groesser
    lohnt fuer Hardware nicht: der algorithmische Fehler ist bei m = 4 schon
    2e-3 ueber 6 Schritte und damit weit unter dem Geraeterauschen.

    delta = 10.0 (nicht 0.02, und auch nicht 2.0).  Im komprimierten Krylov-Raum sind die
    Ablesezeilen dicht und lang, und die Verstaerkung A = ||r|| lambda_t
    bestimmt den Ablesefehler A/(2 sqrt(N)).  Gemessen bei m = 4:

    Massgeblich ist der GESAMTfehler aus Statistik und Algorithmus, gemessen
    bei m = 4, lam = 3, t = 6, 4096 Shots:

        delta   Shots f. 0.03   Algorithmus   Statistik   GESAMT    CZ
        0.02             23633     1.97e-03      0.0716   0.0716    63
        0.5               1040     1.97e-03      0.0151   0.0152    63
        2                  439     1.98e-03      0.0098   0.0100    63
        10                 292     2.68e-03      0.0080   0.0084    63   <- Optimum
        30                 279     4.70e-03      0.0080   0.0093    63

    Die Gatterzahl ist von delta UNABHAENGIG (63 CZ ueberall) -- delta kostet
    also nichts in der Tiefe, nur in Shots und Genauigkeit.  Ab delta ~ 10
    saettigt der Shot-Gewinn (292 -> 279), waehrend der Algorithmusfehler
    weiterwaechst.

    Bei delta = 0.02 bekommt der RR-Kanal unter EINEN Treffer je 4096 Shots;
    dann kollabiert Im rho_ab auf -(rho_aa+rho_bb)/2.  Fruehere Fassungen
    empfahlen delta = 2 -- das war zu vorsichtig: der Algorithmusfehler bei
    delta = 10 liegt mit 2.7e-03 weit unter der statistischen Schwelle von
    ~0.010 und faellt gar nicht ins Gewicht.
    """
    t0 = time.time()
    if model is None:
        model, rho0 = make_model(n_sites)
    elif rho0 is None:
        rho0 = np.diag([1.0] + [0.0] * (model['H'].shape[0] - 1))
    n_sites = model['H'].shape[0]
    m_build = max(m, min(m_build, 4 * len(rho0) ** 2))

    g = build_grid(dt_fs, rho0=rho0, delta=delta, m=m_build, depth=depth,
                   Nk=Nk, scale_ados=True, verbose=False, **model)
    gm = regrid(g, m)   # man könnte auch gleich build_grid mit der richtigen dim aufrufen aber regrid ist gut für parameter sweeps (z.B. über m)das es viel schneller ist
    gm['dt_fs'] = dt_fs
    gm['depth'] = depth
    gm['Nk'] = Nk
    gm['model'] = model
    gm['rho0'] = rho0
    gm['n_sites'] = n_sites

    # --- SVD-Dilatation nach Seneviratne et al. ---------------------------
    # M = U Sigma V^dag; dilatiert wird NUR die Diagonale.  U und V wirken auf
    # ein Qubit weniger als die volle Sz.-Nagy-Dilatation (4^n statt 4^(n+1)),
    # und eine Diagonale kostet ~2^n.  Gemessen bei m = 4 auf linearer Kette:
    # 63 statt 166 Zweiqubit-Gatter bei t = 6, also Faktor 2.6.
    # sigma_j <= 1 ist garantiert, weil s >= ||H_m||.
    Mpad = np.zeros((gm['mp'], gm['mp']), complex)
    Mpad[:gm['m'], :gm['m']] = gm['Hm'] / gm['s']
    Usvd, sig, Vh = np.linalg.svd(Mpad)
    sig = np.clip(sig, 0.0, 1.0)
    gm['svd'] = dict(U=Usvd, sigma=sig, Vh=Vh,
                     diag=np.concatenate([sig + 1j * np.sqrt(1 - sig ** 2),
                                          sig - 1j * np.sqrt(1 - sig ** 2)]))
    if verbose:
        print(f"  {n_sites} Sites, Tiefe {depth}, Nk = {Nk}")
        print(f"  Gitter: m = {gm['m']} -> gepolstert {gm['mp']}, "
              f"{gm['n_qubits']} Qubits, U ist {gm['U'].shape[0]}x{gm['U'].shape[1]}")
        print(f"  s = {gm['s']:.8f}   ||H_m|| = {np.linalg.norm(gm['Hm'], 2):.6f}")
        print(f"  Singulaerwerte von H_m/s: {gm['svd']['sigma'].min():.6f} bis "
              f"{gm['svd']['sigma'].max():.6f}")
        print(f"  Aufbau {time.time() - t0:.1f} s (einmalig, unabhaengig von der "
              f"Zahl der Zeitschritte)")
    return gm


def _prep_unitary(chi):
    """B unitaer mit B|0...0> = chi (QR mit Phasenkorrektur).

    Die Phasenkorrektur Q * dg/|dg| macht aus der QR-Zerlegung die eindeutige
    Variante mit reellem, positivem R-Diagonal.  Achtung: hat chi eine exakte
    Null an einer Stelle, wird auch ein Diagonalelement von R exakt null, und
    dg/|dg| gibt 0/0 = nan -- die Matrix ist dann nicht mehr unitaer und
    UnitaryGate wirft ValueError.  Das passiert bei jeder Ablesezeile mit einer
    exakten Null, also z.B. beim Nullkanal aus `_null_row`.  Phase 1 ist dort
    die richtige Wahl: die Spalte ist ohnehin schon orthonormal.
    """
    n = chi.shape[0]
    M = np.eye(n, dtype=complex)
    M[:, 0] = chi
    Q, R = np.linalg.qr(M)
    dg = np.diag(R).copy()
    dg[np.abs(dg) < 1e-14] = 1.0
    return Q * (dg / np.abs(dg))


def readout_rows(grid, pairs=None):
    """Die Ablesezeilen: d Populationen, je Paar zwei fuer Re und Im.

    vec rho_S = R y, also ist rho_jj = r_jj . y mit r_jj der Zeile j + d*j.
    Im komprimierten Krylov-Raum sind diese Zeilen DICHT -- anders als ohne
    Kompression braucht deshalb jede eine eigene Basisdrehung.
    """
    d, R = grid['d'], grid['R']
    row = lambda i, j: R[i + d * j, :]
    out = [(('pop', j, j), row(j, j)) for j in range(d)]
    for (a, b) in (pairs or []):
        out.append((('pp', a, b),
                    0.5 * (row(a, a) + row(b, b) + row(a, b) + row(b, a))))
        out.append((('RR', a, b),
                    0.5 * (row(a, a) + row(b, b) - 1j * row(a, b) + 1j * row(b, a))))
    return out


# --------------------------------------------------------------------------
# 2.  hardwaretaugliche Schaltkreise
# --------------------------------------------------------------------------

def _step(qc, q_sys, q_anc, grid, method):
    """Ein Zeitschritt.  Beide Varianten sind Dilatationen DESSELBEN M -- der
    |0>-Ancillazweig traegt in beiden Faellen M, sie unterscheiden sich nur im
    verworfenen |1>-Zweig.  Die Durchgaengigkeit des Gitters aendert sich nicht:
    das Systemregister wird nie gemessen und nie neu praepariert."""
    if method == 'sznagy':
        qc.append(UnitaryGate(grid['U'], label='U'), list(q_sys) + list(q_anc))
    else:
        sv = grid['svd']
        qc.append(UnitaryGate(sv['Vh'], label='V+'), q_sys[:])
        qc.h(q_anc[0])
        qc.append(DiagonalGate(list(sv['diag'])), list(q_sys) + list(q_anc))
        qc.h(q_anc[0])
        qc.append(UnitaryGate(sv['U'], label='U'), q_sys[:])


def _null_row(grid, t):
    """Ablesezeile mit q_wahr = 0 EXAKT -- misst den Rauschboden.

    Der dominante Fehler des ersten Hardware-Laufs war kein Tiefenzerfall,
    sondern ein ADDITIVER Boden: Kanaele mit q_wahr ~ 0 lieferten q ~ 0.066,
    und weil rho ~ sqrt(q) eingeht, wurde daraus eine Population von 0.26
    statt 0.  Bei t=1 war der Kanal ('pop',2,2) so um Faktor 466 zu gross.

    Diese Zeile steht senkrecht auf dem idealen Krylow-Vektor x_t =
    (Hm/s)^t y0, also ist |<r_null, x_t>|^2 = 0 ohne jede Naeherung.  Was die
    Maschine hier trotzdem zaehlt, IST der Boden -- direkt gemessen, nicht
    geschaetzt.  Bei rein depolarisierendem Rauschen gilt
        q_mess = (1 - p) q_wahr + p/mp,     q_null = p/mp,
    also  q_korr = (q_mess - q_null) / (1 - mp*q_null).

    Kostet einen Schaltkreis je Zeitschritt, bei 10 Zeiten also +17 % Laufzeit.
    """
    # 1. Normierte Propagationsmatrix des komprimierten Krylov-Raums bilden
    # Hm ist die effektive HEOM-Matrix, s die Skalierungskonstante (s >= ||Hm||)
    A = grid['Hm'] / grid['s']
    
    # 2. Anfangszustand im Krylov-Raum laden und als komplexe Kopie initialisieren
    x = grid['y0'].astype(complex).copy()

    # 3. Den Zustand x über t diskrete Zeitschritte klassisch propagieren: x_t = A^t * x_0
    for _ in range(t):
        x = A @ x

    # 4. Euklidische Norm des resultierenden Zustandsvektors berechnen
    n = np.linalg.norm(x)

    # 5. Numerischer Schutz vor Division durch Null (z. B. wenn x vollständig dissipiert ist)
    # Falls der Vektor praktisch verschwindet, wird ein beliebiger Einheitsvektor zurückgegeben
    if n < 1e-14:                       # entartet -- irgendein Einheitsvektor
        r = np.zeros(len(x), complex) 
        r[0] = 1.0
        return r
    
    # 6. Den aktuellen Zustandsvektor exakt auf Einheitslänge normieren: ||x|| = 1
    x = x / n

    # 7. Die Standard-Basisachse finden, zu der x den KLEINSTEN Betrag / Überlapp hat
    # Grund: Startet man mit dem kleinsten Überlapp, bleibt bei der Projektion der
    # Hauptanteil des Einheitsvektors erhalten -> verhindert numerische Auslöschung
    j = int(np.argmin(np.abs(x)))

    # 8. Den entsprechenden kanonischen Einheitsvektor e_j = [0, ..., 1, ..., 0]^T aufbauen
    e = np.zeros(len(x), complex) 
    e[j] = 1.0

    # 9. Gram-Schmidt-Orthogonalisierung:
    # Ziehe die parallele Komponente <x, e_j> * x vollständig von e_j ab.
    # Da ||x|| = 1 gilt, ist r nun mathematisch exakt orthogonal zu x (d.h. <x, r> = 0).
    r = e - np.vdot(x, e) * x

    # 10. Den orthogonalen Vektor r auf Länge 1 normieren und als Ablesezeile zurückgeben
    return r / np.linalg.norm(r)


def hardware_circuits(grid, times, pairs=None, *, method='svd', defer=True,
                      floor=False):
    """Je Auslesezeit und Ablesezeile ein Schaltkreis.

    `defer=True` (Voreinstellung): AUFGESCHOBENE MESSUNG.  Statt eine Ancilla zu
    messen und zurueckzusetzen, bekommt jeder Zeitschritt ein eigenes frisches
    Ancilla-Qubit, und ALLE Messungen stehen am Ende.  Nach dem Prinzip der
    aufgeschobenen Messung ist das exakt aequivalent, solange kein Gatter von
    einem Messergebnis abhaengt -- und das tut hier keines.

    Warum: eine Mid-Circuit-Messung dauert Mikrosekunden, und waehrenddessen
    dephasieren die benachbarten Systemqubits; der Reset ist auf IBM-Hardware
    selbst wieder eine Messung mit bedingtem X.  Im ersten Hardware-Lauf
    (ibm_kingston, t=1..10) liess sich die Messung mit den publizierten
    Fehlerraten NICHT erklaeren -- 2Q = 3.05e-3 gab RMS 0.134 gegen 0.148
    rauschfrei, erst ein effektiver 2Q-Fehler von 5 % (16-fach) traf die Daten
    (RMS 0.056).  Und das gemessene q hing kaum von t ab: der Kanal ('pp',0,1)
    lieferte 0.443/0.433/0.417/0.434 bei t=1,2,3,10, waehrend ideal
    0.449/0.659/0.772/0.666 stuenden.

    Gemessene Kosten auf ibm_kingston (echte Heavy-Hex-Topologie, max. Grad 3,
    NICHT die optimistische lineare Kette): bei t=10 kostet defer 138 statt 103
    CZ, also +34 % durch SWAP-Routing -- alle t Ancillas muessen dasselbe
    kry-Paar beruehren.  Dafuer schrumpft die Kreisdauer von 53.0 us auf
    15.2 us, denn Messung (2.18 us) und Reset (2.21 us) machten bei t=10
    43.9 us aus, also 83 % des alten Kreises; solange lag das Systemregister
    untaetig gegen T2 = 143 us.

    Nur legale Befehle: unitary, measure, reset.  Kein save_statevector.
    Klassisches Register: Bits 0..t-1 die Ancilla-Aufzeichnung, danach das
    Systemregister -- so gibt es keine Zweideutigkeit in Qiskits
    Registerreihenfolge.
    """
    mp = grid['mp']
    n_sys = int(np.log2(mp))
    rows = readout_rows(grid, pairs)

    jobs = []
    for t in times:
        # Nullkanal je Zeitschritt: misst den additiven Rauschboden direkt
        rows_t = list(rows) + ([(('null', -1, -1), _null_row(grid, t))]
                               if floor else [])
        for tag, r in rows_t:
            nv = float(np.linalg.norm(r))
            chi = np.zeros(mp, complex)
            chi[:len(r)] = np.conj(r) / nv
            B = _prep_unitary(chi)

            q_sys = QuantumRegister(n_sys, 'kry')
            q_anc = QuantumRegister(t if defer else 1, 'anc')
            creg = ClassicalRegister(t + n_sys, 'c')  # ertsen t für ancilla, danach n_sys für finale systemmessung
            qc = QuantumCircuit(q_sys, q_anc, creg)
            # y0 = ||x~0|| e_1  ->  das Register startet schon richtig
            for k in range(1, t + 1):
                anc = [q_anc[k - 1]] if defer else [q_anc[0]]
                _step(qc, q_sys, anc, grid, method)
                if not defer:                      # messen und zuruecksetzen
                    qc.measure(anc[0], creg[k - 1])
                    qc.reset(anc[0])
            if defer:                              # alle Ancillas erst am Ende
                for k in range(t):
                    qc.measure(q_anc[k], creg[k])

            qc.append(UnitaryGate(B.conj().T, label='B+'), q_sys[:])

            for i in range(n_sys):
                qc.measure(q_sys[i], creg[t + i])

            jobs.append(dict(t=t, tag=tag, norm=nv, n_sys=n_sys, defer=defer,
                             method=method, circuit=qc))
    return jobs


def _split_counts(counts, n_steps, n_sys):
    """(Gesamt, akzeptiert, Treffer auf |0...0>) aus den Zaehlraten."""
    total = acc = hit = 0
    for bits, c in counts.items():
        r = bits.replace(' ', '')[::-1]
        total += c
        if set(r[:n_steps]) - {'0'}:
            continue
        acc += c
        if set(r[n_steps:n_steps + n_sys]) <= {'0'}:
            hit += c
    return total, acc, hit


# --------------------------------------------------------------------------
# 3.  vor dem Absenden: was kostet das
# --------------------------------------------------------------------------

def report_cost(grid, jobs, *, shots=4096, backend=None, e_2q=2.66e-3,
                t_2q=68e-9, t_meas=2.18e-6, t_reset=2.21e-6, rep_delay=311e-6,
                coupling_map='linear', verbose=True):
    """Transpilieren, Gatter zaehlen, Treue und QPU-Zeit schaetzen.

    Ohne `backend` wird gegen die Heron-Basisgatter transpiliert.  Standard ist
    eine LINEARE Kette, denn das Heavy-Hex-Gitter von IBM hat keine Dreiecke --
    drei paarweise verbundene Qubits gibt es dort nicht, und die fehlenden
    Verbindungen kosten SWAPs.  Gemessen: all-to-all 99 Zweiqubit-Gatter bei
    t = 6, linear 167, also Faktor 1.7.  `coupling_map=None` gibt die
    optimistische all-to-all-Schranke, `backend=...` die echte ISA.

    Die QPU-Zeit wird von `rep_delay` dominiert -- der Wartezeit zwischen den
    Shots, damit die Qubits relaxieren.  Die Voreinstellungen sind an einem
    ECHTEN Lauf kalibriert, nicht geraten: Job da833pe0ukec7383ughg auf
    ibm_kingston (60 Kreise, defer=False, 2048 Shots) kostete laut
    job.usage() genau 42 s, wovon nur 3.7 s auf Gatter entfielen.  Daraus
    folgt rep_delay = 311 us.  t_meas und t_reset stammen aus
    backend.target (2.18 bzw. 2.21 us) -- die frueheren 1.0 us waren um
    Faktor 2 zu klein und die Schaetzung entsprechend 1.24x zu optimistisch.
    """
    if backend is not None:
        tq = transpile([j['circuit'] for j in jobs], backend=backend,
                       optimization_level=3)
    else:
        # Breite aus dem BREITESTEN Schaltkreis nehmen -- bei aufgeschobener
        # Messung braucht t=10 zwoelf Qubits, nicht drei.
        nq = max(j['circuit'].num_qubits for j in jobs)
        cm = ([[i, i + 1] for i in range(nq - 1)]
              + [[i + 1, i] for i in range(nq - 1)]) \
            if coupling_map == 'linear' else coupling_map
        tq = transpile([j['circuit'] for j in jobs], basis_gates=HERON_BASIS,
                       coupling_map=cm, optimization_level=3,
                       seed_transpiler=7)
    for j, c in zip(jobs, tq):
        j['transpiled'] = c
        ops = c.count_ops()
        j['n_2q'] = ops.get('cz', 0) + ops.get('cx', 0) + ops.get('ecr', 0)
        j['depth'] = c.depth()

    n2 = np.array([j['n_2q'] for j in jobs])
    ts = np.array([j['t'] for j in jobs])
    fid = (1 - e_2q) ** n2
    # Bei aufgeschobener Messung gibt es KEINE Messung und KEINEN Reset
    # mitten im Schaltkreis -- alle Qubits werden am Ende gleichzeitig
    # ausgelesen.  Das spart je Schritt t_meas + t_reset ~ 2 us.
    defer = bool(jobs[0].get('defer', False))
    dur = (n2 * t_2q + t_meas if defer
           else ts * (t_meas + t_reset) + n2 * t_2q + t_meas)
    qpu = float(np.sum((dur + rep_delay) * shots))
    nq_max = max(j['transpiled'].num_qubits for j in jobs)

    if verbose:
        breite = (f"{grid['n_qubits']} System- + bis {nq_max - grid['n_qubits']} "
                  f"Ancilla-Qubits = {nq_max}" if defer
                  else f"{grid['n_qubits']} System- + 1 Ancilla-Qubit")
        print(f"  {len(jobs)} Schaltkreise, {breite}, "
              f"{shots} Shots je Schaltkreis, Dilatation "
              f"'{jobs[0].get('method', 'sznagy')}'")
        print(f"  Messung: {'aufgeschoben -- 0 Resets, alles am Ende' if defer else 'mitten im Kreis, mit Reset'}")
        print(f"  Zweiqubit-Gatter: {n2.min()} bis {n2.max()} "
              f"(Median {int(np.median(n2))}), Tiefe bis "
              f"{max(j['depth'] for j in jobs)}")
        print(f"\n  {'t':>3s} {'2Q-Gatter':>10s} {'Tiefe':>6s} "
              f"{'Treue (Gatter)':>15s}")
        for t in sorted(set(ts)):
            sel = ts == t
            print(f"  {t:3d} {int(np.median(n2[sel])):10d} "
                  f"{int(np.median([j['depth'] for j in jobs if j['t']==t])):6d} "
                  f"{np.median(fid[sel]):15.3f}")
        print(f"\n  geschaetzte QPU-Zeit : {qpu:.1f} s  ({qpu/60:.2f} min)")
        print(f"    davon rep_delay    : {rep_delay*shots*len(jobs):.1f} s "
              f"({rep_delay*shots*len(jobs)/qpu:.0%})")
        topo = ('echte ISA' if backend is not None else
                'lineare Kette' if coupling_map == 'linear' else 'all-to-all')
        print(f"  Annahmen: {e_2q:.2e} Fehler je 2Q-Gatter, {t_2q*1e9:.0f} ns "
              f"je 2Q, {rep_delay*1e6:.0f} us rep_delay, Topologie: {topo}")
        print(f"  (rep_delay/t_meas/t_reset kalibriert an Job "
              f"da833pe0ukec7383ughg: 42 s tatsaechlich)")
    return dict(n_2q=n2, fidelity=fid, qpu_seconds=qpu, circuits=len(jobs))


# --------------------------------------------------------------------------
# 4.  kostenlos gegenpruefen
# --------------------------------------------------------------------------

def verify_in_aer(grid, jobs, *, shots=4096, pairs=None, verbose=True):
    """Dieselben Schaltkreise in Aer -- kostet nichts und findet Fehler frueh."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method='statevector')
    circs = [j.get('transpiled', j['circuit']) for j in jobs]
    res = sim.run(transpile(circs, sim), shots=shots).result()
    counts = [res.get_counts(i) for i in range(len(jobs))]
    return analyze(grid, jobs, counts, pairs=pairs, verbose=verbose,
                   quelle='Aer')


# --------------------------------------------------------------------------
# 5.  auf die echte Maschine
# --------------------------------------------------------------------------

def _sampler_options(twirling=True, dd=True, verbose=True):
    """Fehlerunterdrueckung fuer SamplerV2, defensiv aufgebaut.

    twirling  Pauli-Twirling der Zweiqubit-Gatter und der Messung.  Wandelt
              KOHAERENTE Fehler in stochastische um.  Genau der richtige Hebel
              fuer einen Schaltkreis, der DASSELBE Gatter n-mal wiederholt:
              ein systematischer Rotationsfehler addiert sich sonst kohaerent
              auf (Fehler ~ n statt ~ sqrt(n)).
    dd        Dynamical Decoupling (XY4) auf den Leerlaufzeiten.  Hier besonders
              wirksam, weil das Systemregister waehrend JEDER Ancilla-Messung
              und jedes Resets untaetig herumliegt: bei ~2 us je Schritt und
              10 Schritten sind das ~20 us Leerlauf gegen T2 ~ 200 us.
              Bei defer=True schuetzt es zusaetzlich die wartenden Ancillas,
              die bis zu ihrem Schritt in |+> liegen.

    Beides kostet keine zusaetzliche QPU-Zeit -- Twirling verteilt die Shots nur
    auf mehrere zufaellige Varianten desselben Schaltkreises.
    """
    from qiskit_ibm_runtime.options import SamplerOptions
    o = SamplerOptions()
    try:
        o.twirling.enable_gates = bool(twirling)
        o.twirling.enable_measure = bool(twirling)
        o.dynamical_decoupling.enable = bool(dd)
        if dd:
            o.dynamical_decoupling.sequence_type = 'XY4'
            # ALAP = "as late as possible".  Entscheidend bei aufgeschobener
            # Messung: die t Ancillas haben KEINEN Vorgaenger, Qiskit zeichnet
            # ihr H daher ganz an den Anfang (siehe circuit.draw()).  Dann laege
            # Ancilla 10 den GANZEN Schaltkreis lang in |+>, und |+> dephasiert
            # mit T2 ~ 100 us, waehrend |0> nur T1 ~ 300 us sieht.  ALAP schiebt
            # jedes H direkt vor sein Diagonal-Gatter -- der Leerlauf in |+>
            # schrumpft von "ganzer Kreis" auf "ein Schritt", und das kostet
            # kein einziges Gatter.  (Eine Barriere wuerde dasselbe erzwingen,
            # aber +22 % CZ kosten, weil dann U und V+ ueber die Schrittgrenze
            # nicht mehr verschmolzen werden duerfen: 82 -> 100 CZ bei t=10.)
            o.dynamical_decoupling.scheduling_method = 'alap'
    except Exception as e:                      # falls sich die API aendert
        if verbose:
            print(f"  WARNUNG: Mitigation-Optionen nicht gesetzt ({e}) -- "
                  f"Lauf OHNE Twirling/DD.")
        return SamplerOptions()
    if verbose:
        print(f"  Mitigation: Twirling {'an' if twirling else 'aus'}, "
              f"Dynamical Decoupling XY4 {'an' if dd else 'aus'}")
    return o


def isa_report(jobs, verbose=True):
    """Was tatsaechlich auf der Maschine laeuft.

    Der ISA-Schaltkreis in `j['transpiled']` IST der ausgefuehrte -- die Runtime
    zerlegt ihn nicht noch einmal.  Twirling und Dynamical Decoupling fuegen zur
    Laufzeit nur Pauli-Gatter bzw. Pulse ein und aendern die Struktur nicht.

    Ansehen mit  jobs[i]['transpiled'].draw('mpl', fold=-1)
    Speichern mit qiskit.qpy.dump([j['transpiled'] for j in jobs], open('isa.qpy','wb'))
    Nach dem Lauf liefert auch  job.inputs['pubs']  die abgeschickten Kreise zurueck.
    """
    tq = [j.get('transpiled') for j in jobs]
    if any(c is None for c in tq):
        raise RuntimeError("noch nicht transpiliert -- erst report_cost(...) "
                           "oder run_on_backend(...) aufrufen")
    if verbose:
        print(f"  ISA-Schaltkreise (das, was wirklich laeuft):")
        print(f"    {'t':>3s} {'Kanal':>14s} {'2Q':>5s} {'Tiefe':>6s} {'Qubits':>7s}")
        for j, c in zip(jobs, tq):
            o = c.count_ops()
            n2 = o.get('cz', 0) + o.get('cx', 0) + o.get('ecr', 0)
            print(f"    {j['t']:3d} {str(j['tag']):>14s} {n2:5d} {c.depth():6d} "
                  f"{c.num_qubits:7d}")
    return tq


def run_on_backend(jobs, *, backend_name='ibm_kingston', shots=4096,
                   instance=None, twirling=True, dd=True, optimization_level=3,
                   verbose=True):
    """Absenden und auf das Ergebnis warten.  Braucht qiskit-ibm-runtime.

    Der Token wird einmalig hinterlegt mit

        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(token='...', overwrite=True)

    Backend-Wahl fuer 63 CZ bei t=6 und 2 Auslesequbits:

        QPU             Typ        2Q layered   Auslesen   Treue    Queue
        ibm_kingston    Heron r2      3.05e-3   1.001e-2   0.809       41   <-
        ibm_fez         Heron r2      4.75e-3   9.888e-3   0.727      568
        ibm_marrakesh   Heron r2      5.16e-3   1.318e-2   0.703       20

    kingston hat die besten Gatter und praktisch dasselbe Auslesen wie fez,
    dessen Queue mit 568 Jobs unbrauchbar ist.  marrakesh hat die kuerzeste
    Schlange, aber 70 % mehr Gatterfehler.

    `twirling` und `dd` schalten Fehlerunterdrueckung zu, siehe
    `_sampler_options`.  Beides kostet keine zusaetzliche QPU-Zeit.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError as e:
        raise ImportError(
            "qiskit-ibm-runtime fehlt.  Installieren mit\n"
            "    pip install qiskit-ibm-runtime\n"
            f"(urspruenglicher Fehler: {e})")

    service = QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()
    backend = service.backend(backend_name)
    if verbose:
        print(f"  Backend {backend.name}: {backend.num_qubits} Qubits, "
              f"Basisgatter {backend.operation_names}")

    # erstellt in Qiskit eine vordefinierte Transpiler-Pipeline (Pass Manager), die abstrakte Quantenschaltkreise in hardwarekonforme ISA-Schaltkreise (Instruction Set Architecture) für das angegebene backend übersetzt.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)  # optimization_level=3 ist max -> qc wird in so wenig gatter wie möglich zerlegt
    # läuft noch lokal auf meinem PC
    isa = pm.run([j['circuit'] for j in jobs])
    for j, c in zip(jobs, isa):
        j['transpiled'] = c

    if verbose:
        isa_report(jobs)

    sampler = SamplerV2(mode=backend, options=_sampler_options(twirling, dd, verbose))
    t0 = time.time()

    # sendet den Job an die echte QPU und verbraucht Rechenzeit
    job = sampler.run(isa, shots=shots)

    if verbose:
        print(f"  Job {job.job_id()} abgesendet.", flush=True)
        print(f"  Falls die Anzeige unten abbricht, weiter mit:", flush=True)
        print(f"      hw.watch_job('{job.job_id()}')", flush=True)

    # Warteschlange live mitverfolgen (blockiert bis fertig, Strg-C beendet
    # nur die Anzeige).  Danach liegt das Ergebnis bereit.
    if verbose:
        watch_job(job, verbose=True)

    # Das ergebnis von IBM
    result = job.result()

    if verbose:
        print(f"  fertig nach {time.time() - t0:.0f} s Wandzeit")

    out = []
    for i, j in enumerate(jobs):
        creg = j['transpiled'].cregs[0].name
        out.append(getattr(result[i].data, creg).get_counts())
        
    return out


def watch_job(job, *, interval=15, service=None, verbose=True):
    """Live-Anzeige: wie viele Jobs stehen noch vor meinem?

    `job` ist entweder das Objekt aus `sampler.run(...)` oder eine Job-ID als
    Zeichenkette (dann wird sie beim Dienst nachgeschlagen).

    ACHTUNG, haeufige Falle: `job.queue_info()` gibt es hier NICHT.  Das ist die
    alte qiskit-ibm-provider-API; RuntimeJobV2 kennt die Methode nicht und wirft
    AttributeError.  Die Warteschlange steht stattdessen in `job.metrics()`:

        {'position_in_queue': 12, 'position_in_provider': 3,
         'estimated_start_time': '2026-08-27T14:02:11Z', ...}

    Beide Felder duerfen None sein -- solange IBM noch keine Schaetzung hat,
    oder sobald der Job laeuft.  Darum wird hier alles defensiv abgefragt.

    Abbrechen mit Strg-C beendet nur die ANZEIGE, nicht den Job.

    Rueckgabe: das Job-Objekt im Endzustand -- `job.result()` liefert dann sofort.
    """
    if isinstance(job, str):
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = service or QiskitRuntimeService()
        job = service.job(job)

    t0, letzte = time.time(), None
    try:
        while not job.in_final_state():
            stat = str(job.status())
            try:                                # metrics() darf fehlschlagen
                m = job.metrics() or {}
            except Exception:
                m = {}
            pos = m.get('position_in_queue')
            start = m.get('estimated_start_time')
            zeile = f"  [{time.time() - t0:5.0f} s] {stat:<10s}"
            if pos is not None:
                zeile += f" | noch {pos:>4d} Jobs vor mir"
            elif stat == 'RUNNING':
                zeile += " | laeuft auf der QPU"
            else:
                zeile += " | Position noch unbekannt"
            if start:
                zeile += f" | Start ~ {str(start)[:19]}"
            if verbose and zeile != letzte:      # nur bei Aenderung neu drucken
                print(zeile, flush=True)
                letzte = zeile
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n  Anzeige abgebrochen -- der Job {job.job_id()} LAEUFT WEITER.")
        print(f"  Spaeter fortsetzen mit  hw.watch_job('{job.job_id()}')")
        return job

    stat = str(job.status())
    print(f"  [{time.time() - t0:5.0f} s] Endzustand: {stat}")
    if stat == 'ERROR' and verbose:
        print(f"  Fehlermeldung: {job.error_message()}")
    try:                                         # tatsaechlich verbrauchte QPU-Zeit
        u = job.usage()
        print(f"  verbrauchte QPU-Zeit: {u} s")
    except Exception:
        pass
    return job


# --------------------------------------------------------------------------
# 5b.  Staffel-Schema:  dt = T/k, immer k Schritte, ein Gitter je Zielzeit
# --------------------------------------------------------------------------
#
# Die Krylov-Kompression ist EXAKT fuer t <= m-1 -- der Raum
# K_m = span{y0, P y0, ..., P^(m-1) y0} enthaelt P^t y0 dann per Konstruktion.
# Bei m = 4 sind das drei Schritte.  Statt EINEN Schaltkreis ueber viele
# Schritte laufen zu lassen (und dabei den Kompressionsfehler einzusammeln),
# baut dieses Schema je Zielzeit T ein eigenes Gitter mit dt = T/k und geht
# immer nur k = 3 Schritte.  Gemessen ueber T = 60 .. 1000 fs:
#
#   T[fs]   dt[fs]      s     p_ideal   CZ  Treue   Kompressionsfehler
#      60    20.00  0.994281   0.9405   35  0.899        0.0000000
#     400   133.33  0.975038   0.8004   35  0.899        0.0000002
#    1000   333.33  1.014031   0.5054   35  0.899        0.0000001
#
# Zum Vergleich: ein Kreis mit 10 Schritten bei dt = 20 fs hat 138 CZ, Treue
# 0.656 und Kompressionsfehler 0.038; mit 25 Schritten 440 CZ, Treue 0.261,
# Fehler 0.116 und eine Akzeptanz von 0.002.
#
# WICHTIG, damit die Aussage ehrlich bleibt: der Hebel T/dt = k betraegt hier
# nur 3.  Der Schaltkreis verdreifacht also die klassisch vorbereitete Zeit,
# waehrend ein 10-Schritt-Lauf sie verzehnfacht.  Bei k = 1 waere der Hebel 1,
# und dann berechnete die klassische Vorbereitung exp(A T) -- also bereits die
# Antwort.  k >= 2 ist die Untergrenze, unter der das Verfahren seinen Sinn
# verliert.

# Nur diese Schluessel werden je Gitter behalten.  Ein volles Gitter enthaelt
# fuenf 720 x 720-Matrizen (A, P, Pt, Wh, Wih), zusammen rund 20 MB -- bei 50
# Zielzeiten waeren das ueber 5 GB.  Schaltkreis und Auswertung brauchen davon
# nichts.
_STAFFEL_KEYS = ('U', 's', 'm', 'mp', 'n_qubits', 'y0', 'R', 'Hm', 'd', 'D',
                 'dt_fs', 'depth', 'Nk', 'model', 'rho0', 'svd', 'n_sites')


def _schlank(grid):
    """Gitter auf das reduzieren, was Schaltkreis und Ablesung brauchen."""
    return {k: grid[k] for k in _STAFFEL_KEYS if k in grid}


def staffel_grids(times_fs, *, n_sites=4, m=4, k=3, depth=2, Nk=1, delta=10.0,
                  model=None, rho0=None, cache=None, verbose=True):
    """Ein schlankes Gitter je Zielzeit T, gebaut mit dt = T/k.

    `times_fs`  die Zielzeiten in fs, z.B. np.arange(20, 1001, 20)
    `k`         Schritte je Schaltkreis; muss <= m-1 bleiben, sonst ist die
                Kompression nicht mehr exakt (bei m=4 also k <= 3)
    `cache`     Pfad einer .npz; vorhanden wird geladen, sonst nach dem Bau
                geschrieben.  Ein Bau kostet ~5 s (expm einer 720 x 720-Matrix
                plus Arnoldi), 50 Zielzeiten also rund vier Minuten.

    Returns eine Liste [(T, grid), ...].
    """
    if k > m - 1:
        raise ValueError(f"k = {k} > m-1 = {m-1}: die Krylov-Kompression waere "
                         f"nicht mehr exakt.  Entweder k senken oder m erhoehen "
                         f"(m=8 erlaubt k<=7, kostet aber 78 statt 14 CZ je Schritt).")
    times_fs = [float(T) for T in times_fs]

    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        gs = list(z['grids'])
        Ts = list(z['times'])
        if [round(t, 6) for t in Ts] == [round(t, 6) for t in times_fs]:
            if verbose:
                print(f"  {len(gs)} Gitter aus {cache} geladen")
            return list(zip(Ts, gs))
        if verbose:
            print(f"  {cache} passt nicht zu diesen Zielzeiten -- neu gebaut")

    if model is None:
        model, rho0 = make_model(n_sites)
    elif rho0 is None:
        rho0 = np.diag([1.0] + [0.0] * (model['H'].shape[0] - 1))

    out, t0 = [], time.time()
    for i, T in enumerate(times_fs):
        g = build_hardware_grid(n_sites=n_sites, m=m, dt_fs=T / k, delta=delta,
                                depth=depth, Nk=Nk, model=model, rho0=rho0,
                                verbose=False)
        out.append((T, _schlank(g)))
        if verbose and ((i + 1) % 10 == 0 or i + 1 == len(times_fs)):
            print(f"    {i+1}/{len(times_fs)} Gitter gebaut "
                  f"({time.time() - t0:.0f} s)", flush=True)

    if cache:
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        np.savez_compressed(cache,
                            grids=np.array([g for _, g in out], dtype=object),
                            times=np.array([T for T, _ in out]))
        if verbose:
            print(f"  Gitter gesichert in {cache}")
    return out


def staffel_circuits(grids, pairs=None, *, k=3, method='svd', defer=True,
                     floor=False):
    """Je Zielzeit k Schritte.  Jeder Job merkt sich, zu welchem Gitter er gehoert.

    Anders als `hardware_circuits` laeuft hier NICHT ein Gitter ueber viele
    Zeiten, sondern viele Gitter ueber je dieselbe Schrittzahl k.  Die Jobs
    tragen daher zusaetzlich `T_fs` und `grid_index`.
    """
    jobs = []
    for gi, (T, g) in enumerate(grids):
        for j in hardware_circuits(g, [k], pairs, method=method, defer=defer,
                                   floor=floor):
            j['T_fs'] = float(T)
            j['grid_index'] = gi
            jobs.append(j)
    return jobs


def analyze_staffel(grids, jobs, counts_list, *, pairs=None, verbose=True,
                    quelle='QPU'):
    """Wie `analyze`, aber je Zielzeit mit dem GITTER dieser Zeit.

    Entscheidend: s, ||y0|| und die Ablesenormen ||r_j|| gehoeren jeweils zu
    dem dt, mit dem der betreffende Schaltkreis gebaut wurde -- sie sind hier
    also NICHT global, sondern je Zeitpunkt verschieden.
    """
    d = grids[0][1]['d']
    pairs = list(pairs or [])
    Ts = [float(T) for T, _ in grids]
    idx = {(j['grid_index'], j['tag']): i for i, j in enumerate(jobs)}

    rho = np.full((len(Ts), d, d), np.nan, complex)
    rho_tr = np.full((len(Ts), d, d), np.nan, complex)
    err_re = np.full((len(Ts), d, d), np.nan)
    err_im = np.full((len(Ts), d, d), np.nan)
    prob = np.full(len(Ts), np.nan)
    q_null = np.full(len(Ts), np.nan)

    for gi, (T, g) in enumerate(grids):
        s = g['s']
        scale0 = float(np.linalg.norm(g['y0']))
        vals, accs = {}, {}
        for j in jobs:
            if j['grid_index'] != gi:
                continue
            i = idx[(gi, j['tag'])]
            tot, acc, hit = _split_counts(counts_list[i], j['t'], j['n_sys'])
            accs[j['tag']] = acc
            vals[j['tag']] = (hit / acc if acc else np.nan, acc, j['norm'])
        a0 = accs.get(('pop', 0, 0), 0)
        if not a0:
            continue
        i0 = idx[(gi, ('pop', 0, 0))]
        t_steps = jobs[i0]['t']
        prob[gi] = a0 / sum(counts_list[i0].values())
        lam = scale0 * s ** t_steps * np.sqrt(prob[gi])

        if ('null', -1, -1) in vals:
            q_null[gi] = vals[('null', -1, -1)][0]

        pop = np.zeros(d)
        dpop = np.zeros(d)
        for j2 in range(d):
            q, acc, nv = vals[('pop', j2, j2)]
            pop[j2] = nv * lam * np.sqrt(max(q, 0.0))
            dpop[j2] = nv * lam / (2 * np.sqrt(acc)) if acc else np.nan
        M = np.diag(pop).astype(complex)
        E, Eim = np.diag(dpop), np.zeros((d, d))

        for (a, b) in pairs:
            got = {}
            for tag in ('pp', 'RR'):
                q, acc, nv = vals[(tag, a, b)]
                got[tag] = (nv * lam * np.sqrt(max(q, 0.0)),
                            nv * lam / (2 * np.sqrt(acc)) if acc else np.nan)
            half = 0.5 * (pop[a] + pop[b])
            quart = 0.25 * dpop[a] ** 2 + 0.25 * dpop[b] ** 2
            M[a, b] = (got['pp'][0] - half) + 1j * (got['RR'][0] - half)
            M[b, a] = np.conj(M[a, b])
            E[a, b] = E[b, a] = np.sqrt(got['pp'][1] ** 2 + quart)
            Eim[a, b] = Eim[b, a] = np.sqrt(got['RR'][1] ** 2 + quart)

        rho[gi], err_re[gi], err_im[gi] = M, E, Eim
        tr = np.real(np.trace(M))
        if tr > 0:
            rho_tr[gi] = M / tr

    # Referenz: qutip auf genau den Zielzeiten, plus t=0 fuer die Kurve
    g0 = grids[0][1]
    t_ref = np.concatenate([[0.0], np.array(Ts)])
    ref = qutip_reference_rho(t_ref, rho0=g0['rho0'], depth=g0['depth'],
                              Nk=g0['Nk'], **g0['model'])[1:]

    ok = np.isfinite(np.real(rho[:, 0, 0]))
    if verbose and ok.any():
        print(f"\n  {quelle} gegen qutip HEOMSolver (Staffel, k={jobs[0]['t']}):")
        for gi in np.where(ok)[0]:
            dp = np.abs(np.real(np.diag(rho[gi]))
                        - np.real(np.diag(ref[gi]))).max()
            dt_ = np.abs(np.real(np.diag(rho_tr[gi]))
                         - np.real(np.diag(ref[gi]))).max()
            bd = (f" | Boden {q_null[gi]:.4f}" if np.isfinite(q_null[gi]) else "")
            print(f"    T = {Ts[gi]:6.0f} fs (dt = {grids[gi][1]['dt_fs']:6.2f}) | "
                  f"p_succ {prob[gi]:.4f} | Spur {np.real(np.trace(rho[gi])):.4f} | "
                  f"max|dPop| ueber p_t {dp:.4f} | spurnormiert {dt_:.4f}{bd}")
    return rho, err_re, dict(T_fs=np.array(Ts), p_success=prob,
                             rho_err_im=err_im, rho_tr=rho_tr, reference=ref,
                             pairs=pairs, q_null=q_null, quelle=quelle)


def verify_staffel_in_aer(grids, jobs, *, shots=4096, pairs=None, verbose=True):
    """Die Staffel-Schaltkreise rauschfrei in Aer -- misst nur den Algorithmus."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(seed_simulator=7)
    counts = []
    for i, j in enumerate(jobs):
        c = transpile(j['circuit'], sim, optimization_level=0)
        r = sim.run(c, shots=shots).result().get_counts()
        counts.append({kk.replace(' ', ''): v for kk, v in r.items()})
        if verbose and (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(jobs)} simuliert", flush=True)
    return analyze_staffel(grids, jobs, counts, pairs=pairs, verbose=verbose,
                           quelle='Aer')


# --------------------------------------------------------------------------
# 6.  auswerten
# --------------------------------------------------------------------------

def analyze(grid, jobs, counts_list, *, pairs=None, verbose=True, quelle='QPU'):
    """Zaehlraten -> Dichtematrix, gegen den exakten qutip-Solver.

    rho_jj   = ||r_jj|| lambda_t sqrt(q_jj),  lambda_t = ||y0|| s^t sqrt(p_t)
    Re rho_ab = rho_++ - (rho_aa + rho_bb)/2,   Im analog ueber rho_RR
    """
    d, s = grid['d'], grid['s']
    scale0 = float(np.linalg.norm(grid['y0']))
    times = sorted(set(j['t'] for j in jobs))
    pairs = list(pairs or [])
    idx = {(j['t'], j['tag']): i for i, j in enumerate(jobs)}

    q_null = np.full(len(times), np.nan)      # gemessener Rauschboden je t
    rho = np.full((len(times), d, d), np.nan, complex)
    rho_tr = np.full((len(times), d, d), np.nan, complex)
    err_re = np.full((len(times), d, d), np.nan)
    err_im = np.full((len(times), d, d), np.nan)
    prob = np.full(len(times), np.nan)

    for k, t in enumerate(times):
        vals, accs = {}, {}
        for j in jobs:
            if j['t'] != t:
                continue
            i = idx[(t, j['tag'])]
            tot, acc, hit = _split_counts(counts_list[i], t, j['n_sys'])
            accs[j['tag']] = acc
            vals[j['tag']] = (hit / acc if acc else np.nan, acc, j['norm'])
        a0 = accs.get(('pop', 0, 0), 0)
        if not a0:
            continue
        prob[k] = a0 / sum(counts_list[idx[(t, ('pop', 0, 0))]].values())
        lam = scale0 * s ** t * np.sqrt(prob[k])

        if ('null', -1, -1) in vals:          # Nullkanal mitgelaufen?
            q_null[k] = vals[('null', -1, -1)][0]
        pop = np.zeros(d)
        dpop = np.zeros(d)
        for j2 in range(d):
            q, acc, nv = vals[('pop', j2, j2)]
            pop[j2] = nv * lam * np.sqrt(max(q, 0.0))
            dpop[j2] = nv * lam / (2 * np.sqrt(acc)) if acc else np.nan
        M = np.diag(pop).astype(complex)
        E, Eim = np.diag(dpop), np.zeros((d, d))

        for (a, b) in pairs:
            got = {}
            for tag in ('pp', 'RR'):
                q, acc, nv = vals[(tag, a, b)]
                got[tag] = (nv * lam * np.sqrt(max(q, 0.0)),
                            nv * lam / (2 * np.sqrt(acc)) if acc else np.nan)
            half = 0.5 * (pop[a] + pop[b])
            quart = 0.25 * dpop[a] ** 2 + 0.25 * dpop[b] ** 2
            M[a, b] = (got['pp'][0] - half) + 1j * (got['RR'][0] - half)
            M[b, a] = np.conj(M[a, b])
            E[a, b] = E[b, a] = np.sqrt(got['pp'][1] ** 2 + quart)
            Eim[a, b] = Eim[b, a] = np.sqrt(got['RR'][1] ** 2 + quart)
        rho[k], err_re[k], err_im[k] = M, E, Eim
        # Zweite, UNABHAENGIGE Ablesung ueber Tr rho_S = 1.  Sie braucht p_t
        # gar nicht -- und genau darin steckt auf Hardware der groesste
        # systematische Fehler: p_t geht ueber lambda_t = s^t sqrt(p_t) in die
        # Skala ein, und ein zu kleines p_t (Reset- und Auslesefehler der
        # Ancilla) zieht ALLE Eintraege gemeinsam nach unten.  Die
        # Spurnormierung kuerzt diesen gemeinsamen Faktor exakt heraus.
        tr = np.real(np.trace(M))
        if tr > 0:
            rho_tr[k] = M / tr

    ts = np.array(times)
    ref = qutip_reference_rho(np.arange(max(times) + 1) * grid['dt_fs'],
                              rho0=grid.get('rho0', RHO0), depth=grid['depth'],
                              Nk=grid['Nk'], **grid.get('model', MODEL))
    ok = np.isfinite(np.real(rho[:, 0, 0]))
    if verbose and ok.any():
        print(f"\n  {quelle} gegen qutip HEOMSolver:")
        for k in np.where(ok)[0]:
            t = ts[k]
            dp = np.abs(np.real(np.diag(rho[k])) - np.real(np.diag(ref[t]))).max()
            dt_ = np.abs(np.real(np.diag(rho_tr[k]))
                         - np.real(np.diag(ref[t]))).max()
            bd = (f" | Boden q0 {q_null[k]:.4f}" if np.isfinite(q_null[k])
                  else "")
            print(f"    t = {t*grid['dt_fs']:4.0f} fs | p_succ {prob[k]:.3f} | "
                  f"Spur {np.real(np.trace(rho[k])):.4f} | max|dPop| "
                  f"ueber p_t {dp:.4f} | spurnormiert {dt_:.4f}{bd}")
    return rho, err_re, dict(t_index=ts, p_success=prob, rho_err_im=err_im,
                             rho_tr=rho_tr, reference=ref, pairs=pairs,
                             q_null=q_null, quelle=quelle)
