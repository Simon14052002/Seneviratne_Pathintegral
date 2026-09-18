"""
4-Site-FMO, MARKOVSCHE Dynamik, auf echter IBM-Hardware -- 3 Qubits.

Was hier anders ist als im nicht-markovschen Ordner
---------------------------------------------------
Nichts an der Maschinerie.  Schaltkreisbau, aufgeschobene Messung,
SVD-Dilatation, Kostenschaetzung, Absenden und Auswertung sind Zeile fuer
Zeile dieselben wie in
`non_markov/hardware_4_site_FMO_with_compression/hardware.py`; ausgetauscht
ist allein der Generator, der in `markov_grid.py` steckt.  Genau deshalb sind
die beiden Laeufe vergleichbar: was sich in den Kurven unterscheidet, ist
Born-Markov gegen die exakte Hierarchie und nicht das Verfahren.

Warum die Route ueber den Superoperator und nicht ueber Kraus-Operatoren
-----------------------------------------------------------------------
Der naheliegende markovsche Weg ist die Stinespring-Dilatation der
Kraus-Operatoren aus `Markov/single_grid_markov.ipynb`: rho_n =
Tr_E[U(|0><0|_E (x) rho_{n-1})U^dag], ein Gatter auf System + Umgebung, in
jedem Schritt dasselbe.  Der ist mathematisch schoener -- CPTP, also
p_success = 1, keine Nachselektion und keine Skala lambda_t.  Er ist auf
heutiger Hardware aber nicht lauffaehig.  Gemessen fuer das 4-Site-FMO bei
lam = 35, dt = 20 fs, lineare Kette, Heron-Basisgatter, als Isometrie
synthetisiert (billiger als die volle Unitaere):

    behaltene Kraus-Op.   n_env   Qubits   CZ je SCHRITT   algorithm. Fehler
    16 (exakt)            4       6        591             0
     8                    3       5        254             5e-3 ... 5e-2
     4                    2       4        109             9e-2 ... 2e-1
     2                    1       3         31             2.3e-1

Der Grund steht im Choi-Spektrum: bei lam = 35 und dt = 20 fs sind die
Eigenwerte 1.729 / 0.810 / 0.574 / 0.420 / 0.155 / 0.134 / ... -- sie fallen
NICHT ab, der Kanal hat vollen Kraus-Rang d^2 = 16.  Abschneiden kostet also
sofort zweistellige Prozente, und 591 CZ je Schritt geben bei 3e-3 Fehler je
CZ schon bei t = 1 eine Treue von 0.17.

Der Superoperator-Weg propagiert stattdessen den VEKTOR vec(rho) durch die
Krylov-komprimierte, geeichte Matrix H_m/s -- 3 Qubits und ~13 CZ je Schritt.
Er bezahlt das mit Nachselektion (p_t ~ 0.98) und der sqrt(q)-Ablesung, genau
wie der nicht-markovsche Lauf.

Ablauf
------
    grid  = build_hardware_grid(m=4, delta=2.0)
    jobs  = hardware_circuits(grid, times=range(1, 11), pairs=[(0, 1)])
    report_cost(grid, jobs)           # VOR dem Absenden: Gatter, Treue, QPU-Zeit
    verify_in_aer(grid, jobs)         # kostenlos gegenpruefen
    counts = run_on_backend(jobs, backend_name='ibm_kingston', shots=2048)
    rho, err, info = analyze(grid, jobs, counts)
"""

import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate, DiagonalGate

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import markov_grid as mg
from markov_grid import (make_model, build_markov_grid, regrid,
                         qutip_reference, qutip_reference_rho,
                         reference_trajectory, post_selection_curve,
                         H_FMO_FULL, FS_TO_CM, KB_CM)

__all__ = ['MODEL', 'RHO0', 'H_FMO_FULL', 'make_model', 'isa_report',
           'build_hardware_grid', 'readout_rows', 'hardware_circuits',
           'report_cost', 'verify_in_aer', 'run_on_backend', 'watch_job',
           'analyze', 'qutip_reference', 'qutip_reference_rho',
           'reference_trajectory', 'post_selection_curve']

N_SITES = 4
MODEL, RHO0 = make_model(N_SITES)        # Voreinstellung: lam = 35, gamma = 50

# Heron-Basisgatter; fuer die Offline-Schaetzung ohne Backend
HERON_BASIS = ['cz', 'rz', 'sx', 'x', 'measure', 'reset']


# --------------------------------------------------------------------------
# 1.  das kleine Gitter
# --------------------------------------------------------------------------

def build_hardware_grid(*, n_sites=4, m=4, dt_fs=20.0, delta=2.0,
                        model=None, rho0=None, verbose=True):
    """Das durchgaengige markovsche Gitter, auf m Krylov-Dimensionen komprimiert.

    m = 4 -> 3 Qubits (2 Krylov + 1 Ancilla), m = 8 -> 4 Qubits, m = 16 -> 5
    Qubits.  Weil D = d^2 = 16 ist, ist m = 16 schon die UNKOMPRIMIERTE
    Rechnung -- der markovsche Raum ist von Haus aus klein, anders als die
    720 Dimensionen der HEOM-Hierarchie.

    Zur Wahl von m und delta: siehe die Tabellen im Notebook.  Kurz --
    m = 4 ist bis t = 3 exakt und driftet dann (5.8e-2 bei t = 10), m = 8 ist
    ueber den ganzen Bereich t <= 10 praktisch exakt (2.5e-4), kostet aber ein
    Qubit und deutlich mehr Gatter.

    Die SVD-Dilatation nach Seneviratne et al.
    ------------------------------------------
    M = U Sigma V^dag; dilatiert wird NUR die Diagonale.  U und V^dag wirken
    auf ein Qubit weniger als die volle Sz.-Nagy-Dilatation, und eine Diagonale
    auf n Qubits kostet ~2^n statt 4^n.  sigma_j <= 1 ist garantiert, weil
    s >= ||H_m|| gilt.
    """
    t0 = time.time()
    gm = build_markov_grid(dt_fs, rho0=rho0, delta=delta, m=m, model=model,
                           n_sites=n_sites, verbose=verbose)

    Mpad = np.zeros((gm['mp'], gm['mp']), complex)
    Mpad[:gm['m'], :gm['m']] = gm['Hm'] / gm['s']
    Usvd, sig, Vh = np.linalg.svd(Mpad)
    sig = np.clip(sig, 0.0, 1.0)
    gm['svd'] = dict(U=Usvd, sigma=sig, Vh=Vh,
                     diag=np.concatenate([sig + 1j * np.sqrt(1 - sig ** 2),
                                          sig - 1j * np.sqrt(1 - sig ** 2)]))
    if verbose:
        print(f"  Singulaerwerte von H_m/s: {sig.min():.6f} bis {sig.max():.6f}")
        print(f"  Aufbau {time.time() - t0:.2f} s (einmalig, unabhaengig von "
              f"der Zahl der Zeitschritte)")
    return gm


def _prep_unitary(chi):
    """B unitaer mit B|0...0> = chi (QR mit Phasenkorrektur).

    Die Phasenkorrektur Q * dg/|dg| macht aus der QR-Zerlegung die eindeutige
    Variante mit reellem, positivem R-Diagonal.  Achtung: hat chi eine exakte
    Null, wird auch ein Diagonalelement von R exakt null und dg/|dg| gibt
    0/0 = nan -- die Matrix ist dann nicht mehr unitaer und UnitaryGate wirft
    ValueError.  Phase 1 ist dort die richtige Wahl, die Spalte ist ohnehin
    schon orthonormal.
    """
    n = chi.shape[0]
    M = np.eye(n, dtype=complex)
    M[:, 0] = chi
    Q, R = np.linalg.qr(M)
    dg = np.diag(R).copy()
    dg[np.abs(dg) < 1e-14] = 1.0
    return Q * (dg / np.abs(dg))


def readout_rows(grid, pairs=None):
    """Die Ablesezeilen: d Populationen, je Paar zwei weitere fuer Re und Im.

    vec rho_S = R y, also ist rho_jj = r_jj . y mit r_jj der Zeile j + d*j.
    Im komprimierten Krylov-Raum sind diese Zeilen DICHT -- anders als ohne
    Kompression, wo sie Basisvektoren waeren -- deshalb braucht jede ihre
    eigene Basisdrehung B^dag.
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
    verworfenen |1>-Zweig.  Die Durchgaengigkeit aendert sich nicht: das
    Systemregister wird nie gemessen und nie neu praepariert."""
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

    Sie steht senkrecht auf dem idealen Krylov-Vektor x_t = (H_m/s)^t y0, also
    ist |<r_null, x_t>|^2 = 0 ohne jede Naeherung.  Was die Maschine hier
    trotzdem zaehlt, IST der additive Boden -- direkt gemessen, nicht
    geschaetzt.  Bei rein depolarisierendem Rauschen gilt

        q_mess = (1 - p) q_wahr + p/mp,     q_null = p/mp,

    also q_korr = (q_mess - q_null) / (1 - mp*q_null).  Weil rho ~ sqrt(q)
    eingeht, schlaegt ein Boden von 0.066 bei q_wahr ~ 0 als Population 0.26
    durch -- deshalb lohnt der eine Extraschaltkreis je Zeitschritt.
    """
    A = grid['Hm'] / grid['s']
    x = grid['y0'].astype(complex).copy()
    for _ in range(t):
        x = A @ x
    n = np.linalg.norm(x)
    if n < 1e-14:                       # entartet -- irgendein Einheitsvektor
        r = np.zeros(len(x), complex)
        r[0] = 1.0
        return r
    x = x / n
    # Die Basisachse mit dem KLEINSTEN Ueberlapp nehmen: dann bleibt bei der
    # Projektion der Hauptanteil erhalten und es gibt keine Ausloeschung.
    j = int(np.argmin(np.abs(x)))
    e = np.zeros(len(x), complex)
    e[j] = 1.0
    r = e - np.vdot(x, e) * x           # Gram-Schmidt: <x, r> = 0 exakt
    return r / np.linalg.norm(r)


def hardware_circuits(grid, times, pairs=None, *, method='svd', defer=True,
                      floor=False):
    """Je Auslesezeit und Ablesezeile ein Schaltkreis.

    `defer=True` (Voreinstellung): AUFGESCHOBENE MESSUNG.  Statt eine Ancilla
    zu messen und zurueckzusetzen, bekommt jeder Zeitschritt ein eigenes
    frisches Ancilla-Qubit, und ALLE Messungen stehen am Ende.  Nach dem
    Prinzip der aufgeschobenen Messung ist das exakt aequivalent, solange kein
    Gatter von einem Messergebnis abhaengt -- und das tut hier keines.

    Warum: eine Mid-Circuit-Messung dauert Mikrosekunden, und waehrenddessen
    dephasieren die benachbarten Systemqubits; der Reset ist auf IBM-Hardware
    selbst wieder eine Messung mit bedingtem X.  Im ersten nicht-markovschen
    Hardware-Lauf (ibm_kingston, t = 1..10, MIT Reset) liess sich das Ergebnis
    mit den publizierten Fehlerraten NICHT erklaeren -- erst ein effektiver
    2Q-Fehler von 5 %, also das 16-Fache, traf die Daten.  Dieselbe Diagnose
    gilt hier, weil der Schaltkreis dieselbe Form hat.

    `defer=False` reproduziert die alte Variante mit Reset.

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
        t = int(t)
        rows_t = list(rows) + ([(('null', -1, -1), _null_row(grid, t))]
                               if floor else [])
        for tag, r in rows_t:
            nv = float(np.linalg.norm(r))
            chi = np.zeros(mp, complex)
            chi[:len(r)] = np.conj(r) / nv
            B = _prep_unitary(chi)

            q_sys = QuantumRegister(n_sys, 'kry')
            q_anc = QuantumRegister(t if defer else 1, 'anc')
            creg = ClassicalRegister(t + n_sys, 'c')   # erst t Ancilla-, dann n_sys Systembits
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

def report_cost(grid, jobs, *, shots=2048, backend=None, e_2q=2.66e-3,
                t_2q=68e-9, t_meas=2.18e-6, t_reset=2.21e-6, rep_delay=311e-6,
                coupling_map='linear', verbose=True):
    """Transpilieren, Gatter zaehlen, Treue und QPU-Zeit schaetzen.

    Ohne `backend` wird gegen die Heron-Basisgatter transpiliert.  Standard ist
    eine LINEARE Kette, denn das Heavy-Hex-Gitter von IBM hat keine Dreiecke --
    drei paarweise verbundene Qubits gibt es dort nicht, und die fehlenden
    Verbindungen kosten SWAPs.  `coupling_map=None` gibt die optimistische
    all-to-all-Schranke, `backend=...` die echte ISA.

    Die QPU-Zeit wird von `rep_delay` dominiert -- der Wartezeit zwischen den
    Shots.  Die Voreinstellungen sind an einem ECHTEN Lauf kalibriert, nicht
    geraten: Job da833pe0ukec7383ughg auf ibm_kingston (60 Kreise,
    defer=False, 2048 Shots) kostete laut job.usage() genau 42 s, wovon nur
    3.7 s auf Gatter entfielen.  Daraus folgt rep_delay = 311 us.  t_meas und
    t_reset stammen aus backend.target (2.18 bzw. 2.21 us).
    """
    if backend is not None:
        tq = transpile([j['circuit'] for j in jobs], backend=backend,
                       optimization_level=3)
    else:
        # Breite aus dem BREITESTEN Schaltkreis nehmen -- bei aufgeschobener
        # Messung braucht t = 10 dreizehn Qubits, nicht drei.
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
    # Bei aufgeschobener Messung gibt es KEINE Messung und KEINEN Reset mitten
    # im Schaltkreis -- alle Qubits werden am Ende gleichzeitig ausgelesen.
    defer = bool(jobs[0].get('defer', False))
    dur = (n2 * t_2q + t_meas if defer
           else ts * (t_meas + t_reset) + n2 * t_2q + t_meas)
    qpu = float(np.sum((dur + rep_delay) * shots))
    nq_max = max(j['transpiled'].num_qubits for j in jobs)

    if verbose:
        breite = (f"{grid['n_qubits']} Gitter- + bis "
                  f"{nq_max - grid['n_qubits']} Ancilla-Qubits = {nq_max}"
                  if defer else f"{grid['n_qubits']} Gitter-Qubits")
        print(f"  {len(jobs)} Schaltkreise, {breite}, "
              f"{shots} Shots je Schaltkreis, Dilatation "
              f"'{jobs[0].get('method', 'sznagy')}'")
        print(f"  Messung: "
              f"{'aufgeschoben -- 0 Resets, alles am Ende' if defer else 'mitten im Kreis, mit Reset'}")
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
    return dict(n_2q=n2, fidelity=fid, qpu_seconds=qpu, circuits=len(jobs))


# --------------------------------------------------------------------------
# 4.  kostenlos gegenpruefen
# --------------------------------------------------------------------------

def verify_in_aer(grid, jobs, *, shots=2048, pairs=None, verbose=True):
    """Dieselben Schaltkreise in Aer -- kostet nichts und findet Fehler frueh.

    Es laufen die TRANSPILIERTEN Kreise, sofern `report_cost` schon gelaufen
    ist; simuliert wird also genau das, was die Maschine ausfuehren wird, und
    nicht die abstrakte Vorstufe.
    """
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
    dd        Dynamical Decoupling (XY4) auf den Leerlaufzeiten.  Bei
              defer=True schuetzt es vor allem die wartenden Ancillas, die bis
              zu ihrem Schritt in |+> liegen.

    Beides kostet keine zusaetzliche QPU-Zeit -- Twirling verteilt die Shots
    nur auf mehrere zufaellige Varianten desselben Schaltkreises.
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
            # ihr H daher ganz an den Anfang.  Dann laege Ancilla 10 den GANZEN
            # Schaltkreis lang in |+>, und |+> dephasiert mit T2, waehrend |0>
            # nur T1 sieht.  ALAP schiebt jedes H direkt vor sein
            # Diagonal-Gatter und kostet kein einziges Gatter.
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

    Der ISA-Schaltkreis in `j['transpiled']` IST der ausgefuehrte -- die
    Runtime zerlegt ihn nicht noch einmal.  Twirling und Dynamical Decoupling
    fuegen zur Laufzeit nur Pauli-Gatter bzw. Pulse ein.

    Ansehen mit  jobs[i]['transpiled'].draw('mpl', fold=-1)
    Speichern mit qiskit.qpy.dump([j['transpiled'] for j in jobs], open('isa.qpy','wb'))
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


def run_on_backend(jobs, *, backend_name='ibm_kingston', shots=2048,
                   instance=None, twirling=True, dd=True, optimization_level=3,
                   verbose=True):
    """Absenden und auf das Ergebnis warten.  Braucht qiskit-ibm-runtime.

    Der Token wird einmalig hinterlegt mit

        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(token='...', overwrite=True)

    Es wird KEINE Datei hochgeladen: `generate_preset_pass_manager` transpiliert
    lokal gegen die echte ISA (Kopplungsgraph, Basisgatter, Kalibrierung), und
    nur die serialisierten Schaltkreise gehen raus.  Deshalb muss je Backend
    neu transpiliert werden -- das ist der eine Parameter `backend_name`.

    Backend-Wahl (Stand des nicht-markovschen Laufs):

        QPU             Typ        2Q layered   Auslesen   Queue
        ibm_kingston    Heron r2      3.05e-3   1.001e-2      41   <- beste Gatter
        ibm_fez         Heron r2      4.75e-3   9.888e-3     568
        ibm_marrakesh   Heron r2      5.16e-3   1.318e-2      20
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError as e:
        raise ImportError(
            "qiskit-ibm-runtime fehlt.  Installieren mit\n"
            "    pip install qiskit-ibm-runtime\n"
            f"(urspruenglicher Fehler: {e})")

    service = (QiskitRuntimeService(instance=instance) if instance
               else QiskitRuntimeService())
    backend = service.backend(backend_name)
    if verbose:
        print(f"  Backend {backend.name}: {backend.num_qubits} Qubits, "
              f"Basisgatter {backend.operation_names}")

    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=optimization_level)
    isa = pm.run([j['circuit'] for j in jobs])     # laeuft noch lokal
    for j, c in zip(jobs, isa):
        j['transpiled'] = c

    if verbose:
        isa_report(jobs)

    sampler = SamplerV2(mode=backend,
                        options=_sampler_options(twirling, dd, verbose))
    t0 = time.time()

    job = sampler.run(isa, shots=shots)            # ab hier kostet es QPU-Zeit

    if verbose:
        print(f"  Job {job.job_id()} abgesendet.", flush=True)
        print(f"  Falls die Anzeige unten abbricht, weiter mit:", flush=True)
        print(f"      hw.watch_job('{job.job_id()}')", flush=True)
        watch_job(job, verbose=True)

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

    `job` ist das Objekt aus `sampler.run(...)` oder eine Job-ID als
    Zeichenkette (dann wird sie beim Dienst nachgeschlagen).

    Haeufige Falle: `job.queue_info()` gibt es hier NICHT -- das ist die alte
    qiskit-ibm-provider-API, RuntimeJobV2 wirft AttributeError.  Die
    Warteschlange steht in `job.metrics()['position_in_queue']`, und das Feld
    darf None sein, solange IBM noch keine Schaetzung hat.

    Strg-C beendet nur die ANZEIGE, nicht den Job.
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
        print(f"  verbrauchte QPU-Zeit: {job.usage()} s")
    except Exception:
        pass
    return job


# --------------------------------------------------------------------------
# 6.  auswerten
# --------------------------------------------------------------------------

def analyze(grid, jobs, counts_list, *, pairs=None, verbose=True, quelle='QPU'):
    """Zaehlraten -> Dichtematrix, gegen qutips `mesolve`.

    rho_jj    = ||r_jj|| lambda_t sqrt(q_jj),  lambda_t = ||y0|| s^t sqrt(p_t)
    Re rho_ab = rho_++ - (rho_aa + rho_bb)/2,   Im analog ueber rho_RR

    Die Referenz ist hier `mesolve` mit DENSELBEN Sprungoperatoren, nicht der
    HEOMSolver -- verglichen wird das Gitter gegen die gewoehnliche
    Lindblad-Rechnung, also Verfahren gegen Verfahren bei gleicher Physik.
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
        # Skala ein, und ein zu kleines p_t (Auslesefehler der Ancillas) zieht
        # ALLE Eintraege gemeinsam nach unten.  Die Spurnormierung kuerzt
        # diesen gemeinsamen Faktor exakt heraus.
        tr = np.real(np.trace(M))
        if tr > 0:
            rho_tr[k] = M / tr

    ts = np.array(times)
    ref = qutip_reference_rho(np.arange(max(times) + 1) * grid['dt_fs'],
                              rho0=grid.get('rho0', RHO0),
                              **grid.get('model', MODEL))
    ok = np.isfinite(np.real(rho[:, 0, 0]))
    if verbose and ok.any():
        print(f"\n  {quelle} gegen qutip mesolve (Lindblad):")
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
