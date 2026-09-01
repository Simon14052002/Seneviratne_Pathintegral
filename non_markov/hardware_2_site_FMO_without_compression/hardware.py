"""4-Site-FMO OHNE Kompression -- Schaltkreise fuer echte IBM-Hardware.

Dieselbe Aufgabe wie `../hardware_4_site_FMO/hardware.py`, aber OHNE Arnoldi.
Der Zustandsvektor ist der volle HEOM-Vektor (System + alle ADOs), nicht seine
Projektion auf einen Krylov-Unterraum.  Damit verschwindet der
Kompressionsfehler vollstaendig -- und mit ihm die Ausfuehrbarkeit.

    WARNUNG, vorab und in Zahlen gemessen (ibm_kingston, Heron r2):

        Sites  Tiefe  Qubits   CZ je Schritt   Treue nach EINEM Schritt
          2      1       6            908              0.0624
          2      2       7          3,692              0.0000
          3      1       7          3,692              0.0000
          4      2      11       ~524,000              0.0000

        zum Vergleich MIT Kompression (m=4, 4 Sites, 3 Qubits):
                              138 CZ fuer ALLE 10 Schritte, Treue 0.656

    Der kleinste sinnvolle unkomprimierte Fall braucht fuer EINEN Zeitschritt
    6.6-mal mehr Zweiqubit-Gatter als der komprimierte Lauf fuer zehn.  Auf
    heutiger Hardware ist damit KEIN einziger Zeitschritt erreichbar.  Dieses
    Modul erzeugt trotzdem die vollstaendigen, hardwaretauglichen Schaltkreise
    -- es rechnet die Kosten ehrlich aus, laeuft in Aer korrekt durch, und
    `run_on_backend` funktioniert, sobald Maschinen gut genug sind.

Warum es so teuer ist: der Propagator P = exp(A dt) ist eine DICHTE
n x n-Matrix auf dem ganzen ADO-Raum.  Die Zahl der ADOs waechst wie
C(n_exp + Tiefe, Tiefe) mit n_exp = n_sites (Nk+1), und der Vektor hat
n = d^2 * ADOs Eintraege.  Eine allgemeine q-Qubit-Unitaere kostet ~4^q/4
Zweiqubit-Gatter -- der Aufwand waechst also exponentiell in der Qubitzahl und
diese logarithmisch in der ADO-Zahl, unterm Strich also LINEAR in n^2.

    n_sites=2, Tiefe=1:   5 ADOs,  n =  20 -> 32 gepolstert -> 6 Qubits
    n_sites=2, Tiefe=2:  15 ADOs,  n =  60 -> 64            -> 7 Qubits
    n_sites=4, Tiefe=1:   9 ADOs,  n = 144 -> 256           -> 9 Qubits
    n_sites=4, Tiefe=2:  45 ADOs,  n = 720 -> 1024          -> 11 Qubits

Anders als im komprimierten Fall haengt die Qubitzahl hier SEHR WOHL an der
Systemgroesse.  Der Schritt von 4 auf 2 Sites spart 4 Qubits und damit den
Faktor 4^4 = 256 an Gattern.  (Mit Kompression war das anders: dort bestimmt
allein die Krylov-Dimension m die Registerbreite, und 4 -> 2 Sites brachte
gar nichts.)

Ein zweiter Unterschied, diesmal zum Vorteil: ohne Kompression sind die
Ablesezeilen fuer die Populationen die SKALIERTEN EINHEITSVEKTOREN.  Ein
einziger Schaltkreis, dessen Register am Ende in der Rechenbasis gemessen
wird, liefert deshalb ALLE d Populationen gleichzeitig -- das Histogramm ueber
das Systemregister ist bereits die Antwort.  Nur die Kohaerenzen brauchen noch
je eine Basisdrehung.  Statt d + 2*Paare Schaltkreisen je Zeitschritt sind es
also 1 + 2*Paare.

Konventionen (identisch zum komprimierten Modul):
  * Klassisches Register: Bits 0..t-1 die Ancilla-Aufzeichnung, danach das
    Systemregister.  So gibt es keine Zweideutigkeit in Qiskits Reihenfolge.
  * Nachselektion auf alle Ancillabits = 0.
  * `defer=True`: ein frisches Ancilla-Qubit je Zeitschritt, alle Messungen am
    Ende (Prinzip der aufgeschobenen Messung).  Auf ibm_kingston hat das im
    komprimierten Fall den RMS gegen qutip von 0.13 auf 0.043 gedrueckt, weil
    die Mid-Circuit-Messung die Nachbarqubits dephasiert.
  * Verglichen wird ausschliesslich gegen qutips HEOMSolver.
"""

import time

import numpy as np
from scipy.linalg import expm
from qiskit import (QuantumCircuit, QuantumRegister, ClassicalRegister,
                    transpile)
from qiskit.circuit.library import UnitaryGate, DiagonalGate
from qiskit_aer import AerSimulator

from heom_gauge import (heom_generator, dilate, safe_norm2,
                        qutip_reference, qutip_reference_rho)

__all__ = ['MODEL', 'RHO0', 'H_FMO_FULL', 'make_model', 'build_hardware_grid',
           'readout_rows', 'hardware_circuits', 'report_cost', 'verify_in_aer',
           'analyze', 'isa_report', 'run_on_backend', 'watch_job',
           'qutip_reference', 'qutip_reference_rho']

_C_CM = 2.99792458e10
FS_TO_CM = 1e-15 * 2 * np.pi * _C_CM
KB_CM = 0.6950348004

# Read et al., Biophys. J. 95, 847 (2008); Pfad 1 -> 2 -> 3 -> 4 des FMO.
H_FMO_FULL = np.array([[12375.0, -87.7,   5.5,  -5.9],
                       [-87.7, 12495.0,  30.8,   8.2],
                       [5.5,      30.8, 12175.0, -53.4],
                       [-5.9,      8.2, -53.4, 12285.0]])

HERON_BASIS = ['cz', 'rz', 'sx', 'x', 'measure', 'reset']


def make_model(n_sites=4, *, lam=35.0, gamma=50.0, T_K=300.0, H=None):
    """Modell und Anfangszustand fuer n_sites Sites.

    Ohne eigenes `H` wird der fuehrende n_sites x n_sites-Block des
    FMO-Hamiltonians genommen und spurlos gemacht.  Die Anregung startet auf
    Site 1, also rho_S(0) = |1><1| -- damit ist x~0 = e_0 und das Register
    startet in |0...0>, ohne Vorbereitungsschaltkreis.

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
MODEL, RHO0 = make_model(N_SITES)


def _vec_index(d, i, j):
    """Spaltenweise vec-Konvention: vec(rho)[i + d*j] = rho[i, j]."""
    return i + d * j


# --------------------------------------------------------------------------
# 1.  das Gitter
# --------------------------------------------------------------------------

def build_hardware_grid(*, n_sites=4, dt_fs=20.0, depth=2, Nk=1,
                        scale_ados=True, model=None, rho0=None, verbose=True):
    """Ein Gatter fuer einen Zeitschritt -- ohne jede Kompression.

    Schritte:
      1. Erzeuger A des HEOM auf System + ADOs, mit diagonaler ADO-Skalierung
         nach Shi et al.  Diese Skalierung ist der Grund, warum hier KEINE
         Lyapunov-Eichung noetig ist: sie drueckt ||P||_2 von ~10.7 auf
         ~1.0007, und nur dann ist die Dilatation ueberhaupt sinnvoll.
      2. P = exp(A dt) -- der exakte Propagator ueber einen Zeitschritt.  Weil
         P^t genau die Loesung nach t Schritten ist, gibt es keinen
         Zeitschrittfehler; der einzige Fehler ist die HEOM-Abschneidung
         selbst (Tiefe, Nk), die auch qutip hat.
      3. auf eine Zweierpotenz auffuellen und Sz.-Nagy-dilatieren.
      4. zusaetzlich die SVD P/s = U Sigma V^dag bereitlegen.  Auf Hardware
         ist die SVD-Route billiger, weil nur die DIAGONALE dilatiert werden
         muss statt der ganzen Matrix (Seneviratne et al., ACS Omega 9, 9666
         (2024): gemessen 2.6-mal weniger CZ).

    Returns ein dict mit U, s, n, np_, n_qubits, scale, d, D, dt_fs, depth,
    Nk, model, rho0, svd.
    """
    model = MODEL if model is None else model
    rho0 = RHO0 if rho0 is None else rho0
    d = model['H'].shape[0]
    D = d * d
    dt_cm = dt_fs * model['FS_TO_CM']
    t_all = time.time()

    A, ginfo = heom_generator(depth=depth, Nk=Nk, scale_ados=scale_ados,
                              **model)
    n = ginfo['n']
    scale = ginfo['ado_scale'] if scale_ados else np.ones(n)
    if verbose:
        print(f"  [1/4] Erzeuger A: {n} x {n} ({ginfo['n_ados']} ADOs, "
              f"Tiefe {depth}, Nk={Nk})", flush=True)

    P = expm(A * dt_cm)
    normP = safe_norm2(P)
    if verbose:
        print(f"  [2/4] P = expm(A dt): ||P||_2 = {normP:.8f}", flush=True)

    np_ = 2 ** int(np.ceil(np.log2(n)))
    Ppad = np.zeros((np_, np_), complex)
    Ppad[:n, :n] = P                    # der Rest bleibt null, wird nie besetzt
    U, s = dilate(Ppad)
    n_qubits = int(np.log2(np_)) + 1
    if verbose:
        print(f"  [3/4] Dilatation: {n} -> {np_} gepolstert, U ist "
              f"{2*np_} x {2*np_} = {n_qubits} Qubits, s = {s:.8f}", flush=True)

    M = Ppad / s                        # Singulaerwerte <= 1
    Um, sig, Vh = np.linalg.svd(M)
    sig = np.clip(sig, 0.0, 1.0)
    diag = np.concatenate([sig + 1j * np.sqrt(1 - sig ** 2),
                           sig - 1j * np.sqrt(1 - sig ** 2)])
    if verbose:
        print(f"  [4/4] SVD: Singulaerwerte {sig.min():.6f} bis {sig.max():.6f}"
              f"   Aufbau {time.time() - t_all:.1f} s", flush=True)

    # x~0 = e_0 setzt voraus, dass die Skalierung den Systemblock unveraendert
    # laesst -- sonst startet das Register nicht in |0...0>.
    assert np.allclose(scale[:D], 1.0), \
        "ADO-Skalierung veraendert den Systemblock; x~0 = e_0 gilt dann nicht"

    return dict(U=U, s=s, n=n, np_=np_, n_qubits=n_qubits, scale=scale,
                d=d, D=D, dt_fs=dt_fs, depth=depth, Nk=Nk,
                model=model, rho0=rho0,
                svd=dict(U=Um, Vh=Vh, sigma=sig, diag=diag))


def readout_rows(grid, pairs=None):
    """Die Ablesezeilen im VOLLEN Raum.

    vec rho_S = R x mit R = diag(scale) auf den ersten D Eintraegen, also ist
    rho_ij = scale[i + d*j] * x[i + d*j].  Die Zeile ist damit ein SKALIERTER
    EINHEITSVEKTOR -- und genau das ist der Vorteil gegenueber der Kompression,
    wo dieselbe Zeile dicht ist und eine eigene Basisdrehung braucht.

    Fuer die Populationen wird diese Funktion daher gar nicht gebraucht: misst
    man das Register in der Rechenbasis, ist das Histogramm bereits |x_k|^2 fuer
    alle k gleichzeitig.  Nur die Kohaerenzkanaele ++ und RR mischen vier
    Eintraege und brauchen deshalb eine Drehung.
    """
    d, npd, scale = grid['d'], grid['np_'], grid['scale']

    def row(i, j):
        k = _vec_index(d, i, j)
        r = np.zeros(npd, complex)
        r[k] = scale[k]
        return r

    out = [(('pop', j, j), row(j, j)) for j in range(d)]
    for (a, b) in (pairs or []):
        out.append((('pp', a, b),
                    0.5 * (row(a, a) + row(b, b) + row(a, b) + row(b, a))))
        out.append((('RR', a, b),
                    0.5 * (row(a, a) + row(b, b) - 1j * row(a, b)
                           + 1j * row(b, a))))
    return out


def _prep_unitary(chi):
    """B unitaer mit B|0...0> = chi (QR mit Phasenkorrektur).

    Hat chi eine exakte Null, wird ein Diagonalelement von R null und
    dg/|dg| gaebe 0/0 = nan -- die Matrix waere nicht mehr unitaer.  Beim
    unkomprimierten Gitter ist chi fast ueberall null, dieser Fall ist hier
    also der Normalfall und nicht die Ausnahme.
    """
    n = chi.shape[0]
    M = np.eye(n, dtype=complex)
    M[:, 0] = chi
    Q, R = np.linalg.qr(M)
    dg = np.diag(R).copy()
    dg[np.abs(dg) < 1e-14] = 1.0
    return Q * (dg / np.abs(dg))


# --------------------------------------------------------------------------
# 2.  Schaltkreise
# --------------------------------------------------------------------------

def _step(qc, q_sys, q_anc, grid, method):
    """Ein Zeitschritt.  `q_anc` ist eine Folge mit genau einem Qubit."""
    if method == 'sznagy':
        qc.append(UnitaryGate(grid['U'], label='U'), list(q_sys) + list(q_anc))
        return
    sv = grid['svd']
    qc.append(UnitaryGate(sv['Vh'], label='V+'), q_sys[:])
    qc.h(q_anc[0])
    qc.append(DiagonalGate(list(sv['diag'])), list(q_sys) + list(q_anc))
    qc.h(q_anc[0])
    qc.append(UnitaryGate(sv['U'], label='U'), q_sys[:])


def hardware_circuits(grid, times, pairs=None, *, method='svd', defer=True):
    """Je Auslesezeit EIN Populationskreis und je Paar zwei Kohaerenzkreise.

    Der Unterschied zum komprimierten Modul: dort braucht jede der d
    Populationen einen eigenen Schaltkreis, weil ihre Ablesezeile im
    Krylov-Raum dicht ist.  Hier ist sie ein skalierter Einheitsvektor, also
    genuegt EIN Kreis ohne Drehung -- das Histogramm ueber das Register
    liefert alle d Populationen auf einmal.

    Kreise je Zeitschritt:  1 + 2 * len(pairs)   statt   d + 2 * len(pairs)

    `defer=True`: ein frisches Ancilla-Qubit je Zeitschritt, alle Messungen am
    Ende.  Aequivalent nach dem Prinzip der aufgeschobenen Messung, solange
    kein Gatter von einem Messergebnis abhaengt -- und das tut hier keines.
    Auf ibm_kingston war das im komprimierten Fall der Unterschied zwischen
    RMS 0.13 und 0.043, weil eine Mid-Circuit-Messung 2.18 us dauert und die
    Nachbarqubits solange dephasieren.
    """
    npd = grid['np_']
    n_sys = int(np.log2(npd))
    pairs = list(pairs or [])
    rows = {tag: r for tag, r in readout_rows(grid, pairs)}

    jobs = []
    for t in times:
        kanaele = [(('pop', -1, -1), None)]          # ohne Drehung
        for (a, b) in pairs:
            for tag in ('pp', 'RR'):
                kanaele.append(((tag, a, b), rows[(tag, a, b)]))

        for tag, r in kanaele:
            if r is None:
                B, nv = None, 1.0
            else:
                nv = float(np.linalg.norm(r))
                chi = np.zeros(npd, complex)
                chi[:len(r)] = np.conj(r) / nv
                B = _prep_unitary(chi)

            q_sys = QuantumRegister(n_sys, 'heom')
            q_anc = QuantumRegister(t if defer else 1, 'anc')
            creg = ClassicalRegister(t + n_sys, 'c')
            qc = QuantumCircuit(q_sys, q_anc, creg)
            # x~0 = e_0 -> das Register startet schon richtig
            for k in range(1, t + 1):
                anc = [q_anc[k - 1]] if defer else [q_anc[0]]
                _step(qc, q_sys, anc, grid, method)
                if not defer:
                    qc.measure(anc[0], creg[k - 1])
                    qc.reset(anc[0])
            if defer:
                for k in range(t):
                    qc.measure(q_anc[k], creg[k])

            if B is not None:
                qc.append(UnitaryGate(B.conj().T, label='B+'), q_sys[:])
            for i in range(n_sys):
                qc.measure(q_sys[i], creg[t + i])

            jobs.append(dict(t=t, tag=tag, norm=nv, n_sys=n_sys, defer=defer,
                             method=method, circuit=qc))
    return jobs


def _split_counts(counts, n_steps, n_sys):
    """(Gesamt, akzeptiert, Histogramm ueber das Systemregister).

    Anders als im komprimierten Modul wird hier das ganze HISTOGRAMM
    zurueckgegeben, nicht nur die Treffer auf |0...0> -- denn ein einziger
    Populationskreis traegt alle d Populationen gleichzeitig.
    """
    total = acc = 0
    hist = {}
    for bits, c in counts.items():
        r = bits.replace(' ', '')[::-1]
        total += c
        if set(r[:n_steps]) - {'0'}:            # Nachselektion
            continue
        acc += c
        idx = sum((1 << i) for i in range(n_sys) if r[n_steps + i] == '1')
        hist[idx] = hist.get(idx, 0) + c
    return total, acc, hist


# --------------------------------------------------------------------------
# 3.  was der Lauf kostet
# --------------------------------------------------------------------------

def report_cost(grid, jobs, *, shots=4096, backend=None, e_2q=3.05e-3,
                t_2q=68e-9, t_meas=2.18e-6, t_reset=2.21e-6, rep_delay=311e-6,
                coupling_map='linear', verbose=True, max_qubits_synth=8):
    """Transpilieren, Gatter zaehlen, Treue und QPU-Zeit schaetzen.

    `rep_delay`, `t_meas` und `t_reset` sind an einem ECHTEN Lauf kalibriert
    (Job da833pe0ukec7383ughg auf ibm_kingston: 60 Kreise, 2048 Shots,
    job.usage() = 42 s, davon 3.7 s Gatterzeit).

    ACHTUNG: das Transpilieren einer allgemeinen q-Qubit-Unitaeren ist selbst
    exponentiell teuer.  Ab etwa 9 Qubits dauert die Synthese Minuten bis
    Stunden und braucht viele GB.  Oberhalb von `max_qubits_synth` wird
    deshalb NICHT transpiliert, sondern die bekannte Schranke
    ~2 * 4^(q-1) / 4 CZ je Schritt eingesetzt (zwei allgemeine
    (q-1)-Qubit-Unitaere plus eine Diagonale).  Die Schaetzung ist dann eine
    UNTERGRENZE -- der Transpiler erreicht sie in der Praxis nicht.
    """
    n_sys = jobs[0]['n_sys']
    q = n_sys + 1
    geschaetzt = q > max_qubits_synth

    if geschaetzt:
        cz_schritt = 2 * (4 ** n_sys) // 4
        for j in jobs:
            j['transpiled'] = None
            j['n_2q'] = cz_schritt * j['t']
            j['depth'] = None
    else:
        if backend is not None:
            tq = transpile([j['circuit'] for j in jobs], backend=backend,
                           optimization_level=3)
        else:
            nq = max(j['circuit'].num_qubits for j in jobs)
            cm = ([[i, i + 1] for i in range(nq - 1)]
                  + [[i + 1, i] for i in range(nq - 1)]) \
                if coupling_map == 'linear' else coupling_map
            tq = transpile([j['circuit'] for j in jobs],
                           basis_gates=HERON_BASIS, coupling_map=cm,
                           optimization_level=3, seed_transpiler=7)
        for j, c in zip(jobs, tq):
            j['transpiled'] = c
            ops = c.count_ops()
            j['n_2q'] = ops.get('cz', 0) + ops.get('cx', 0) + ops.get('ecr', 0)
            j['depth'] = c.depth()

    n2 = np.array([j['n_2q'] for j in jobs])
    ts = np.array([j['t'] for j in jobs])
    fid = (1 - e_2q) ** n2
    defer = bool(jobs[0].get('defer', False))
    dur = (n2 * t_2q + t_meas if defer
           else ts * (t_meas + t_reset) + n2 * t_2q + t_meas)
    qpu = float(np.sum((dur + rep_delay) * shots))

    if verbose:
        breite = (f"{n_sys} System- + bis {max(ts)} Ancilla-Qubits = "
                  f"{n_sys + max(ts)}" if defer
                  else f"{n_sys} System- + 1 Ancilla-Qubit")
        print(f"  {len(jobs)} Schaltkreise, {breite}, {shots} Shots je "
              f"Schaltkreis, Dilatation '{jobs[0].get('method', 'svd')}'")
        print(f"  Messung: "
              f"{'aufgeschoben -- 0 Resets' if defer else 'mitten im Kreis'}")
        if geschaetzt:
            print(f"  !! {q} Qubits -- NICHT transpiliert (Synthese waere "
                  f"exponentiell teuer).")
            print(f"     Eingesetzt: 2*4^{n_sys}/4 = {cz_schritt:,d} CZ je "
                  f"Schritt als UNTERGRENZE.")
        print(f"\n  {'t':>3s} {'2Q-Gatter':>12s} {'Treue (Gatter)':>15s}")
        for t in sorted(set(ts)):
            sel = ts == t
            print(f"  {t:3d} {int(np.median(n2[sel])):12,d} "
                  f"{np.median(fid[sel]):15.4f}")
        print(f"\n  geschaetzte QPU-Zeit : {qpu:.1f} s  ({qpu/60:.2f} min)")
        print(f"  Annahmen: {e_2q:.2e} je 2Q, {t_2q*1e9:.0f} ns je 2Q, "
              f"{rep_delay*1e6:.0f} us rep_delay")
        print(f"  (kalibriert an Job da833pe0ukec7383ughg: 42 s tatsaechlich)")

        beste = fid.max()
        print(f"\n  URTEIL: beste Gattertreue im Lauf {beste:.4f}.  "
              f"{'Unbrauchbar' if beste < 0.3 else 'Grenzwertig' if beste < 0.6 else 'Machbar'}"
              f" -- zum Vergleich erreichte der komprimierte 10-Schritt-Lauf "
              f"0.656 bei RMS 0.043 gegen qutip.")
    return dict(n_2q=n2, fidelity=fid, qpu_seconds=qpu, circuits=len(jobs),
                geschaetzt=geschaetzt)


# --------------------------------------------------------------------------
# 4.  kostenlos gegenpruefen
# --------------------------------------------------------------------------

def verify_in_aer(grid, jobs, *, shots=4096, pairs=None, verbose=True):
    """Dieselben Schaltkreise rauschfrei in Aer -- misst NUR den Algorithmus.

    Weil hier nicht komprimiert wird, ist der einzige verbleibende Fehler die
    Statistik der Shots plus die HEOM-Abschneidung, die auch qutip hat.  Die
    Abweichung sollte also mit 1/sqrt(shots) skalieren und sonst nichts.
    """
    sim = AerSimulator(seed_simulator=7)
    counts = []
    for i, j in enumerate(jobs):
        c = transpile(j['circuit'], sim, optimization_level=0)
        r = sim.run(c, shots=shots).result().get_counts()
        counts.append({k.replace(' ', ''): v for k, v in r.items()})
        if verbose and (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(jobs)} simuliert", flush=True)
    return analyze(grid, jobs, counts, pairs=pairs, verbose=verbose,
                   quelle='Aer')


# --------------------------------------------------------------------------
# 5.  auswerten
# --------------------------------------------------------------------------

def analyze(grid, jobs, counts_list, *, pairs=None, verbose=True, quelle='QPU'):
    """Zaehlraten -> Dichtematrix, gegen den exakten qutip-HEOMSolver.

    Populationen kommen aus EINEM Kreis:

        p_t     = akzeptiert / gesamt
        lambda_t = s^t sqrt(p_t)                     (||x~0|| = 1)
        rho_jj  = scale[j + d*j] * lambda_t * sqrt(hist[j + d*j] / akzeptiert)

    Kohaerenzen wie im komprimierten Modul ueber die ++/RR-Drehung:

        Re rho_ab = rho_++ - (rho_aa + rho_bb)/2,    Im analog ueber rho_RR
    """
    d, s = grid['d'], grid['s']
    scale = grid['scale']
    times = sorted(set(j['t'] for j in jobs))
    pairs = list(pairs or [])
    idx = {(j['t'], j['tag']): i for i, j in enumerate(jobs)}

    rho = np.full((len(times), d, d), np.nan, complex)
    rho_tr = np.full((len(times), d, d), np.nan, complex)
    err_re = np.full((len(times), d, d), np.nan)
    err_im = np.full((len(times), d, d), np.nan)
    prob = np.full(len(times), np.nan)

    for k, t in enumerate(times):
        i0 = idx.get((t, ('pop', -1, -1)))
        if i0 is None:
            continue
        tot, acc, hist = _split_counts(counts_list[i0], t, jobs[i0]['n_sys'])
        if not acc:
            continue
        prob[k] = acc / tot
        lam = s ** t * np.sqrt(prob[k])

        pop = np.zeros(d)
        dpop = np.zeros(d)
        for j2 in range(d):
            kk = _vec_index(d, j2, j2)
            pop[j2] = scale[kk] * lam * np.sqrt(hist.get(kk, 0) / acc)
            dpop[j2] = scale[kk] * lam / (2 * np.sqrt(acc))
        M = np.diag(pop).astype(complex)
        E, Eim = np.diag(dpop), np.zeros((d, d))

        for (a, b) in pairs:
            got = {}
            for tag in ('pp', 'RR'):
                i = idx.get((t, (tag, a, b)))
                if i is None:
                    continue
                to2, ac2, h2 = _split_counts(counts_list[i], t,
                                             jobs[i]['n_sys'])
                nv = jobs[i]['norm']
                q = h2.get(0, 0) / ac2 if ac2 else np.nan
                got[tag] = (nv * lam * np.sqrt(max(q, 0.0)),
                            nv * lam / (2 * np.sqrt(ac2)) if ac2 else np.nan)
            if len(got) < 2:
                continue
            half = 0.5 * (pop[a] + pop[b])
            quart = 0.25 * dpop[a] ** 2 + 0.25 * dpop[b] ** 2
            M[a, b] = (got['pp'][0] - half) + 1j * (got['RR'][0] - half)
            M[b, a] = np.conj(M[a, b])
            E[a, b] = E[b, a] = np.sqrt(got['pp'][1] ** 2 + quart)
            Eim[a, b] = Eim[b, a] = np.sqrt(got['RR'][1] ** 2 + quart)

        rho[k], err_re[k], err_im[k] = M, E, Eim
        # Zweite, unabhaengige Ablesung ueber Tr rho_S = 1.  Sie braucht p_t
        # nicht -- und darin steckt auf Hardware der groesste systematische
        # Fehler, weil p_t als gemeinsamer Faktor in alle Eintraege eingeht.
        tr = np.real(np.trace(M))
        if tr > 0:
            rho_tr[k] = M / tr

    ts = np.array(times)
    ref = qutip_reference_rho(np.arange(max(times) + 1) * grid['dt_fs'],
                              rho0=grid['rho0'], depth=grid['depth'],
                              Nk=grid['Nk'], **grid['model'])
    ok = np.isfinite(np.real(rho[:, 0, 0]))
    if verbose and ok.any():
        print(f"\n  {quelle} gegen qutip HEOMSolver:")
        for k in np.where(ok)[0]:
            t = ts[k]
            dp = np.abs(np.real(np.diag(rho[k]))
                        - np.real(np.diag(ref[t]))).max()
            dt_ = np.abs(np.real(np.diag(rho_tr[k]))
                         - np.real(np.diag(ref[t]))).max()
            print(f"    t = {t*grid['dt_fs']:5.0f} fs | p_succ {prob[k]:.4f} | "
                  f"Spur {np.real(np.trace(rho[k])):.4f} | max|dPop| "
                  f"ueber p_t {dp:.6f} | spurnormiert {dt_:.6f}")
    return rho, err_re, dict(t_index=ts, p_success=prob, rho_err_im=err_im,
                             rho_tr=rho_tr, reference=ref, pairs=pairs,
                             quelle=quelle)


# --------------------------------------------------------------------------
# 6.  auf die echte Maschine
# --------------------------------------------------------------------------

def _sampler_options(twirling=True, dd=True, verbose=True):
    """Fehlerunterdrueckung fuer SamplerV2, defensiv aufgebaut.

    twirling  wandelt kohaerente Fehler in stochastische -- der richtige Hebel
              fuer einen Kreis, der DASSELBE Gatter t-mal wiederholt.
    dd        XY4 auf den Leerlaufzeiten, mit ALAP-Scheduling: die t Ancillas
              haben keinen Vorgaenger, Qiskit zoege ihr H sonst ganz an den
              Anfang, und sie laegen den ganzen Kreis in |+> (T2, nicht T1).
    """
    from qiskit_ibm_runtime.options import SamplerOptions
    o = SamplerOptions()
    try:
        o.twirling.enable_gates = bool(twirling)
        o.twirling.enable_measure = bool(twirling)
        o.dynamical_decoupling.enable = bool(dd)
        if dd:
            o.dynamical_decoupling.sequence_type = 'XY4'
            o.dynamical_decoupling.scheduling_method = 'alap'
    except Exception as e:
        if verbose:
            print(f"  WARNUNG: Mitigation nicht gesetzt ({e})")
        return SamplerOptions()
    if verbose:
        print(f"  Mitigation: Twirling {'an' if twirling else 'aus'}, "
              f"DD XY4/alap {'an' if dd else 'aus'}")
    return o


def isa_report(jobs, verbose=True):
    """Was tatsaechlich auf der Maschine laeuft."""
    tq = [j.get('transpiled') for j in jobs]
    if any(c is None for c in tq):
        raise RuntimeError("noch nicht transpiliert -- erst report_cost(...) "
                           "oder run_on_backend(...) aufrufen")
    if verbose:
        print(f"    {'t':>3s} {'Kanal':>14s} {'2Q':>8s} {'Tiefe':>7s} "
              f"{'Qubits':>7s}")
        for j, c in zip(jobs, tq):
            o = c.count_ops()
            n2 = o.get('cz', 0) + o.get('cx', 0) + o.get('ecr', 0)
            print(f"    {j['t']:3d} {str(j['tag']):>14s} {n2:8,d} "
                  f"{c.depth():7d} {c.num_qubits:7d}")
    return tq


def watch_job(job, *, interval=15, service=None, verbose=True):
    """Live-Anzeige: wie viele Jobs stehen noch vor meinem?

    ACHTUNG: `job.queue_info()` gibt es in qiskit-ibm-runtime NICHT -- das ist
    die alte qiskit-ibm-provider-API.  Die Warteschlange steht in
    `job.metrics()['position_in_queue']`, und das Feld darf None sein.

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
            try:
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
            if verbose and zeile != letzte:
                print(zeile, flush=True)
                letzte = zeile
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n  Anzeige abgebrochen -- Job {job.job_id()} LAEUFT WEITER.")
        return job

    print(f"  [{time.time() - t0:5.0f} s] Endzustand: {job.status()}")
    try:
        print(f"  verbrauchte QPU-Zeit: {job.usage()} s")
    except Exception:
        pass
    return job


def run_on_backend(jobs, *, backend_name='ibm_kingston', shots=4096,
                   instance=None, twirling=True, dd=True,
                   optimization_level=3, verbose=True, force=False):
    """Absenden und auf das Ergebnis warten.

    Weigert sich standardmaessig, wenn die erwartete Gattertreue unter 1 %
    liegt -- das waere reines Verbrennen von QPU-Zeit.  Mit `force=True`
    trotzdem absenden.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = (QiskitRuntimeService(instance=instance) if instance
               else QiskitRuntimeService())
    backend = service.backend(backend_name)
    if verbose:
        print(f"  Backend {backend.name}: {backend.num_qubits} Qubits")

    pm = generate_preset_pass_manager(backend=backend,
                                      optimization_level=optimization_level)
    if verbose:
        print(f"  transpiliere {len(jobs)} Kreise -- bei vielen Qubits kann "
              f"die Synthese SEHR lange dauern.", flush=True)
    isa = pm.run([j['circuit'] for j in jobs])
    for j, c in zip(jobs, isa):
        j['transpiled'] = c

    n2 = max((c.count_ops().get('cz', 0) for c in isa), default=0)
    treue = (1 - 3.05e-3) ** n2
    if verbose:
        isa_report(jobs)
        print(f"  tiefster Kreis: {n2:,d} CZ -> erwartete Treue {treue:.2e}")
    if treue < 0.01 and not force:
        raise RuntimeError(
            f"Abbruch: erwartete Gattertreue {treue:.2e} beim tiefsten Kreis "
            f"({n2:,d} CZ).  Das Ergebnis waere reines Rauschen und wuerde nur "
            f"QPU-Zeit verbrennen.  Mit force=True trotzdem absenden.")

    sampler = SamplerV2(mode=backend,
                        options=_sampler_options(twirling, dd, verbose))
    job = sampler.run(isa, shots=shots)
    if verbose:
        print(f"  Job {job.job_id()} abgesendet.")
        print(f"  Weiter mit:  hw.watch_job('{job.job_id()}')")
        watch_job(job, verbose=True)

    result = job.result()
    out = []
    for i, j in enumerate(jobs):
        creg = j['transpiled'].cregs[0].name
        out.append(getattr(result[i].data, creg).get_counts())
    return out
