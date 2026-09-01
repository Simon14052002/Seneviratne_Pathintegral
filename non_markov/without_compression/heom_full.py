"""
HEOM auf einem durchgaengigen Gitter -- OHNE Krylov-Kompression.

Der Unterschied zur komprimierten Variante in ../../gauge_circuit.py ist genau
ein weggelassener Schritt, und es ist der einzige, der ueberhaupt etwas
Trajektorienartiges klassisch gerechnet hat:

    komprimiert:  A -> W-Eichung -> P~ -> ARNOLDI(m Matvecs) -> H_m -> dilate
    hier:         A -> D-Skalierung -> P                     -------> dilate

Arnoldi braucht m Anwendungen von P~ auf einen Vektor.  Das ist zwar keine
Zeitpropagation im physikalischen Sinn, aber es ist m-mal "die Dynamik
anfassen".  Ohne Kompression faellt das weg: der Schaltkreis arbeitet direkt im
vollen HEOM-Raum, und klassisch bleibt nur noch das Bauen des einen Gatters.

Warum die ADO-Skalierung
------------------------
Die Dilatation U = [[P/s, B], [C, -(P/s)^dag]] erzwingt s >= ||P||_2, und die
Post-Selektion zahlt s^(-2t).  Ungeskaliert ist ||P||_2 = 10.66 bei dt = 10 fs,
also p(100) ~ 1e-206 -- unbrauchbar.  Die diagonale ADO-Skalierung (Shi et al.,
rho_n -> rho_n / sqrt(prod n_k! |c_k|^n_k)) druckt sie auf 1.00068 und damit
p(100) >= 0.87.  Sie ist eine exakte Aehnlichkeitstransformation, eintragsweise
berechenbar und O(n) -- der einzige Eichungsschritt, der fuer ein grosses System
ueberhaupt eine Perspektive hat.  Details in kritik.ipynb.

Feste Annahmen dieses Moduls
----------------------------
    rho_S(0) = |1><1|   Anregung auf Site 1.  Damit ist vec(rho_0) = e_0 und,
                        weil der nullte ADO den Skalenfaktor 1 hat, auch
                        x~0 = e_0 mit ||x~0|| = 1.  Das Register startet also
                        in |0...0>, es gibt KEINE State-Preparation, und in
                        lambda_t = ||x~0|| s^t sqrt(p_t) faellt der erste
                        Faktor weg.  `build_grid_full` prueft das per assert.

Was hier NICHT mehr klassisch passiert
--------------------------------------
    - keine Arnoldi-Iteration
    - keine Zeitpropagation, weder ganz noch teilweise
    - keine State-Preparation: das Register startet in |0...0>
    - keine Ablese-Matrix R: die Ruecktransformation ist die Diagonale
      vec rho_S = scale[:d^2] * x~[:d^2]

Was klassisch bleibt (und in kritik.ipynb kritisch geprueft wird)
----------------------------------------------------------------
    1. A aufstellen                 O(n) Nichtnullen, strukturiert
    2. P = expm(A dt)               DICHT, O(n^3)   <-- Flaschenhals
    3. dilate(P)                    zwei Matrixwurzeln, O(n^3)  <-- Flaschenhals
    4. U als Gatter laden           2^(2q) Amplituden

Keiner dieser Schritte haengt von der Zahl der Zeitschritte ab -- das ist die
Aussage des Papers und sie gilt hier unveraendert.  Aber 2. bis 4. haengen sehr
wohl von der Hilbertraum-DIMENSION ab, und das ist eine andere Aussage.
"""

import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

# heom_gauge.py liegt im Schwesterordner; robust suchen, damit ein Umbenennen
# der Ordner den Import nicht zerlegt.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _up in range(1, 5):
    _root = os.path.abspath(os.path.join(_HERE, *(['..'] * _up)))
    for _cand in (_root, os.path.join(_root, 'with_compression')):
        if os.path.exists(os.path.join(_cand, 'heom_gauge.py')):
            if _cand not in sys.path:
                sys.path.insert(0, _cand)
            break
    else:
        continue
    break
else:
    raise ImportError('heom_gauge.py nicht gefunden -- liegt es noch im Projekt?')

from heom_gauge import (heom_generator, dilate, safe_norm2,
                        qutip_reference, qutip_reference_rho)

__all__ = ['build_grid_full', 'build_circuit_full', 'estimate_cost',
           'run_full', 'run_full_counts', 'counts_cost',
           'qutip_reference', 'qutip_reference_rho']


# --------------------------------------------------------------------------
# 1.  die einmalige Konstruktion
# --------------------------------------------------------------------------

def build_grid_full(dt_fs, *, depth=2, Nk=1, scale_ados=True,
                    verbose=True, **MODEL):
    """Das eine Gatter, ohne jede Kompression.

    Setzt rho_S(0) = |1><1| voraus (siehe Kopfdokumentation): der Register
    startet in |0...0> und ||x~0|| = 1.

    Returns ein dict mit U (2*np_ x 2*np_), s, n, np_, n_qubits, scale, R, ...
    """
    d = MODEL['H'].shape[0]
    D = d * d
    dt_cm = dt_fs * MODEL['FS_TO_CM']
    t_all = time.time()

    # --- 1. der Erzeuger, mit diagonaler ADO-Skalierung ------------------
    t0 = time.time()
    A, ginfo = heom_generator(depth=depth, Nk=Nk, scale_ados=scale_ados, **MODEL)
    n = ginfo["n"]
    scale = ginfo["ado_scale"] if scale_ados else np.ones(n)
    t_gen = time.time() - t0

    if verbose:
        print(f"  [1/4] Erzeuger A: {n} x {n} ({ginfo['n_ados']} ADOs, "
              f"Tiefe {depth}, Nk={Nk})   {t_gen:.1f} s", flush=True)

    # --- 2. der Propagator ueber einen Zeitschritt ------------------------
    t0 = time.time()
    P = expm(A * dt_cm)

    normP = safe_norm2(P)
    t_prop = time.time() - t0

    if verbose:
        print(f"  [2/4] P = expm(A dt): ||P||_2 = {normP:.8f}   {t_prop:.1f} s", flush=True,)

    # --- 3. auf Zweierpotenz auffuellen und dilatieren --------------------
    t0 = time.time()
    np_ = 2 ** int(np.ceil(np.log2(n)))
    Ppad = np.zeros((np_, np_), complex)
    Ppad[:n, :n] = P              # der Rest bleibt null und wird nie besetzt
    U, s = dilate(Ppad)
    t_dil = time.time() - t0
    n_qubits = int(np.log2(np_)) + 1

    if verbose:
        print(f"  [3/4] Dilatation: {n} -> {np_} gepolstert, U ist "
              f"{2*np_} x {2*np_} = {n_qubits} Qubits, s = {s:.8f}   "
              f"{t_dil:.1f} s", flush=True)


    # Ablesung: vec rho_S = scale[:D] * x~[:D]
    R = np.zeros((D, np_), complex)
    R[:, :D] = np.diag(scale[:D])   # wird in run_full() benötigt für tilde{sigma} -> sigma

    # Der nullte ADO hat den Skalenfaktor 1, also ist x~0 = e_0 exakt.
    assert np.allclose(scale[:D], 1.0), (
        "scale[:d^2] != 1 -- dann ist x~0 nicht e_0 und die Annahme "
        "|0...0> als Anfangszustand traegt nicht.")

    if verbose:
        print(f"  [4/4] Anfangszustand |0,...,0> -- keine State-Preparation, "
              f"||x~0|| = 1")
        print(f"  Konstruktion gesamt: {time.time()-t_all:.1f} s   "
              f"(einmalig, unabhaengig von der Zahl der Zeitschritte)",
              flush=True)
        print(f"  Post-Selektion allein aus der Dilatation: "
              f"s^-200 = {s ** -200:.4f} nach 100 Schritten", flush=True)

    return dict(U=U, s=float(s), n=n, np_=np_, n_qubits=n_qubits, d=d, D=D,
                scale=scale, R=R, P=P, A=A,
                dt_fs=dt_fs, dt_cm=dt_cm, depth=depth, Nk=Nk, scale_ados=True,
                norm_P=float(normP), n_ados=ginfo['n_ados'],
                build_times=dict(generator=t_gen, propagator=t_prop,
                                 dilation=t_dil, total=time.time() - t_all))


# --------------------------------------------------------------------------
# 2.  der Schaltkreis -- ein Gatter, n_steps mal, nichts dazwischen
# --------------------------------------------------------------------------

def build_circuit_full(grid, n_steps, *, save_every=1, save_states=True):
    """Ein durchgaengiger Schaltkreis: einlesen, n_steps mal U, fertig.

    Zwischen t=0 und t=T wird der Zustand NIE ausgelesen und NIE neu praepariert.
    Gemessen wird ausschliesslich die eine Ancilla, und die wird sofort wieder
    auf |0> zurueckgesetzt -- das ist der Preis der Sz.-Nagy-Dilatation, nicht
    eine Unterbrechung der Propagation.  `save_statevector(conditional=True)`
    ist eine reine Simulator-Anweisung: sie liest nichts aus und stoert den
    Zustand nicht, sie legt nur eine Kopie fuer die Auswertung beiseite.
    """
    np_, U = grid['np_'], grid['U']
    n_sys = int(np.log2(np_))
    q_sys = QuantumRegister(n_sys, 'heom')
    q_anc = QuantumRegister(1, 'anc')
    c_anc = ClassicalRegister(n_steps, 'rec')
    qc = QuantumCircuit(q_sys, q_anc, c_anc)

    # x~0 = e_0, das Register startet also schon richtig -- nichts zu tun.
    gate = UnitaryGate(U, label='  U  ')
    for t in range(1, n_steps + 1):
        qc.append(gate, q_sys[:] + q_anc[:])
        qc.measure(q_anc[0], c_anc[t - 1])
        qc.reset(q_anc[0])
        if save_states and (t % save_every == 0 or t == n_steps):
            qc.save_statevector(label=f't{t}', conditional=True)
    return qc


def _accepted_key(saved):
    """Findet den Schluessel (Key) des akzeptierten Zweigs im gespeicherten

    Qiskit-Statevector-Dictionary.
    """
    # -------------------------------------------------------------------------
    # BEISPIEL-SZENARIO:
    # Wenn in Qiskit `save_statevector(conditional=True)` verwendet wird,
    # speichert Aer die Zustandsvektoren als Dictionary mit den Messwerten
    # des klassischen Registers als Keys.
    #
    #   saved = {
    #       '0x4': [0.12, 0.0, ...],   # (Binaer '100': Fehler im 3. Schritt)
    #       '0x1': [0.05, 0.0, ...],   # (Binaer '001': Fehler im 1. Schritt)
    #       '0x0': [0.85, 0.0, ...]    # (Binaer '000': ALLE Ancilla-Messungen = 0 -> ERFOLG)}

    for k in saved:
        # 1. Formatierungen bereinigen:
        #    - '0x' entfernen (Hexadezimal-Praefix von Qiskit/Aer)
        #    - ' ' entfernen (Trennzeichen zwischen mehreren klassischen Registern)
        #    Beispiele:
        #      k = '0x4' -> '4'
        #      k = '0 0' -> '00'
        cleaned = k.replace("0x", "").replace(" ", "")

        # 2. Pruefen, ob der Mess-String AUSSCHLIESSLICH aus '0' besteht:
        #    `set(cleaned)` bildet die Menge aller vorkommenden Zeichen.
        #    `<= {'0'}` prueft, ob diese Menge eine Teilmenge von {'0'} ist.
        #
        #    Beispiel-Auswertungen:
        #      k = '0x4' -> cleaned = '4'  -> set = {'4'}      -> nicht <= {'0'} -> False (Verwerfen)
        #      k = '0x0' -> cleaned = '0'  -> set = {'0'}      -> Teilmenge von {'0'} -> True  (Gefunden!)
        if set(cleaned) <= {"0"}:
            return k  # Gibt sofort den passenden Original-Key zurueck (z. B. '0x0')

    return None    # Falls in keinem einzigen Shot der physikalische Zweig ueberlebt hat


def _prefix_counts(counts, times):
    """Berechnet die kumulative Post-Selektions-Wahrscheinlichkeit p(t)

    fuer jeden Zeitpunkt t aus den Bitstring-Zaehlraten der Ancilla-Messungen.
    """
    # -------------------------------------------------------------------------
    # BEISPIEL-SZENARIO (fuer die Kommentare unten):
    # - 1000 Shots, 3 Zeitschritte (times = [1, 2, 3]).
    # - In jedem Zeitschritt t wird das Hilfsqubit (Ancilla) gemessen:
    #     0 = Erfolg (Zustand bleibt im physikalischen HEOM-Zweig)
    #     1 = Fehler (Kollaps in den unphysikalischen Dilatations-Zweig)
    # - counts = {'000': 700, '100': 200, '010': 80, '001': 20}
    # -------------------------------------------------------------------------

    # 1. Gesamtzahl aller gemessenen Durchlaeufe bestimmen:
    #    total = 700 + 200 + 80 + 20 = 1000
    total = sum(counts.values())

    # 2. Qiskits Bit-Reihenfolge korrigieren (Big-Endian -> chronologisch):
    #    Qiskit schreibt das klassische Bit c[t-1] des letzten Schritts ganz LINKS:
    #      Bitstring 'b' = "c[2] c[1] c[0]"  (Schritt 3, Schritt 2, Schritt 1)
    #    Durch `[::-1]` kehren wir den String um:
    #      String 'r'    = "c[0] c[1] c[2]"  (Schritt 1, Schritt 2, Schritt 3)
    #
    #    Beispiel-Transformationen:
    #      '000' -> '000' (700 Shots: Schritt 1=0, Schritt 2=0, Schritt 3=0)
    #      '100' -> '001' (200 Shots: Schritt 1=0, Schritt 2=0, Schritt 3=1 -> Fehler erst bei t=3)
    #      '010' -> '010' ( 80 Shots: Schritt 1=0, Schritt 2=1, Schritt 3=0 -> Fehler bei t=2)
    #      '001' -> '100' ( 20 Shots: Schritt 1=1, Schritt 2=0, Schritt 3=0 -> Fehler bei t=1)
    #    recs = [('000', 700), ('001', 200), ('010', 80), ('100', 20)]
    recs = [(b.replace(" ", "")[::-1], c) for b, c in counts.items()]

    # 3. Post-Selektion je Zeitschritt t:
    #    Fuer Zeitpunkt t interessieren uns nur die ersten t Messungen (Praefix r[:t]).
    #    `set(r[:t]) <= {'0'}` prueft, ob bis Schritt t AUSSCHLIESSLICH Nullen gemessen wurden.
    #
    #    - Auswertung fuer t=1 (Bedingung r[:1] in {'0'}):
    #        '000'[:1] = '0' (Ja, 700) | '001'[:1] = '0' (Ja, 200) | '010'[:1] = '0' (Ja, 80) | '100'[:1] = '1' (Nein)
    #        Summe = 700 + 200 + 80 = 980 -> p(1) = 980 / 1000 = 0.98
    #
    #    - Auswertung fuer t=2 (Bedingung r[:2] in {'00'}):
    #        '000'[:2] = '00' (Ja, 700) | '001'[:2] = '00' (Ja, 200) | '010'[:2] = '01' (Nein) | '100'[:2] = '10' (Nein)
    #        Summe = 700 + 200 = 900 -> p(2) = 900 / 1000 = 0.90
    #
    #    - Auswertung fuer t=3 (Bedingung r[:3] in {'000'}):
    #        '000'[:3] = '000' (Ja, 700) | alle anderen haben mindestens eine '1' (Nein)
    #        Summe = 700 -> p(3) = 700 / 1000 = 0.70
    #
    #    Ergebnis: np.array([0.98, 0.90, 0.70])
    return np.array([sum(c for r, c in recs if set(r[:t]) <= {"0"}) / total
            for t in times])


# --------------------------------------------------------------------------
# 3.  vorher wissen, ob der Lauf durchfuehrbar ist
# --------------------------------------------------------------------------

def estimate_cost(grid, n_steps, shots, *, probe_steps=3, probe_shots=8,
                  verbose=True):
    """Miss einen Mini-Lauf und rechne auf den geplanten Lauf hoch.

    Aer simuliert jeden Shot als eigene Trajektorie, weil die Ancilla-Messung
    mitten im Schaltkreis den Zustand kollabieren laesst.  Die Kosten sind
    deshalb linear in (Schritte x Shots), und ein kurzer Test genuegt zur
    Hochrechnung.
    """
    np_ = grid['np_']
    qc = build_circuit_full(grid, probe_steps, save_states=False)
    sim = AerSimulator(method='statevector')
    tqc = transpile(qc, sim)
    t0 = time.time()
    sim.run(tqc, shots=probe_shots).result()
    dt = time.time() - t0
    per_app = dt / (probe_steps * probe_shots)
    total = per_app * n_steps * shots
    mem_gate = U_bytes = (2 * np_) ** 2 * 16 / 2**20
    if verbose:
        print(f"  Kostenschaetzung (aus {probe_steps} Schritten x "
              f"{probe_shots} Shots, {dt:.2f} s gemessen)")
        print(f"    Qubits                  : {grid['n_qubits']} "
              f"({np_} Zustaende + 1 Ancilla)")
        print(f"    Gatter U im Speicher    : {mem_gate:.0f} MB")
        print(f"    Gatteranwendungen       : {n_steps} x {shots} = "
              f"{n_steps*shots:,}")
        print(f"    pro Anwendung           : {per_app*1e3:.1f} ms")
        print(f"    HOCHRECHNUNG            : {total/60:.1f} min "
              f"({total:.0f} s)")
    return dict(per_application=per_app, total_seconds=total,
                gate_mib=mem_gate, applications=n_steps * shots)


# --------------------------------------------------------------------------
# 4.  der Lauf, in Shot-Stapeln, mit Fortschritt
# --------------------------------------------------------------------------

def run_full(grid, n_steps, *, shots=256, save_every=1, verbose=True):
    """Den Schaltkreis laufen lassen und jeden gespeicherten Zeitpunkt auslesen.

    Die Shots werden in Stapeln abgearbeitet, damit man den Fortschritt sieht.
    Das unterbricht den Schaltkreis NICHT: jeder einzelne Shot laeuft von t=0
    bis t=T durch, die Stapel sind nur unabhaengige Wiederholungen desselben
    durchgaengigen Laufs.  Was sich ueber die Stapel akkumuliert, ist allein die
    Statistik fuer p_t; die Richtung des Zustands ist im akzeptierten Zweig
    deterministisch und schon aus einem einzigen Shot exakt.

    Returns (pops, pops_tr, info) wie run_grid in der komprimierten Variante.
    """
    d, D, s, R, np_ = grid['d'], grid['D'], grid['s'], grid['R'], grid['np_']

    times = [t for t in range(1, n_steps + 1)
             if t % save_every == 0 or t == n_steps]
    ts = np.array([0] + times)

    qc = build_circuit_full(grid, n_steps, save_every=save_every)
    sim = AerSimulator(method='statevector')
    t0 = time.time()
    tqc = transpile(qc, sim)
    t_trans = time.time() - t0

    if verbose:
        print(f"  Schaltkreis: {grid['n_qubits']} Qubits, {n_steps} "
              f"Anwendungen EINES Gatters, Tiefe {qc.depth()}, "
              f"transpiliert in {t_trans:.1f} s")

    t_start = time.time()
    res = sim.run(tqc, shots=shots).result()
    acc = _prefix_counts(res.get_counts(), times)
    data = res.data()
    states = {}

    for t in times:          # prüft ob es mindestens einen lauf gab in dem die ancilla immer 0 war, denn dann gibt es statevector
        saved = data.get(f't{t}')
        key = _accepted_key(saved) if saved else None
        if key is not None:
            states[t] = np.asarray(saved[key])[:np_]    # nimmt die obere Hälfte des statevectors

    if verbose:
        print(f"  {shots} Shots in {time.time()-t_start:.0f} s, "
              f"p(T) = {acc[-1]/shots:.4f}", flush=True)

    prob = np.concatenate([[1.0], acc / shots])
    prob_err = np.concatenate([[0.0], np.where(
        (acc > 0) & (acc < shots), np.sqrt(acc / shots * (1 - acc / shots) / shots),
        3.0 / shots)])

    pops = np.full((len(ts), d), np.nan)
    pops_tr = np.full((len(ts), d), np.nan)
    rhos = np.full((len(ts), d, d), np.nan, complex)
    rhos_tr = np.full((len(ts), d, d), np.nan, complex)
    rhos_err = np.full((len(ts), d, d), np.nan)

    x_dir = np.zeros(np_, complex)
    x_dir[0] = 1.0                      # x~0 = e_0

    for k, t in enumerate(ts):
        if t > 0:
            if t not in states:
                break                       # kein Shot hat bis hierhin ueberlebt
            x_dir = states[t]
        rho = (R @ x_dir).reshape(d, d, order='F')
        tr = np.trace(rho)
        pops_tr[k] = np.real(np.diag(rho / tr))
        rhos_tr[k] = rho / tr
        rho = rho * np.exp(-1j * np.angle(tr))
        lam = s ** t * np.sqrt(prob[k])      # ||x~0|| = 1
        rho_abs = rho * lam
        rhos[k] = rho_abs
        rhos_err[k] = np.abs(rho_abs) * (prob_err[k] / (2 * prob[k])
                                         if prob[k] > 0 else 0.0)
        pops[k] = np.real(np.diag(rho_abs))

    info = dict(t_index=ts, p_success=prob, p_success_err=prob_err,
                accepted=acc, shots=shots,
                n_qubits=grid['n_qubits'], s=s, depth=qc.depth(),
                n_gate_applications=n_steps, wall_seconds=time.time() - t_start,
                rho=rhos, rho_tr=rhos_tr, rho_err=rhos_err)
    if verbose:
        print(f"  fertig in {info['wall_seconds']:.0f} s, "
              f"Post-Selektion 1.0000 -> {prob[-1]:.4f}", flush=True)
    return pops, pops_tr, info

# --------------------------------------------------------------------------
# 5.  Ablesung aus echten Zaehlraten -- kein Statevector
# --------------------------------------------------------------------------
#
# Ohne Kompression ist die Ablesung DEUTLICH einfacher als in gauge_circuit.py.
# Dort ist vec rho_S = R y eine dichte Linearkombination ueber den ganzen
# Krylov-Raum, jede Population braucht also eine eigene m-dimensionale
# Basisdrehung.  Hier gilt
#
#     vec rho_S = scale[:d^2] * x~[:d^2],
#
# die Ablesezeile r_jj ist also ein EINZELNER Basisvektor.  Daraus folgt:
#
#   * Die Diagonale braucht GAR KEINE Drehung.  Man misst das Systemregister in
#     der Rechenbasis, und ein einziges Histogramm liefert alle d Populationen
#     auf einmal -- statt d Schaltkreisen wie im komprimierten Fall.
#   * r_++ = (r_aa + r_bb + r_ab + r_ba)/2 lebt auf genau VIER Koordinaten.  Die
#     Basisdrehung ist ein eingebetteter 4x4-Block, keine dichte 1024x1024-
#     Matrix -- sie kostet O(1) klassisch statt O(n^3).
#
# Der zweite Punkt ist auch fuer die Skalierungsfrage wichtig: eine dichte
# Drehung waere ein weiterer nicht skalierender Schritt gewesen.

def _vec_index(d, i, j):
    """Spaltenweise vec-Konvention: vec(rho)[i + d*j] = rho[i, j]."""
    return i + d * j


def _prep_unitary(chi):
    """Unitaere Matrix V mit V|0...0> = chi (chi normiert).

    Wortgleich mit `_prep_unitary` in gauge_circuit.py.  Gemessen wird nach
    Anwendung von V^dagger die Wahrscheinlichkeit des Basiszustands |0...0>,
    und die ist |<chi|x~>|^2 -- denn V^dag|chi> = |0...0>.

    Kosten: eine QR-Zerlegung auf np_ x np_.  Bei np_ = 1024 sind das 337 ms,
    einmal je Ablesekanal (nicht je Zeitschritt) -- gegen einen Lauf von Minuten
    also nichts.  Das Ergebnis ist allerdings eine DICHTE Drehung; auf Hardware
    waere das der Unterschied zwischen ein paar Gattern und 4^10.  Hier
    unkritisch, weil dieses Modul mit 11 Qubits ohnehin nur in Aer laeuft.
    """
    n = chi.shape[0]                        # Dimension des (gepolsterten) Registers
    M = np.eye(n, dtype=complex)            # Startmatrix: Identitaet
    M[:, 0] = chi                           # erste Spalte durch chi ersetzen
    Q, Rr = np.linalg.qr(M)                 # QR: M = Q R, also Q[:,0] = chi / R[0,0]
    dg = np.diag(Rr)                        # diag(R) traegt die Phasen
    return Q * (dg / np.abs(dg))            # macht diag(R) positiv -> Q[:,0] == chi


def _counts_circuit(grid, n_steps, B=None):
    """Derselbe durchgaengige Schaltkreis, aber am Ende wird gemessen.

    Ein einziges klassisches Register: Bits 0..n_steps-1 sind die
    Ancilla-Aufzeichnung, die restlichen das Systemregister.  So gibt es keine
    Zweideutigkeit in der Registerreihenfolge von Qiskit.
    """
    np_ = grid['np_']
    n_sys = int(np.log2(np_))
    q_sys = QuantumRegister(n_sys, 'heom')
    q_anc = QuantumRegister(1, 'anc')
    creg = ClassicalRegister(n_steps + n_sys, 'c')
    qc = QuantumCircuit(q_sys, q_anc, creg)

    # x~0 = e_0, das Register startet also schon richtig -- nichts zu tun.
    gate = UnitaryGate(grid['U'], label='  U  ')
    for t in range(1, n_steps + 1):
        qc.append(gate, q_sys[:] + q_anc[:])
        qc.measure(q_anc[0], creg[t - 1])
        qc.reset(q_anc[0])
    if B is not None:
        qc.append(UnitaryGate(B.conj().T, label='B+'), q_sys[:])
    for i in range(n_sys):
        qc.measure(q_sys[i], creg[n_steps + i])
    return qc


def _split_counts(counts: dict[str, int], n_steps: int, n_sys: int) -> tuple[int, int, dict[int, int]]:
    """Extrahiert akzeptierte Shots und das Zustands-Histogramm aus den Messzaehlraten.
    """
    # Gesamtzahl aller uebergebenen Shots berechnen
    total = sum(counts.values())        # counts = {
    acc = 0                             #    '00101': 42,   # Zählrate c = 42 (wurde 42-mal gemessen)
    hist = {}                           #    '00000': 150,  # Zählrate c = 150 (wurde 150-mal gemessen)
                                        #    '10101': 832   # Zählrate c = 832
                                        #        }
    for bits, c in counts.items():
        # Leerzeichen entfernen und Bitstring umkehren (Qiskit Big-Endian -> Little-Endian),
        # sodass r[k] exakt dem klassischen Register-Bit / Qubit k entspricht.
        r = bits.replace(' ', '')[::-1]

        # Post-Selektion: Pruefe, ob unter den ersten n_steps Bits ein Bit ungleich '0' ist.
        # Falls ja, ist der Zweig ungueltig (z. B. fehlgeschlagene Kraus-Operation) -> verwerfen.
        if set(r[:n_steps]) - {'0'}:
            continue

        # Shot erfuellt die Bedingung -> zur Anzahl akzeptierter Durchlaeufe addieren
        acc += c

        # System-Bits (ab Index n_steps bis n_steps + n_sys - 1) in Dezimalindex umwandeln:
        # sum_{i=0}^{n_sys-1} (bit_i * 2^i). z.B. 101 -> 5
        idx = sum((1 << i) for i in range(n_sys) if r[n_steps + i] == '1')

        # Haeufigkeit des gemessenen System-Basiszustands im Histogramm akkumulieren
        # Angenommen, hist war vor diesem Schritt hist = {0: 100, 5: 10}
        # hist[5] = hist.get(5, 0) + 42  # 10 + 42 = 52 -> hist enthält nun {0: 100, 5: 52}.
        hist[idx] = hist.get(idx, 0) + c    

    return total, acc, hist


def counts_cost(grid, n_steps, shots, *, n_pairs=0, save_every=None,
                per_application=None, verbose=True):
    """Wie lange ein Zaehlraten-Lauf dauert -- BEVOR man ihn startet.

    Misst die Kosten je Gatteranwendung an einem Mini-Schaltkreis (oder nimmt
    `per_application`, falls schon bekannt) und rechnet hoch.  Jede Auslesezeit
    t braucht einen eigenen Schaltkreis mit t Anwendungen, weil die Messung den
    Zustand zerstoert; die Kosten sind also die SUMME der Auslesezeiten, nicht
    die letzte.
    """
    times = _readout_times(n_steps, save_every)
    n_circ = 1 + 2 * n_pairs                    # 1 Diagonale + 2 je Paar
    apps = sum(times) * n_circ * shots
    if per_application is None:
        # GRENZkosten messen, nicht die Gesamtkosten eines Mini-Laufs: ein kurzer
        # Schaltkreis besteht ueberwiegend aus Fixkosten (Transpilation, Aufsetzen
        # je Shot), und daraus hochgerechnet kommt rund das DOPPELTE heraus
        # (gemessen: 101 ms statt der tatsaechlichen 55 ms je Anwendung).
        # Deshalb zwei Groessen laufen lassen und die Differenz nehmen.
        sim = AerSimulator(method='statevector')
        t_probe, s_probe = (2, 6), 24
        secs = []
        for tp in t_probe:
            tqc = transpile(_counts_circuit(grid, tp), sim)
            t0 = time.time()
            sim.run(tqc, shots=s_probe).result()
            secs.append(time.time() - t0)
        per_application = ((secs[1] - secs[0])
                           / ((t_probe[1] - t_probe[0]) * s_probe))
    total = apps * per_application
    if verbose:
        print(f"  Kostenschaetzung Zaehlraten-Ablesung")
        print(f"    Auslesezeiten        : {list(times)}")
        print(f"    Schaltkreise je Zeit : {n_circ} "
              f"(1 Diagonale + 2 x {n_pairs} Paare)")
        print(f"    Shots je Schaltkreis : {shots}")
        print(f"    Gatteranwendungen    : sum(t) x {n_circ} x {shots} = {apps:,}")
        print(f"    pro Anwendung        : {per_application*1e3:.1f} ms")
        print(f"    HOCHRECHNUNG         : {total/60:.1f} min ({total:.0f} s)")
    return dict(per_application=per_application, total_seconds=total,
                applications=apps, times=list(times), circuits_per_time=n_circ)


def _readout_times(n_steps, save_every=None):
    if save_every is None:
        return [n_steps]
    return [t for t in range(1, n_steps + 1)
            if t % save_every == 0 or t == n_steps]


def _measure_kanal(grid, t, B, shots, sim):
    """Einen Ablesekanal messen: Schaltkreis bauen, laufen lassen, auswerten.

    `B = None` heisst "keine Basisdrehung" -- das ist der Diagonalkanal, dessen
    Histogramm alle d Populationen auf einmal liefert.  Mit `B` misst man
    stattdessen den Ueberlapp mit EINEM gedrehten Zustand.

    Liefert (acc, hist): akzeptierte Shots und das Histogramm ueber die
    Basisindizes des Systemregisters, beides schon post-selektiert.
    """
    n_sys = int(np.log2(grid['np_']))                     # Qubits im Systemregister
    qc = transpile(_counts_circuit(grid, t, B), sim)      # Schaltkreis bauen und auf den Simulator uebersetzen
    counts = sim.run(qc, shots=shots).result().get_counts()  # ausfuehren, Zaehlraten holen
    _, acc, hist = _split_counts(counts, t, n_sys)        # Post-Selektion anwenden und Histogramm bilden
    return acc, hist


def run_full_counts(grid, n_steps, pairs=None, *, shots=512,
                    save_every=None, verbose=True):
    """Die volle Dichtematrix aus ECHTEN Zaehlraten -- kein Statevector.

    Diagonale: ein Schaltkreis je Auslesezeit, ohne jede Basisdrehung.  Das
    Histogramm ueber die Rechenbasis liefert
        q_jj = P(Basiszustand j + d*j | akzeptiert),
        rho_jj = scale_jj * lambda_t * sqrt(q_jj),   lambda_t = s^t sqrt(p_t).
    Alle d Populationen kommen aus DEMSELBEN Histogramm -- im komprimierten Fall
    braucht das d getrennte Schaltkreise, weil die Ablesezeilen dort dicht sind.

    Off-Diagonale (`pairs`): je Paar zwei weitere Schaltkreise mit der ++/RR-
    Drehung auf den vier Koordinaten (aa, bb, ab, ba):
        rho_++ = (rho_aa + rho_bb)/2 + Re rho_ab
        rho_RR = (rho_aa + rho_bb)/2 + Im rho_ab
    Beides sind echte Besetzungen, also >= 0 (Cauchy-Schwarz + AM-GM); die
    Wurzel ist damit eindeutig und es braucht keine Phasenreferenz.

    Fehler: der absolute Fehler einer Ablesung ist ||r|| lambda_t / (2 sqrt(N_acc)),
    unabhaengig von q.  Fuer Re/Im kommen die beiden Diagonalbeitraege dazu.

    Die Schaltkreise laufen nacheinander, und nach jedem wird gemeldet, der
    wievielte von wie vielen gerade fertig ist.  (Aer parallelisiert nicht ueber
    Experimente -- gemessen 63.3 ms gegen 59.2 ms je Anwendung -- ein
    gemeinsamer Aufruf braechte also nichts ausser weniger Fortschrittsmeldung.)

    Returns (rho, rho_err, info); `info['rho_err_im']` traegt den Im-Fehler,
    `info['rho_tr']` die spurnormierte Variante.
    """
    # --- 1. Parameter und Dimensionen aus dem Grid-Dictionary extrahieren ---
    d, D = grid['d'], grid['D']                                 # d: Anzahl Sites, D = d^2: Dimension von vec(rho_S)
    s, np_ = grid['s'], grid['np_']                             # s: Dilatationsfaktor, np_: auf 2^n_sys gepolsterte Dimension
    scale = grid['scale']                                       # diagonale ADO-Skalierung; scale[:D] == 1, also ||x~0|| = 1
    n_sys = int(np.log2(np_))                                   # Anzahl Qubits im Systemregister
    sim = AerSimulator(method='statevector')                    # Schuss-Simulator; Zaehlraten statt Statevector
    times = _readout_times(n_steps, save_every)                 # Zeitpunkte, an denen ausgelesen wird
    pairs = list(pairs) if pairs else []                        # Kohaerenz-Paare (a, b), evtl. leer

    # --- 2. Ablesekanaele festlegen (haengen NICHT von t ab, also einmal) ---
    # Ein Eintrag ist ein Tupel (tag, Bdag, nrm) -- wie `rows`, `Bdag`
    # und `nrm` in gauge_circuit.run_grid_counts, nur zusammengefasst:
    #   tag  identifiziert den Kanal in der Auswertung
    #   Bdag die Basisdrehung (None = Rechenbasis, also die ganze Diagonale)
    #   nrm  ||r||, der Vorfaktor in rho = ||r|| lambda_t sqrt(q)
    # Nach B^dag traegt IMMER der Basiszustand |0...0> den Ueberlapp -- genau wie
    # in gauge_circuit.py, deshalb braucht es keinen eigenen Schluessel je Kanal.
    rows = [('diag', None, 1.0)]                             # Diagonalkanal: keine Drehung, ein Histogramm fuer alle d Populationen
    for (a, b) in pairs:
        idx = [_vec_index(d, a, a), _vec_index(d, b, b),        # die vier vec-Indizes, auf denen r_++ und r_RR leben
               _vec_index(d, a, b), _vec_index(d, b, a)]
        sc = np.array([scale[i] for i in idx])                  # zugehoerige Skalenfaktoren (hier alle 1, aber allgemein gehalten)
        # r_++ = (r_aa + r_bb + r_ab + r_ba)/2  und  r_RR mit -i/+i auf ab/ba.
        # chi = conj(r)/||r||, deshalb stehen unten die KONJUGIERTEN Vorzeichen.
        for tag, coef in (('pp', np.array([1., 1., 1., 1.])),
                          ('RR', np.array([1., 1., -1j, 1j]))):
            r = 0.5 * sc * coef                                 # die Ablesezeile, eingeschraenkt auf die vier Koordinaten
            nrm = float(np.linalg.norm(r))                       # ||r||: die Verstaerkung, die spaeter den Fehler bestimmt
            chi = np.zeros(np_, complex)                        # Analysevektor im ganzen gepolsterten Raum
            chi[idx] = np.conj(r) / nrm                          # nur auf den vier Koordinaten besetzt, sonst null
            Bdag = _prep_unitary(chi)                              # Drehung mit B|0...0> = chi
            rows.append(((tag, a, b), Bdag, nrm))                # Kanal merken

    # --- 3. Speicher-Arrays fuer die Ergebnisse allokieren ---
    n_circ = len(times) * len(rows)                          # Gesamtzahl der Schaltkreise
    apps = sum(times) * len(rows) * shots                    # Gatteranwendungen: sum(t) x Kanaele x Shots
    rho = np.full((len(times), d, d), np.nan, complex)          # volle Dichtematrix je Auslesezeit
    rho_tr = np.full((len(times), d, d), np.nan, complex)       # dieselbe, spurnormiert
    err_re = np.full((len(times), d, d), np.nan)                # 1-sigma-Fehler des Realteils (auf der Diagonale: der Population)
    err_im = np.full((len(times), d, d), np.nan)                # 1-sigma-Fehler des Imaginaerteils
    prob = np.full(len(times), np.nan)                          # Post-Selektions-Wahrscheinlichkeit p_total(t)

    if verbose:
        print(f"  {grid['n_qubits']} Qubits | {len(times)} Auslesezeiten "
              f"{list(times)} | {len(rows)} Schaltkreise je Zeit "
              f"| {n_circ} Kreise gesamt")
        print(f"  {shots} Shots je Schaltkreis -> {apps:,} Gatteranwendungen",
              flush=True)

    # --- 4. Hauptschleife ueber die Auslesezeiten ---
    t_start = time.time()                                       # fuer Laufzeit und Restschaetzung
    done = 0                                                    # Zaehler der bereits gelaufenen Schaltkreise
    for k, t in enumerate(times):

        # 4a. Der Diagonalkanal.  Er kommt zuerst, weil seine akzeptierten Shots
        #     p_total(t) festlegen und die Populationen fuer die Subtraktion in
        #     4b ohnehin gebraucht werden.
        acc, hist = _measure_kanal(grid, t, None, shots, sim)   # messen, ohne Drehung
        done += 1
        if verbose:
            el = time.time() - t_start                          # bisher verbrauchte Zeit
            print(f"    Schaltkreis {done} von {n_circ} (t={t}, diag) -- "
                  f"{el:5.0f} s verbraucht, noch etwa "
                  f"{el/done*(n_circ-done):4.0f} s", flush=True)
        if acc == 0:                                            # kein Shot hat die Post-Selektion ueberlebt
            if verbose:
                print(f"  t={t}: kein Shot ueberlebt die Post-Selektion")
            done += len(rows) - 1                            # die restlichen Kanaele dieses t entfallen
            continue                                            # dieser Zeitpunkt bleibt NaN

        prob[k] = acc / shots                                   # empirische Post-Selektions-Wahrscheinlichkeit
        lam = s ** t * np.sqrt(prob[k])                         # lambda_t = ||x~0|| s^t sqrt(p_t), und ||x~0|| = 1

        # rho_jj = scale_jj * lambda_t * sqrt(q_jj) mit q_jj aus DEMSELBEN Histogramm
        pop = np.array([scale[_vec_index(d, j, j)] * lam
                        * np.sqrt(hist.get(_vec_index(d, j, j), 0) / acc)
                        for j in range(d)])
        # Der absolute Fehler ist ||r|| lambda_t / (2 sqrt(N_acc)) -- unabhaengig
        # von q, weil sich das sqrt(q) aus der Statistik gegen das 1/sqrt(q) aus
        # der Ableitung der Wurzel kuerzt.
        dpop = np.array([scale[_vec_index(d, j, j)] * lam / (2 * np.sqrt(acc))
                         for j in range(d)])

        M = np.diag(pop).astype(complex)                        # Dichtematrix, zunaechst nur die Diagonale
        E, Eim = np.diag(dpop), np.zeros((d, d))                # Fehlermatrizen fuer Re und Im

        # 4b. Je Paar die beiden gedrehten Basen ++ und RR.
        for (a, b) in pairs:
            vals = {}                                          # sammelt (rho_++, drho_++) und (rho_RR, drho_RR)
            for tag, Bdag, nrm in rows[1:]:                      # rows[0] ist der Diagonalkanal
                if tag[1:] != (a, b):                           # nur die Kanaele dieses Paares
                    continue
                acc2, hist2 = _measure_kanal(grid, t, Bdag, shots, sim)
                done += 1
                if verbose:
                    el = time.time() - t_start
                    print(f"    Schaltkreis {done} von {n_circ} "
                          f"(t={t}, {tag}) -- {el:5.0f} s verbraucht, noch etwa "
                          f"{el/done*(n_circ-done):4.0f} s", flush=True)
                if acc2 == 0:                                   # dieser Kanal ist leer
                    vals = None
                    break
                lam2 = s ** t * np.sqrt(acc2 / shots)           # eigenes lambda_t, weil eigene Post-Selektion
                vals[tag[0]] = (nrm * lam2 * np.sqrt(hist2.get(0, 0) / acc2),
                                 nrm * lam2 / (2 * np.sqrt(acc2)))
            if vals is None:                                   # Paar nicht auswertbar -> ueberspringen
                continue

            (p_pp, d_pp), (p_RR, d_RR) = vals['pp'], vals['RR']
            half = 0.5 * (pop[a] + pop[b])                      # (rho_aa + rho_bb)/2, wird von beiden abgezogen
            quart = 0.25 * dpop[a] ** 2 + 0.25 * dpop[b] ** 2   # deren Fehlerbeitrag, quadratisch
            M[a, b] = (p_pp - half) + 1j * (p_RR - half)        # Re aus ++, Im aus RR
            M[b, a] = np.conj(M[a, b])                          # rho ist hermitesch
            E[a, b] = E[b, a] = np.sqrt(d_pp ** 2 + quart)      # Fehler des Realteils
            Eim[a, b] = Eim[b, a] = np.sqrt(d_RR ** 2 + quart)  # Fehler des Imaginaerteils

        # 4c. Ergebnisse dieses Zeitpunkts sichern
        rho[k], err_re[k], err_im[k] = M, E, Eim
        tr = np.real(np.trace(M))                               # Spur; sollte 1 sein
        if tr > 0:
            rho_tr[k] = M / tr                                  # spurnormierte Variante, robuster gegen Skalenfehler

    # --- 5. Metadaten zusammenstellen und zurueckgeben ---
    wall = time.time() - t_start
    if verbose:
        print(f"  fertig in {wall:.0f} s ({wall/apps*1e3:.1f} ms je Anwendung)",
              flush=True)
    return rho, err_re, dict(
        t_index=np.array(times), p_success=prob, shots=shots, pairs=pairs,
        rho_tr=rho_tr, rho_err_im=err_im, n_qubits=grid['n_qubits'], s=s,
        circuits_per_time=len(rows), n_circuits=n_circ,
        applications=apps, wall_seconds=wall)
