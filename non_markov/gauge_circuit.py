"""
The grid itself: one Qiskit circuit, one gate, arbitrarily many steps.

The circuit carries two registers,

    krylov   n_sys = log2(mp) qubits   holding the compressed HEOM state y(t)
    anc      1 qubit                   the Sz.-Nagy ancilla,

and repeats a single fixed unitary U = dilate(H_m).  Applying U to |y>|0> and
projecting the ancilla onto |0> realises

    |y>  -->  (H_m / s) |y> ,

so after t repetitions with every ancilla reading 0 the krylov register holds
H_m^t y0 / ||H_m^t y0||, and the measurement record itself supplies the missing
norm,

    ||y_t|| = s^t sqrt(p_t) ||y0||.                                      (5.1)

Because the gauge makes ||H_m|| <= s ~ 1, p_t does not collapse: what is left of
it is the physical relaxation of the HEOM state, not a representation artefact.

Why the post-selected branch is simulated exactly
-------------------------------------------------
Post-selection and deterministic simulation do not mix.  Aer implements a
mid-circuit `measure` by *sampling* an outcome, in the density-matrix method
just as in the statevector method: a two-qubit test with an exact 0.3/0.7 split
returns [0,1] at one shot and [0.284,0.716] at 1024.  Any route that reads the
post-selected branch out of a saved density matrix -- for instance by measuring
the ancilla and conditionally dumping failures into a sink state -- is therefore
Monte-Carlo, with an error falling off only as 1/sqrt(shots).

The branch we want does not need that.  Conditioned on the record 0...0 the
post-measurement state is *deterministic*: projecting a pure state onto a fixed
subspace and renormalising leaves no randomness.  A single accepted shot
therefore already carries the exact state, and shots are needed only for the
scalar p_t.  The circuit below measures the ancilla into its own classical bit
at every step and uses `save_statevector(conditional=True)`, which groups the
saved states by the classical record: the entry keyed by all zeros is exactly
the post-selected branch, sampling-free.

Two readouts follow.  `pops` restores ||y_t|| from p_t via (5.1) and is the
operational route -- it uses nothing but the measurement record.  `pops_trace`
uses Tr rho_S(t) = 1 instead and needs no statistics at all.  They are
independent, so their agreement is a self-contained check on the construction.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator


def build_circuit(grid, n_steps, *, save_every=1, save_states=True):
    """The single continuous grid: n_steps repetitions of one gate.

    The initial state needs no preparation: y0 = ||x~0|| e_1 by construction of
    the Arnoldi basis, so the register starts in |0...0> already.
    """
    mp, U = grid['mp'], grid['U']
    n_sys = int(np.log2(mp))
    q_sys = QuantumRegister(n_sys, 'krylov')
    q_anc = QuantumRegister(1, 'anc')
    c_anc = ClassicalRegister(n_steps, 'rec')
    qc = QuantumCircuit(q_sys, q_anc, c_anc)

    gate = UnitaryGate(U, label='  U  ')
    for t in range(1, n_steps + 1):
        qc.append(gate, q_sys[:] + q_anc[:])
        qc.measure(q_anc[0], c_anc[t - 1])
        qc.reset(q_anc[0])
        if save_states and (t % save_every == 0 or t == n_steps):
            qc.save_statevector(label=f't{t}', conditional=True)
    return qc


def _accepted_key(saved):
    """The entry of a conditional save that belongs to the record 0...0."""
    # Iteriert über alle im Dictionary gespeicherten Mess-Schlüssel (Keys)
    for k in saved:
        # 1. k.replace('0x', '').replace(' ', ''): 
        #    Entfernt Hex-Präfixe ('0x') und Leerzeichen, die Qiskit je nach Format setzt (z. B. '0x00 0' -> '000').
        # 2. set(...): 
        #    Erzeugt eine Menge der verbleibenden Zeichen (z. B. '00' -> {'0'}, '010' -> {'0', '1'}).
        # 3. <= {'0'}: 
        #    Prüft, ob die Zeichenmenge eine Teilmenge von {'0'} ist.
        #    Das ist genau dann True, wenn alle gemessenen Ancilla-Bits 0 waren (Erfolgszweig der Post-Selection).
        if set(k.replace('0x', '').replace(' ', '')) <= {'0'}:
            return k  # Gibt den Key des erfolgreichen Messzweiges zurück
            
    return None      # Falls kein Shot überlebt hat (überall mindestens eine 1 gemessen wurde)


def _prefix_probs(counts, times):
    """p_t = fraction of shots whose record is 0 in bits 0..t-1.

    Qiskit prints classical registers with the highest bit first, so the string
    is reversed before indexing by step number.
    """
    # 1. Gesamtzahl aller durchgeführten Circuit-Shots berechnen
    #    counts ist ein Dict der Form {'0010': 350, '0100': 162}; sum(counts.values()) = 512
    total = sum(counts.values())
    
    # 2. Formatierung und zeitliche Ausrichtung der Bitstrings:
    #    - b.replace(' ', ''): Entfernt Formatierungs-Leerzeichen zwischen Registerblöcken.
    #    - [::-1]: Qiskit gibt klassische Register in Big-Endian aus (Bit n-1 steht ganz links bei Index 0).
    #      Durch das Umkehren (Reverse) wandert Bit 0 (erster Zeitschritt) nach ganz links auf Index 0.
    #      Beispiel: Bitstring '001' (Messung bei t=3, t=2, t=1) wird zu '001'[::-1] = '001' (Index 0 ist t=1, Index 1 ist t=2, etc.).
    #    - recs wird eine Liste von Tupeln: [(zeitlich_geordneter_string, anzahl_shots), ...] da counts.itmes = ([('0000', 180), ('1000', 142),...])
    recs = [(b.replace(' ', '')[::-1], c) for b, c in counts.items()]
    
    # 3. Kumulative Post-Selection-Wahrscheinlichkeit für jeden Zeitpunkt t berechnen:
    #    - for t in times: Iteriert über alle Zeitschritte t (z. B. t = 1, 2, ..., n_steps).
    #    - r[:t]: Schneidet den Prefix der ersten t Zeitschritte heraus (Index 0 bis t-1).
    #    - set(r[:t]) <= {'0'}: Bildet eine Menge der Zeichen im Prefix und prüft, ob diese eine
    #      Teilmenge von {'0'} ist. Das ist genau dann True, wenn in den ersten t Schritten ausschließlich
    #      Nullen gemessen wurden (also kein einziger Kollaps in den Fehler-Zweig stattfand).
    #      Was nach Schritt t passiert, ist für den Zustand zum Zeitpunkt t irrelevant.
    #    - sum(c for r, c in recs if ...): Addiert die Anzahl aller Shots, die bis Schritt t überlebt haben.
    #    - / total: Teilt durch die Gesamtzahl der Shots -> ergibt die empirische Erfolgswahrscheinlichkeit p(t) in [0, 1].
    #    - np.array(...): Gibt die Wahrscheinlichkeiten als 1D-NumPy-Array zurück.
    return np.array([sum(c for r, c in recs if set(r[:t]) <= {'0'}) / total
                     for t in times])


def run_grid(grid, n_steps, *, shots=512, save_every=1, verbose=True):
    """Run the grid ONCE and read every stored time out of that single run.

    Returns (pops, pops_trace, info); see the module docstring for the two
    readouts.

    Die VOLLE Dichtematrix steckt in `info`, nicht nur die Diagonale:

        info['rho']     (len(ts), d, d) komplex -- vollstaendig rekonstruiert
        info['rho_tr']  dieselbe, spur-normiert
        info['rho_err'] 1-sigma-Fehler auf |rho_ij|

    Die Kohaerenzen kosten nichts extra: vec rho_S = R y_t liefert ohnehin alle
    d^2 Eintraege, die Diagonale war bisher nur die einzige, die herausgelesen
    wurde.  Der Fehler ist fuer alle Eintraege derselbe relative Faktor, denn
    rho ~ sqrt(p_t) und p_t ist die einzige gesampelte Groesse -- die Richtung
    von y_t kommt aus dem bedingten Statevector und ist exakt.
    """
    # 1. Parameter aus dem Vorbereitungs-Dictionary entpacken
    d, m, s, R = grid['d'], grid['m'], grid['s'], grid['R']  # d: Systemdim, m: Krylovdim, s: Norm/Skalierungsfaktor, R: Rekonstruktionsmatrix
    
    # 2. Qiskit-Quantenschaltkreis mit Messungen & Zwischenspeicherungen bauen
    qc = build_circuit(grid, n_steps, save_every=save_every)  # Baut Circuit mit n_steps Dilatationsgattern & SaveStatevector-Befehlen
    
    # 3. Simulation mit Qiskit Aer Statevector-Engine ausführen
    sim = AerSimulator(method='statevector')                  # Exakter Statevector-Simulator für Quantenschaltkreise
    result = sim.run(transpile(qc, sim), shots=shots).result()  # Transpiliert Circuit auf Simulator und führt ihn Shots-mal aus. man braucht aucht genügend Shots um p_total zu bestimmen
    data = result.data()                                      # Extrahiert alle gespeicherten Statevectors und Messdaten

    # 4. Zeitachsen-Indizes und kumulative Post-Selection-Wahrscheinlichkeiten ermitteln
    times = [t for t in range(1, n_steps + 1)                 # Liste aller Zeitschritte, an denen Daten abgespeichert wurden
             if t % save_every == 0 or t == n_steps]
    ts = np.array([0] + times)                                # Vollständige Zeitachse inklusive Startzeitpunkt t=0
    prob = np.concatenate([[1.0], _prefix_probs(result.get_counts(), times)])  # Empirische Überlebenswahrscheinlichkeit der Post-Selection pro Zeitschritt

    # 5. Datenstrukturen für die Populationen initialisieren
    scale0 = np.linalg.norm(grid['y0'])                       # Norm des ursprünglichen unkomprimierten Startvektors ||Sigma_0||_2
    pops = np.full((len(ts), d), np.nan)                      # Array für vollständig rekonstruierte Populationen (mit NaN vorbelegt)
    pops_tr = np.full((len(ts), d), np.nan)                   # Array für spur-normalisierte Populationen (Trace Tr(rho)=1)

    rhos    = np.full((len(ts), d, d), np.nan, complex)       # VOLLE Dichtematrix je Zeitpunkt (inkl. Kohaerenzen)
    rhos_tr = np.full((len(ts), d, d), np.nan, complex)       # dieselbe, spur-normiert
    rhos_err= np.full((len(ts), d, d), np.nan)                # 1-sigma-Fehler auf |rho_ij|

    # Binomialfehler auf p_t; er wirkt als GEMEINSAMER Faktor auf alle Eintraege,
    # denn rho ~ sqrt(p_t).  Also delta|rho_ij| / |rho_ij| = delta_p / (2 p).
    acc = np.round(prob * shots).astype(int)
    prob_err = np.where((acc > 0) & (acc < shots),
                        np.sqrt(prob * (1 - prob) / shots), 3.0 / shots)
    prob_err[0] = 0.0                                         # bei t=0 wurde noch nicht gemessen

    y_dir = np.zeros(m, complex)                              # Start-Zustandsvektor im m-dimensionalen Krylov-Raum
    y_dir[0] = 1.0                                            # Startzustand ist exakt der erste Basisvektor e_1 = (1, 0, ..., 0)^T
    
    # 6. Iteration über alle gespeicherten Zeitschritte
    for k, t in enumerate(ts):
        if t > 0:
            saved = data.get(f't{t}')                         # Gespeichertes Zustandsvektor-Dictionary zum Zeitschritt t abrufen
            key = _accepted_key(saved) if saved else None     # Sucht den Zweig, bei dem alle Ancilla-Messungen bis t erfolgreich '0' waren
            if key is None:                                   # Falls kein einziger Shot bis zu diesem Zeitschritt überlebt hat:
                break                                         # Breche Schleife ab (nachfolgende Zeiten bleiben NaN)
            y_dir = np.asarray(saved[key])[:m]                # Schneidet das m-dimensionale Systemregister aus dem Gesamtzustand heraus

        # 7. Klassische Rücktransformation in die physikalische Dichtematrix
        rho = (R @ y_dir).reshape(d, d, order='F')            # R @ y_dir liefert vec(rho_S); reshape(..., order='F') baut d x d Matrix

        # 8. Auslesen der spur-normalisierten Populationen
        pops_tr[k] = np.real(np.diag(rho / np.trace(rho)))    # Erzwingt Tr(rho)=1 und liest reelle Diagonalelemente (Besetzungen) aus

        # 9. Globale Phaseneichung und vollständige Skalenrekonstruktion
        # Die Quantenmechanik lässt eine globale Phase e^(i*phi) frei; wir eichen sie so, dass Tr(rho_S) rein reell & positiv ist:
        rho = rho * np.exp(-1j * np.angle(np.trace(rho)))     # Zieht den Phasenwinkel der Spur ab: rho -> rho * e^(-i * arg(Tr(rho)))
        # Multipliziert Startnorm, Dilatations-Kompensation s^t und Messwahrscheinlichkeit sqrt(p) hinzu:
        lam = scale0 * s ** t * np.sqrt(prob[k])              # lambda_t: die klassisch mitgefuehrte Laenge
        rho_abs = rho * lam                                   # vollstaendig rekonstruierte Dichtematrix
        rhos[k]    = rho_abs
        rhos_tr[k] = rho / np.trace(rho)
        rhos_err[k] = np.abs(rho_abs) * (prob_err[k] / (2 * prob[k]) if prob[k] > 0 else 0.0)
        pops[k] = np.real(np.diag(rho_abs))                   # Populationen = Diagonale davon

    # 10. Metadaten für die Analyse zusammenstellen
    info = dict(t_index=ts, p_success=prob, p_success_err=prob_err,
                n_qubits=grid['n_qubits'], m=m, s=s,
                shots=shots, depth=qc.depth(), n_gate_applications=n_steps,
                rho=rhos, rho_tr=rhos_tr, rho_err=rhos_err)
    if verbose:
        print(f"  grid: {grid['n_qubits']} qubits, {n_steps} applications of one "
              f"gate, circuit depth {qc.depth()}, {shots} shots", flush=True)
        print(f"  post-selection probability: 1.0000 -> {prob[-1]:.4f}", flush=True)
        
    return pops, pops_tr, info


def _prefix_counts(counts, times):
    """Wie `_prefix_probs`, gibt aber die absoluten Zahlen zurueck.

    Fuer Fehlerbalken braucht man nicht den Anteil, sondern Zaehler und Nenner:
    aus `accepted` und `shots` folgt der Binomialfehler, aus dem Anteil allein
    nicht.
    """
    total = sum(counts.values())
    recs = [(b.replace(' ', '')[::-1], c) for b, c in counts.items()]
    acc = np.array([sum(c for r, c in recs if set(r[:t]) <= {'0'}) for t in times],
                   dtype=int)
    return acc, total


def run_grid_shots(grid, n_steps, *, shots=2048, save_every=1,
                               verbose=True):
    """Dieselbe Ausgabe wie `run_grid_shots`, aber aus EINEM Schaltkreis.

    Was gegenueber `run_grid` geaendert werden muss
    ----------------------------------------------
    Inhaltlich rechnen beide dasselbe: der akzeptierte Zweig ist deterministisch
    (Proposition 24), und p_t ist in beiden Faellen der gemessene Anteil der
    Shots mit Protokoll 0...0.  Es fehlen `run_grid` nur vier Dinge:

    1. **Absolute Zaehlraten statt nur des Anteils.**  `_prefix_probs` liefert
       `ok/shots`; fuer den Binomialfehler braucht man `ok` und `shots` einzeln.
       Dafuer ist `_prefix_counts` da.
    2. **Fehlerfortpflanzung.**  Aus delta_p = sqrt(p(1-p)/N) wird ueber
       ||y_t|| = s^t sqrt(p_t) ||y_0||  und  d(sqrt p)/sqrt p = dp/(2p)
       der Fehler der Populationen.
    3. **Die Dreierregel.**  Bei ok == shots kollabiert der Wald-Fehler auf
       exakt 0 und unterschaetzt die Unsicherheit; dort nimmt man 3/N.
    4. **Rueckgabesignatur.**  `(pops, pops_err, info)` statt
       `(pops, pops_trace, info)`, und `info` enthaelt zusaetzlich
       `p_success_err` und `accepted`.

    Der eine echte Unterschied
    --------------------------
    Die Statistik ist **verschachtelt statt unabhaengig**.  Hier stammen alle
    p_t aus demselben Satz von `shots` Laeufen, die Ereignisse "Protokoll bis t
    ist null" sind ineinander enthalten, und deshalb gilt shot-weise exakt
    p_{t+1} <= p_t.  Bei `run_grid_shots` wird pro Zeitpunkt neu gewuerfelt, die
    Schaetzer schwanken unabhaengig, und diese Monotonie kann im Sample verletzt
    sein.  Der Erwartungswert ist derselbe.

    Fuer echte Hardware ist die Variante hier sogar die *bessere*: Ein Geraet mit
    Mid-Circuit-Messung und mitgefuehrtem klassischem Protokoll bekommt alle
    Zeiten aus einem Satz Laeufe, statt fuer jeden Zeitpunkt von vorne zu
    beginnen -- bei n Zeitpunkten spart das rund einen Faktor n an Shots.

    Nicht hardware-treu ist in beiden Faellen dasselbe: die *Richtung* von y_t
    stammt aus Aers bedingtem Statevector.  Auf einem Geraet braeuchte man dafuer
    Tomographie des Krylov-Registers, weil vec rho_S = R y ein LINEARES
    Funktional ist, Zaehlraten aber nur |y_i|^2 liefern.

    Returns
    -------
    pops     : (len(times), d)  Populationen
    pops_err : (len(times), d)  1-sigma-Statistikfehler
    info     : dict mit t_index, p_success, p_success_err, accepted, shots, ...
    """
    d, m, s, R = grid['d'], grid['m'], grid['s'], grid['R']

    qc = build_circuit(grid, n_steps, save_every=save_every)
    sim = AerSimulator(method='statevector')
    result = sim.run(transpile(qc, sim), shots=shots).result()
    data = result.data()

    times = [t for t in range(1, n_steps + 1)
             if t % save_every == 0 or t == n_steps]
    ts = np.array([0] + times)

    # (1) absolute Zaehlraten statt nur des Anteils
    acc_t, total = _prefix_counts(result.get_counts(), times)
    accepted = np.concatenate([[total], acc_t])
    prob = accepted / total

    # (3) Dreierregel, wo der Wald-Fehler kollabiert
    prob_err = np.where((accepted > 0) & (accepted < total),
                        np.sqrt(prob * (1 - prob) / total),
                        3.0 / total)
    prob_err[0] = 0.0        # bei t=0 wurde noch nicht gemessen: p = 1 ist exakt

    scale0 = np.linalg.norm(grid['y0'])
    pops = np.full((len(ts), d), np.nan)
    pops_err = np.full((len(ts), d), np.nan)

    y_dir = np.zeros(m, complex)
    y_dir[0] = 1.0

    for k, t in enumerate(ts):
        if t > 0:
            if accepted[k] == 0:                 # (4) Abbruch wie im Shot-Lauf
                if verbose:
                    print(f"  Abbruch bei t={t}: 0 von {total} Shots akzeptiert.")
                break
            saved = data.get(f't{t}')
            key = _accepted_key(saved) if saved else None
            if key is None:
                break
            y_dir = np.asarray(saved[key])[:m]

        rho = (R @ y_dir).reshape(d, d, order='F')
        # globale Phase ueber die Spurerhaltung fixieren
        rho = rho * np.exp(-1j * np.angle(np.trace(rho)))
        pops[k] = np.real(np.diag(rho)) * scale0 * s ** t * np.sqrt(prob[k])
        # (2) Fehlerfortpflanzung: d(sqrt p)/sqrt p = dp/(2p)
        pops_err[k] = np.abs(pops[k]) * prob_err[k] / (2 * prob[k])

    info = dict(t_index=ts, p_success=prob, p_success_err=prob_err,
                accepted=accepted, shots=shots, n_qubits=grid['n_qubits'],
                m=m, s=s, depth=qc.depth(), n_gate_applications=n_steps)
    if verbose:
        print(f"  ein Schaltkreis: {grid['n_qubits']} Qubits, {n_steps} Gatter, "
              f"Tiefe {qc.depth()}, {shots} Shots", flush=True)
        print(f"  p_succ 1.0000 -> {prob[-1]:.4f} "
              f"({accepted[-1]} von {total} Shots)", flush=True)
    return pops, pops_err, info


def run_grid_final(grid, n_steps, *, shots=20000, verbose=True):
    """The minimal statement of the task: one state in, one fixed gate n times,
    one state out -- the circuit is read only at t = n_steps."""
    pops, pops_tr, info = run_grid(grid, n_steps, shots=shots,
                                   save_every=n_steps, verbose=False)
    if verbose:
        n_ok = int(round(info['p_success'][-1] * shots))
        print(f"  t = {n_steps * grid['dt_fs']:6.0f} fs: p_succ = "
              f"{info['p_success'][-1]:.4f}  ({n_ok} of {shots} shots accepted)",
              flush=True)
    return pops[-1], pops_tr[-1], info


def _prep_unitary(chi):
    """Unitaere Matrix V mit V|0...0> = chi (chi normiert)."""
    n = chi.shape[0]
    M = np.eye(n, dtype=complex)
    M[:, 0] = chi
    Q, Rr = np.linalg.qr(M)
    dg = np.diag(Rr)
    return Q * (dg / np.abs(dg))          # macht diag(R) positiv -> Q[:,0] == chi


def counts_readout_cost(grid, times, shots=None, verbose=True):
    """Sagt VORAB, ob eine zaehlratenbasierte Ablesung ueberhaupt Sinn hat.

    Die Ueberlappe q_j = |<chi_j|y_t>|^2, an denen alles haengt, lassen sich rein
    klassisch aus H_m, y_0 und R ausrechnen -- in Millisekunden, ohne einen
    einzigen Schaltkreis.  Der relative Fehler auf p_j ist etwa
    1/(2*sqrt(q_j * p_t * shots)), man kann also vorher sagen, wieviele Shots man
    braucht, statt es hinterher am Rauschen zu merken.

    Returns (q, needed) mit q[k, j] und `needed` = Shots fuer 1 % relativ.
    """
    d, m, s, R = grid['d'], grid['m'], grid['s'], grid['R']
    y0, Hm = grid['y0'], grid['Hm']
    scale0 = np.linalg.norm(y0)
    rows = [R[j + d * j, :] for j in range(d)]
    chi = [np.conj(r) / np.linalg.norm(r) for r in rows]

    times = list(times)
    q = np.zeros((len(times), d))
    p_t = np.zeros(len(times))
    y = y0.astype(complex).copy()
    tmax = max(times)
    for t in range(tmax + 1):
        if t in times:
            k = times.index(t)
            nt = np.linalg.norm(y)
            p_t[k] = (nt / (s ** t * scale0)) ** 2
            yd = y / nt
            q[k] = [abs(np.vdot(c, yd)) ** 2 for c in chi]
        if t < tmax:
            y = Hm @ y
    live = q[q > 1e-14]
    qmin = live.min() if live.size else 0.0
    needed = np.inf if qmin == 0 else 1.0 / (1e-4 * qmin * p_t.min())
    if verbose:
        print(f"  Vorabdiagnose: kleinstes relevantes q = {qmin:.2e}, "
              f"p_t >= {p_t.min():.3f}")
        print(f"  -> fuer 1 % relative Genauigkeit noetig: ~{needed:.1e} Shots"
              + (f"  (angefragt: {shots})" if shots else ""))
        if shots and shots < needed / 100:          # schlechter als 10 %
            print("  ACHTUNG: viel zu wenige Shots.  Erwartete Genauigkeit ~"
                  f"{1/(2*np.sqrt(max(qmin*p_t.min()*shots, 1e-30))):.0%}.")
            print("  Ursache ist fast immer ein zu KLEINES delta oder "
                  "scale_ados=True:")
            print("  beides vergroessert ||y_0|| = ||W^(1/2) x_0|| und damit "
                  "lambda_t, und q ~ 1/lambda_t^2.")
            print("  Fuer diese Ableseroute ein Gitter mit groesserem delta "
                  "(z.B. 0.02) und OHNE scale_ados verwenden --")
            print("  genau umgekehrt zur Empfehlung fuer p_succ.")
    return q, needed


def run_grid_counts(grid, n_steps, *, shots=8192, save_every=1, verbose=True):
    """Populationen AUSSCHLIESSLICH aus Zaehlraten -- kein Statevector.

    Das Markovsche Analogon (`propagate_stinespring_shots`) funktioniert, weil das
    Systemregister dort direkt rho_S traegt: |psi_i|^2 = rho_ii, die Zaehlraten
    SIND die Populationen.  Hier traegt das Register y_t in der Krylov-Basis, und
    vec rho_S = R y_t ist LINEAR in y_t -- Zaehlraten liefern aber nur |y_i|^2.

    Ausweg.  Sei r_j die Zeile von R, die rho_jj herausgreift, also
    p_j = lambda_t (r_j . y_t).  Praepariert man |chi_j> = r_j^dag/||r_j|| und
    dreht ihn mit B_j^dag auf |0...0>, so ist die gemessene Rate im Nullzustand
        q_j = |<chi_j|y_t>|^2 = (r_j . y_t)^2 / ||r_j||^2,
    und weil rho_S hermitesch und positiv semidefinit ist, ist r_j . y_t reell
    und >= 0 -- die Wurzel ist eindeutig.  Die unbekannte globale Phase von y_t
    faellt im Betragsquadrat heraus, es braucht also keine Phaseneichung mehr.

        p_j = ||r_j|| * sqrt(q_j) * ||y_0|| * s^t * sqrt(p_t)

    Kosten: d Schaltkreise pro Auslesezeit (einer je Site) statt einem.  Dafuer
    ist nichts mehr am Simulator abgelesen -- das laeuft auf Hardware.

    Returns
    -------
    pops     : spur-normierte Populationen, Sum_j p_j = 1 per Konstruktion.
               Das kuerzt lambda_t heraus und macht "Spur 0" oder "Spur 3"
               strukturell unmoeglich -- beides trat auf, solange jede
               Population einzeln ueber lambda_t ~ 730 skaliert wurde und ein
               einzelner Ausreisser-Count entsprechend hochgezogen wurde.
    pops_err : 1-sigma-Statistikfehler dazu.
    info     : u.a. `q` (die gemessenen Ueberlappe), `pops_abs` (die absolute
               Variante ueber lambda_t) und `p_success`.

    Wieviele Shots?  Der relative Fehler ist etwa 1/(2*sqrt(q_j * N_acc)).
    Gemessen fuer das 4-Site-FMO-Gitter (delta=0.02, OHNE ADO-Skalierung):
    q ~ 0.01...0.1, also 2.5e-02 Genauigkeit bei 8192 Shots und ~1e6...1e7 Shots
    fuer 1e-3.  Die Voreinstellung 8192 ist eine Demo-Einstellung, keine
    Produktionseinstellung.

    ACHTUNG, kontraintuitiv: fuer DIESE Ableseroute will man das Gitter OHNE
    `scale_ados`.  Die ADO-Skalierung rettet zwar die Kontraktion, verschlechtert
    aber die Ueberlappe um fuenf Groessenordnungen (q von 1e-2 auf 1e-8, also
    1e11 statt 1e6 Shots) -- sie dreht den Winkel zwischen Registerzustand und
    Auslesefunktional ungueenstig.  Dahinter steht, dass nur ~5e-4 der Laenge von
    Sigma ueberhaupt der Systemblock ist; das Signal ist intrinsisch ein winziger
    Anteil des Registers, und eine Projektionsmessung zahlt das mit N ~ 1/q.
    Der prinzipielle Ausweg waere Amplitudenschaetzung (N ~ 1/sqrt(q)), hier
    nicht implementiert.
    """
    d, m, s, R = grid['d'], grid['m'], grid['s'], grid['R']
    mp, U = grid['mp'], grid['U']
    n_sys = int(np.log2(mp))
    gate = UnitaryGate(U, label='U')
    scale0 = np.linalg.norm(grid['y0'])
    sim = AerSimulator()

    rows = [R[j + d * j, :] for j in range(d)]          # Spaltenstapelung: rho_jj
    nrm = np.array([np.linalg.norm(r) for r in rows])
    Bdag = []
    for r in rows:
        chi = np.zeros(mp, complex)
        chi[:m] = r.conj() / np.linalg.norm(r)
        Bdag.append(UnitaryGate(_prep_unitary(chi).conj().T, label='B'))

    times = [0] + [t for t in range(1, n_steps + 1)
                   if t % save_every == 0 or t == n_steps]

    # Vorabdiagnose, bevor irgendein Schaltkreis laeuft
    counts_readout_cost(grid, times, shots=shots, verbose=verbose)

    pops       = np.full((len(times), d), np.nan)
    pops_err   = np.full((len(times), d), np.nan)
    pops_n     = np.full((len(times), d), np.nan)
    pops_n_err = np.full((len(times), d), np.nan)
    qtab       = np.full((len(times), d), np.nan)
    prob = np.full(len(times), np.nan)
    prob_err = np.zeros(len(times))

    for k, t in enumerate(times):
        row, row_err = np.zeros(d), np.zeros(d)
        qs, qerr, nacc = np.zeros(d), np.zeros(d), np.zeros(d, dtype=int)
        for j in range(d):
            q_sys = QuantumRegister(n_sys, 'krylov')
            q_anc = QuantumRegister(1, 'anc')
            c = ClassicalRegister(t + n_sys, 'c')       # Bits 0..t-1: Protokoll
            qc = QuantumCircuit(q_sys, q_anc, c)        # Bits t..: Ausgang

            for step in range(t):
                qc.append(gate, q_sys[:] + q_anc[:])
                qc.measure(q_anc[0], c[step])
                qc.reset(q_anc[0])

            qc.append(Bdag[j], q_sys[:])

            for i in range(n_sys):
                qc.measure(q_sys[i], c[t + i])

            counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
            acc = zero = 0

            for bits, n in counts.items():
                b = bits.replace(' ', '')[::-1]          # b[0] = c[0]
                if set(b[:t]) <= {'0'}:
                    acc += n
                    if set(b[t:t + n_sys]) <= {'0'}:
                        zero += n
            if acc == 0:
                break
            
            if j == 0:
                prob[k] = acc / shots
                prob_err[k] = (np.sqrt(prob[k] * (1 - prob[k]) / shots)
                               if 0 < acc < shots else 3.0 / shots)
                if t == 0:
                    prob_err[k] = 0.0
            q = zero / acc
            # Bei 0 Counts NICHT einfach 0 melden -- das ist eine Nachweisgrenze,
            # keine Messung.  Die 95-%-Obergrenze ist 3/acc.
            dq = np.sqrt(q * (1 - q) / acc) if 0 < zero < acc else 3.0 / acc
            qs[j], qerr[j], nacc[j] = q, dq, acc
            row[j] = nrm[j] * np.sqrt(q) * scale0 * s ** t * np.sqrt(prob[k])
            rel = np.hypot(dq / (2 * q) if q > 0 else 0.0,
                           prob_err[k] / (2 * prob[k]) if prob[k] > 0 else 0.0)
            row_err[j] = row[j] * rel
        else:
            pops[k], pops_err[k] = row, row_err
            # Spur-normierte Variante: Sum_j p_j = 1 ist exakt bekannt, also
            # teilt man durch die gemessene Summe.  Das kuerzt lambda_t und alle
            # gemeinsamen multiplikativen Fehler heraus -- und macht "Spur 0"
            # oder "Spur 3" strukturell unmoeglich.
            tot_row = row.sum()
            if tot_row > 0:
                pops_n[k] = row / tot_row
                pops_n_err[k] = row_err / tot_row      # gemeinsamer Faktor faellt raus
            qtab[k] = qs
            if verbose:
                nmin = int(qs.min() * nacc.min())
                rel = (1.0 / (2 * np.sqrt(max(nmin, 1)))) if nmin else np.nan
                print(f"  t={t:4d}  p_succ={prob[k]:.4f}  Spur={tot_row:.4f}  "
                      f"q={np.array2string(qs, precision=2)}  "
                      f"min Counts={nmin}  -> erwartete rel. Genauigkeit "
                      f"{rel:.1%}" if nmin else
                      f"  t={t:4d}  p_succ={prob[k]:.4f}  Spur={tot_row:.4f}  "
                      f"q={np.array2string(qs, precision=2)}  "
                      f"min Counts=0 (unter der Nachweisgrenze)", flush=True)
            continue
        break

    info = dict(t_index=np.array(times), p_success=prob, p_success_err=prob_err,
                shots=shots, n_qubits=grid['n_qubits'], m=m, s=s,
                circuits_per_time=d, q=qtab,
                pops_abs=pops, pops_abs_err=pops_err)
    return pops_n, pops_n_err, info
