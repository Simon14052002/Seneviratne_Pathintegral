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
    for k in saved:
        if set(k.replace('0x', '').replace(' ', '')) <= {'0'}:
            return k
    return None


def _prefix_probs(counts, times):
    """p_t = fraction of shots whose record is 0 in bits 0..t-1.

    Qiskit prints classical registers with the highest bit first, so the string
    is reversed before indexing by step number.
    """
    total = sum(counts.values())
    recs = [(b.replace(' ', '')[::-1], c) for b, c in counts.items()]
    return np.array([sum(c for r, c in recs if set(r[:t]) <= {'0'}) / total
                     for t in times])


def run_grid(grid, n_steps, *, shots=512, save_every=1, verbose=True):
    """Run the grid ONCE and read every stored time out of that single run.

    Returns (pops, pops_trace, info); see the module docstring for the two
    readouts.
    """
    d, m, s, R = grid['d'], grid['m'], grid['s'], grid['R']
    qc = build_circuit(grid, n_steps, save_every=save_every)
    sim = AerSimulator(method='statevector')
    result = sim.run(transpile(qc, sim), shots=shots).result()
    data = result.data()

    times = [t for t in range(1, n_steps + 1)
             if t % save_every == 0 or t == n_steps]
    ts = np.array([0] + times)
    prob = np.concatenate([[1.0], _prefix_probs(result.get_counts(), times)])

    scale0 = np.linalg.norm(grid['y0'])
    pops = np.full((len(ts), d), np.nan)
    pops_tr = np.full((len(ts), d), np.nan)

    y_dir = np.zeros(m, complex)
    y_dir[0] = 1.0
    for k, t in enumerate(ts):
        if t > 0:
            saved = data.get(f't{t}')
            key = _accepted_key(saved) if saved else None
            if key is None:                    # no shot survived to this time
                break
            y_dir = np.asarray(saved[key])[:m]
        rho = (R @ y_dir).reshape(d, d, order='F')
        pops_tr[k] = np.real(np.diag(rho / np.trace(rho)))
        # (5.1).  The saved branch carries an arbitrary global phase; it is
        # fixed by Tr rho_S(t) = 1 > 0, i.e. the physical trace must be real
        # and positive.  Only the phase is taken from the trace here -- the
        # magnitude comes from the measurement record.
        rho = rho * np.exp(-1j * np.angle(np.trace(rho)))
        pops[k] = np.real(np.diag(rho)) * scale0 * s ** t * np.sqrt(prob[k])

    info = dict(t_index=ts, p_success=prob, n_qubits=grid['n_qubits'], m=m, s=s,
                shots=shots, depth=qc.depth(), n_gate_applications=n_steps)
    if verbose:
        print(f"  grid: {grid['n_qubits']} qubits, {n_steps} applications of one "
              f"gate, circuit depth {qc.depth()}, {shots} shots", flush=True)
        print(f"  post-selection probability: 1.0000 -> {prob[-1]:.4f}", flush=True)
    return pops, pops_tr, info


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
