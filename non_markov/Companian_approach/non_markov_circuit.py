import numpy as np
from scipy.linalg import sqrtm

# Qiskit Core & Circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import DensityMatrix

# Qiskit Aer Simulator
from qiskit_aer import AerSimulator


def arnoldi(E, x0, m):
    """Orthonormal basis Q_m of the Krylov space and H_m = Q_m^dag E Q_m."""
    Q = np.zeros((E.shape[0], m + 1), complex)
    Q[:, 0] = x0 / np.linalg.norm(x0)
    H = np.zeros((m + 1, m), complex) 
    mm = m

    for j in range(m):
        w = E @ Q[:, j]

        for i in range(j + 1):
            H[i, j] = np.vdot(Q[:, i], w)
            w -= H[i, j] * Q[:, i]

        H[j + 1, j] = np.linalg.norm(w)

        if H[j + 1, j] < 1e-13:
            mm = j + 1; break
        Q[:, j + 1] = w / H[j + 1, j]

    return H[:mm, :mm], Q[:, :mm]


def dilate(A):
    """Sz.-Nagy: A/s as the top-left block of a unitary one qubit larger."""
    n = A.shape[0]; s = float(np.linalg.norm(A, 2)) * (1 + 1e-12); As = A / s
    I = np.eye(n)
    B = sqrtm(I - As @ As.conj().T)
    C = sqrtm(I - As.conj().T @ As)
    B, C = 0.5 * (B + B.conj().T), 0.5 * (C + C.conj().T)
    return np.block([[As, B], [C, -As.conj().T]]), s




def propagate_memory_shots(E, X0, n_steps, m, d, shots=10000):
    """
    Shot-basierte Version der Nicht-Markovschen Simulation.
    """
    D = d * d
    Hm, Qm = arnoldi(E, X0, m)
    m = Hm.shape[0]
    
    mp = 2 ** int(np.ceil(np.log2(m + 1)))
    A = np.zeros((mp, mp), complex)
    A[:m, :m] = Hm
    
    U, s = dilate(A)
    n_sys = int(np.log2(mp))
    gate = UnitaryGate(U, label='U_Hm')

    sim = AerSimulator()
    pops = np.zeros((n_steps + 1, d))
    
    scale = np.linalg.norm(X0)
    y0 = np.zeros(mp, complex); y0[0] = 1.0
    pops[0, :] = np.real(np.diag((Qm @ y0[:m] * scale)[:D].reshape(d, d, order='F')))
    
    return_qc = None
    p_tot = 1.0

    for t in range(1, n_steps + 1):
        print(f"Zeitschritt {t}/{n_steps}", end='\r')
        
        q_sys = QuantumRegister(n_sys, 'sys')
        q_anc = QuantumRegister(1, 'anc')
        c_anc = ClassicalRegister(t, 'c_anc')
        qc = QuantumCircuit(q_sys, q_anc, c_anc)

        psi0 = np.zeros(2**(n_sys + 1), complex)
        psi0[0] = 1.0
        qc.initialize(psi0, q_sys[:] + q_anc[:])

        for step in range(t):
            qc.append(gate, q_sys[:] + q_anc[:])
            qc.measure(q_anc[0], c_anc[step])
            if step < t - 1:
                qc.reset(q_anc[0])

        qc.save_statevector(conditional=True)
        
        # Immer den aktuellen Circuit als Fallback bereithalten
        return_qc = qc

        result = sim.run(transpile(qc, sim), shots=shots).result()
        counts = result.get_counts()
        
        # 1. Zähle alle Shots, die ausschließlich aus '0' (und ggf. Leerzeichen) bestehen
        success_shots = sum(count for bitstr, count in counts.items() if set(bitstr) <= {'0', ' '})
        
        if success_shots == 0:
            print(f"\nAbbruch bei t={t}: 0 von {shots} Shots waren erfolgreich (Post-Selection Grenze erreicht).")
            break
            
        p_tot = success_shots / shots
        state_dict = result.data()['statevector']
        
        # 2. Finde den passenden Key für den reinen Null-Zustand (auch für >64 Bits robust)
        target_key = None
        for k in state_dict.keys():
            # Entfernt Präfixe wie '0x' und Leerzeichen und prüft, ob nur Nullen übrig bleiben
            cleaned = k.replace('0x', '').replace(' ', '')
            if set(cleaned) <= {'0'}:
                target_key = k
                break
                
        if target_key is None:
            print(f"\nAbbruch bei t={t}: Kein Statevector für den Erfolgszweig gefunden.")
            break
            
        psi_full = np.array(state_dict[target_key])
        
        # Da Ancilla 0 ist, liegt das System in den ersten mp Einträgen
        y_t = psi_full[:mp]
        y_t = y_t / np.linalg.norm(y_t)
        
        idx = np.argmax(np.abs(y_t))
        y_t *= np.exp(-1j * np.angle(y_t[idx]))
        
        current_scale = scale * (s ** t) * np.sqrt(p_tot)
        vec_rho = (Qm @ y_t[:m] * current_scale)[:D]
        pops[t, :] = np.real(np.diag(vec_rho.reshape(d, d, order='F')))

    print()
    return pops, return_qc, dict(m=m, n_qubits=n_sys + 1, s=s, p_success=p_tot, norm_Hm=float(np.linalg.norm(Hm, 2)))




def propagate_memory_density_matrix(E, X0, n_steps, m, d):
    """
    ONE circuit, T = n_steps*dt, deterministic via exact density matrix simulation.
    Nutzt extrem schnelle native Qiskit-Gatter (if_test + reset), um den Fehl-Ast 
    in den Mülleimer |N-1> zu werfen, anstatt langsame Kraus-Matrizen zu nutzen.
    """
    D = d * d
    Hm, Qm = arnoldi(E, X0, m)
    m = Hm.shape[0]
    
    n_sys = int(np.ceil(np.log2(m + 1)))
    N = 2 ** n_sys
    
    A = np.zeros((N, N), complex)
    A[:m, :m] = Hm
    
    U, s = dilate(A)
    n_qubits = n_sys + 1
    
    # Eigene Register definieren, um klassische Bedingung (Measurement) zu nutzen
    q_sys = QuantumRegister(n_sys, 'sys')
    q_anc = QuantumRegister(1, 'anc')
    c_anc = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(q_sys, q_anc, c_anc)
    
    rho_init = np.zeros((2**n_qubits, 2**n_qubits), complex)
    rho_init[0, 0] = 1.0
    qc.set_density_matrix(DensityMatrix(rho_init))
    qc.save_density_matrix(q_sys, label='t0')
    
    gate = UnitaryGate(U, label='U_Hm')
    
    for t in range(1, n_steps + 1):
        qc.append(gate, q_sys[:] + q_anc[:])
        
        # 1. Wir messen die Ancilla.
        qc.measure(q_anc, c_anc)
        
        # 2. High-Speed Garbage Sink: Wenn Ancilla == 1, resette das System und setze 
        # es auf |111...1> (was exakt Zustand |N-1> ist).
        with qc.if_test((c_anc, 1)):
            for sq in q_sys:
                qc.reset(sq)
                qc.x(sq)
            
        # 3. Ancilla für den nächsten Durchlauf freimachen (unabhängig vom Messergebnis)
        qc.reset(q_anc)
        qc.save_density_matrix(q_sys, label=f't{t}')
        
    # Führt Schaltkreis auf Qiskits Dichtematrix-Simulator aus
    sim = AerSimulator(method='density_matrix')
    data = sim.run(qc).result().data()
    
    # =========================================================================
    # Klassische Rekonstruktion EXAKT aus den Qiskit-Daten
    # =========================================================================
    scale = np.linalg.norm(X0)
    pops = np.zeros((n_steps + 1, d))
    
    y0 = np.zeros(N, complex); y0[0] = 1.0
    pops[0, :] = np.real(np.diag((Qm @ y0[:m] * scale)[:D].reshape(d, d, order='F')))
    
    p_success_last = 1.0
    for t in range(1, n_steps + 1):
        rho_sim = np.asarray(data[f't{t}'])
        
        # Exakter Erfolgs-Ast ist perfekt isoliert im oberen linken m x m Block
        rho_success = rho_sim[:m, :m]
        p_tot = np.real(np.trace(rho_success))
        
        # Sicherung gegen Abbruch am Ende der Trajektorie
        if p_tot < 1e-15:
            break
            
        p_success_last = p_tot
        
        evals, evecs = np.linalg.eigh(rho_success)
        y_t = evecs[:, np.argmax(evals)]
        
        # =====================================================================
        # DER PHASEN-FIX: Globale Phase über die physikalische Spur extrahieren
        # =====================================================================
        # Vektor testweise in den physikalischen Raum projizieren
        vec_test = (Qm @ y_t)[:D]
        rho_test = vec_test.reshape(d, d, order='F')
        
        # Die physikalische Dichtematrix muss eine reell-positive Spur haben!
        # Jede imaginäre Abweichung davon ist exakt die verlorene Simulator-Phase.
        phase = np.angle(np.trace(rho_test))
        
        # Wir drehen die falsche Phase exakt aus dem Vektor heraus
        y_t *= np.exp(-1j * phase)
        # =====================================================================
        
        current_scale = scale * (s ** t) * np.sqrt(p_tot)
        vec_rho = (Qm @ y_t * current_scale)[:D]
        pops[t, :] = np.real(np.diag(vec_rho.reshape(d, d, order='F')))
        
    return pops, qc, dict(m=m, n_qubits=n_qubits, s=s, p_success=p_success_last, norm_Hm=float(np.linalg.norm(Hm, 2)))



def propagate_memory_classical(E, X0, n_steps, m, d):
    """
    Klassische Referenz-Propagation im Krylov-Unterraum ohne Quantenschaltkreis.
    
    1. Berechnet die Arnoldi-Kompression (H_m, Q_m).
    2. Propagiert den komprimierten Vektor direkt: y_t = (H_m)^t e_1.
    3. Rekonstruiert den physikalischen Zustand: X_t = ||X0|| * Q_m @ y_t.
    """
    D = d * d
    
    # 1. Arnoldi-Zerlegung
    Hm, Qm = arnoldi(E, X0, m)
    m_actual = Hm.shape[0]
    
    scale = np.linalg.norm(X0)
    pops = np.zeros((n_steps + 1, d))
    
    # Startvektor im Krylov-Raum e_1 = (1, 0, ..., 0)^T
    y_t = np.zeros(m_actual, dtype=complex)
    y_t[0] = 1.0
    
    # t = 0
    vec_rho0 = (Qm @ y_t * scale)[:D]
    pops[0, :] = np.real(np.diag(vec_rho0.reshape(d, d, order='F')))
    
    # 2. Direkte Propagation im m-dimensionalen Raum
    for t in range(1, n_steps + 1):
        y_t = Hm @ y_t
        
        # 3. Rückprojektion in den physikalischen Raum & Diagonale auslesen
        vec_rho = (Qm @ y_t * scale)[:D]
        pops[t, :] = np.real(np.diag(vec_rho.reshape(d, d, order='F')))
        
    return pops, None, dict(m=m_actual, norm_Hm=float(np.linalg.norm(Hm, 2)))