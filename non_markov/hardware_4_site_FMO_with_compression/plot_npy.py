"""Plottet die gespeicherten QPU-Zaehlraten direkt -- ohne Aer.

Warum eigene Datei: Zelle 11 des Notebooks simuliert zum Vergleich jeden
Schaltkreis in Aer, und bei 25 Zeitschritten hat der letzte 27 Qubits, also
2 GB Statevector.  Das dauert ~10 min.  Zum blossen Anschauen der bereits
gemessenen Daten braucht man das nicht: Gitter bauen, analyze, fertig -- ein
paar Sekunden.

Aufruf:  python plot_npy.py
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hardware as hw

LAEUFE = [('ibm_kingston_kry_200fs_mit_reset.npy',   'mit Reset (Mid-Circuit)'),
          ('ibm_kingston_kry_200fs_ohne_reset.npy',  'aufgeschobene Messung'),
          ('ibm_kingston_kry_500fs_ohne_reset.npy',  'aufgeschoben, 500 fs')]
DT, M, LAM, DELTA, PAIRS = 20.0, 4, 35.0, 10.0, [(0, 1)]
NMIN = 50                       # darunter ist die Statistik wertlos

mod, rho0 = hw.make_model(4, lam=LAM)
fig, ax = plt.subplots(2, len(LAEUFE), figsize=(5.0 * len(LAEUFE), 7.2))

for s, (datei, titel) in enumerate(LAEUFE):
    c = list(np.load(datei, allow_pickle=True))
    # t_max und Kanalzahl aus der Datei selbst ablesen -- Bitlaenge = t + n_sys
    n_sys = 2
    tmax = max(len(list(d)[0]) for d in c) - n_sys
    kanaele = len(c) // tmax
    floor = (kanaele == 7)
    times = list(range(1, tmax + 1))

    g = hw.build_hardware_grid(n_sites=4, m=M, dt_fs=DT, delta=DELTA,
                               model=mod, rho0=rho0, verbose=False)
    jobs = hw.hardware_circuits(g, times, PAIRS, method='svd',
                                defer=True, floor=floor)
    assert len(jobs) == len(c), f"{datei}: {len(jobs)} Jobs, {len(c)} Zaehlraten"
    rho, err, info = hw.analyze(g, jobs, c, pairs=PAIRS, verbose=False)

    ts, ref, tr = info['t_index'], info['reference'], info['rho_tr']
    T = ts * DT
    t_ex = np.arange(tmax + 1) * DT
    nacc = info['p_success'] * sum(c[0].values())
    gut = nacc >= NMIN                      # statistisch belastbar
    cols = plt.cm.tab10(np.arange(4))

    for j in range(4):                      # --- Populationen
        ax[0, s].plot(t_ex, np.real(ref[:, j, j]), color=cols[j], lw=1.6,
                      label=f'Site {j+1}')
        ax[0, s].errorbar(T[gut], np.real(tr[gut, j, j]), yerr=err[gut, j, j],
                          fmt='s', ms=5, capsize=3, color=cols[j], lw=1.1)
        if (~gut).any():                    # duenne Statistik: blass und hohl
            ax[0, s].plot(T[~gut], np.real(tr[~gut, j, j]), 's', ms=5,
                          mfc='white', color=cols[j], alpha=.35)
    ax[0, s].set_title(f'{titel}\n{len(c)} Kreise, t=1..{tmax}', fontsize=10)
    ax[0, s].set_ylabel('Population' if s == 0 else '')
    ax[0, s].set_ylim(-0.05, 1.05)
    if s == 0:
        ax[0, s].legend(ncol=2, fontsize=7)

    a, b = PAIRS[0]                         # --- Kohaerenz
    for teil, lab, cc, ee in ((np.real, r'\mathrm{Re}', 'C0', err),
                              (np.imag, r'\mathrm{Im}', 'C2',
                               info['rho_err_im'])):
        ax[1, s].plot(t_ex, teil(ref[:, a, b]), lw=1.6, color=cc,
                      label=rf'${lab}\,\rho_{{{a+1}{b+1}}}$')
        ax[1, s].errorbar(T[gut], teil(tr[gut, a, b]), yerr=ee[gut, a, b],
                          fmt='s', ms=5, capsize=3, color=cc, lw=1.1)
        if (~gut).any():
            ax[1, s].plot(T[~gut], teil(tr[~gut, a, b]), 's', ms=5,
                          mfc='white', color=cc, alpha=.35)
    ax[1, s].axhline(0, color='0.85', lw=.8, zorder=0)
    ax[1, s].set_ylabel(r'$\rho_{12}$' if s == 0 else '')
    if s == 0:
        ax[1, s].legend(fontsize=7)

    d = [np.abs(np.real(np.diag(tr[k])) - np.real(np.diag(ref[ts[k]]))).max()
         for k in range(len(ts))]
    rms = np.sqrt(np.mean([d[k]**2 for k in range(len(ts)) if gut[k]]))
    print(f"  {datei:38s} t=1..{tmax:2d} | RMS (N_acc>={NMIN}) {rms:.4f}"
          f" | {int((~gut).sum())} Punkte verworfen")

for A in ax.ravel():
    A.set_xlabel('t [fs]'); A.grid(alpha=.25)
fig.suptitle('QPU gegen qutip HEOMSolver (DrudeLorentzPade, ein Bad je Site, '
             f"max_depth={g['depth']}, Nk={g['Nk']})\n"
             'Linie: qutip HEOMSolver   Quadrat: QPU spurnormiert   '
             f'hohl+blass: N_acc < {NMIN} (statistisch wertlos)', fontsize=11)
fig.tight_layout()
fig.savefig('pictures/npy_runs.png', dpi=150, bbox_inches='tight')
print("\n  gespeichert: pictures/npy_runs.png")
