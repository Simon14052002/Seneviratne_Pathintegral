"""
Independent exact reference for the 4-site FMO dynamics: HEOM (qutip/BoFiN)
with Drude-Lorentz baths (Pade decomposition), same model as the paper:
lam = 35 cm^-1, gamma = 106.18 cm^-1, T = 300 K, one local bath per site.

Usage:  python3 heom_check.py <Nk> <max_depth> <tag>
"""

import sys
import time
import numpy as np
import qutip as qt
from qutip.solver.heom import HEOMSolver, DrudeLorentzPadeBath

from pathintegral_map import FS_TO_CM, KB_CM

Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 3
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
tag = sys.argv[3] if len(sys.argv) > 3 else f"Nk{Nk}_d{max_depth}"

H = np.array([[12375.0, -87.7,   5.5,  -5.9],
              [-87.7, 12495.0,  30.8,   8.2],
              [5.5,      30.8, 12175.0, -53.4],
              [-5.9,      8.2, -53.4, 12285.0]])
H = H - np.trace(H) / 4 * np.eye(4)
LAM, GAM, TEMP_K = 35.0, 106.18, 300.0
T_ENERGY = KB_CM * TEMP_K                     # temperature in cm^-1

t_list_fs = np.linspace(0.0, 1000.0, 201)
t_list = t_list_fs * FS_TO_CM                 # time in cm units

Hq = qt.Qobj(H)
baths = []
for site in range(4):
    Q = qt.basis(4, site) * qt.basis(4, site).dag()
    baths.append(DrudeLorentzPadeBath(Q, lam=LAM, gamma=GAM, T=T_ENERGY,
                                      Nk=Nk))

print(f"[{tag}] HEOM: Nk = {Nk} (Pade), max_depth = {max_depth}")
t0 = time.time()
solver = HEOMSolver(Hq, baths, max_depth=max_depth,
                    options={"nsteps": 15000, "rtol": 1e-10, "atol": 1e-12})
rho0 = qt.basis(4, 0) * qt.basis(4, 0).dag()
result = solver.run(rho0, t_list)
print(f"[{tag}] done in {time.time() - t0:.1f} s")

pops = np.array([np.real(s.diag()) for s in result.states])
np.savez(f"heom_{tag}.npz", t_list_fs=t_list_fs, pops=pops,
         Nk=Nk, max_depth=max_depth)
print(f"[{tag}] saved heom_{tag}.npz ; populations at 1000 fs:", pops[-1])
