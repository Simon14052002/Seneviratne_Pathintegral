"""
Transfer-tensor (SMatPI-style) consistency check, cf. refs 70-73 of the
paper: from the exact short-time maps L_1..L_N (N = memory steps) extract
discrete transfer tensors
    T_1 = L_1,   T_n = L_n - sum_{m=1}^{n-1} T_m L_{n-m}
and propagate beyond the memory time via
    L_n = sum_{m=1}^{N} T_m L_{n-m} .
If the influence-functional memory truncation at N steps is converged,
this reproduces the directly computed TEMPO maps at all later times.
"""

import numpy as np


def transfer_tensors(maps, n_mem):
    """maps: [L_0 = 1, L_1, ...];  returns [T_1 .. T_n_mem]."""
    T = []
    for n in range(1, n_mem + 1):
        Tn = maps[n].copy()
        for m in range(1, n):
            Tn -= T[m - 1] @ maps[n - m]
        T.append(Tn)
    return T


def extend_maps(maps_short, T, n_total):
    """Extend the map sequence to n_total steps using transfer tensors."""
    maps = [m.copy() for m in maps_short]
    for n in range(len(maps), n_total + 1):
        L = sum(T[m - 1] @ maps[n - m] for m in range(1, len(T) + 1))
        maps.append(L)
    return maps
