# Putting open-system dynamics on a quantum grid — paper

arXiv-ready manuscript (APS `revtex4-2`, two-column) on the memory-window /
transfer-tensor grid algorithm for simulating open-quantum-system dynamics on
quantum computers.

## Idea
Turn a dynamical map into ONE dilated unitary and let the quantum computer
generate the whole trajectory by re-applying that one circuit (read the state
out, feed it back). A Markovian (Lindblad) map is a semigroup, so one one-step
operator suffices. A non-Markovian map (HEOM, path integral) is not a semigroup
and a system-only channel cannot hold memory, so the memory is put on the
register: the dynamics is a discrete convolution over transfer tensors that
decay within a finite window, packed into one companion operator whose Sz.-Nagy
dilation is the reused circuit. The classical cost covers only the memory window.

## Contents
- `main.tex` — the paper (revtex4-2, `pra`, two-column). Figs 2/3 (circuit,
  workflow) are inline TikZ/`quantikz`.
- `references.bib` — bibliography (apsrev4-2 / BibTeX).
- `main.pdf` — compiled paper (7 pages incl. refs).
- `figures/` — the three data figures + the generators.

## Figures (`figures/`)
- `fig_grid.pdf` — the key result: 3 maps (Lindblad, HEOM, path integral) on one
  reused circuit vs the exact solvers, with the memory-window markers.
- `fig_memory.pdf` — transfer-tensor decay and accuracy vs memory window.
- `fig_deviation.pdf` — grid-vs-solver error, including the K=1 (memory
  discarded) failure at order one.
- `fig_validity.pdf` — validity check: companion-operator spectrum in the unit
  disk (Schur stability) and trace preservation of the propagated state.
- `proposal_reuse.png` — the state-reuse concept (paper Fig. 1), from the proposal.
- `grid_ref.py` — pure-numpy grid reference (transfer tensors, companion
  operator, Sz.-Nagy dilation, propagation). Reproduces the reused Qiskit circuit.
- `make_enhanced_figures.py` — regenerates the three data figures from the saved
  dynamical maps in `../../maps_to_grid/normal/data/`.

## Style / structure
- Three authors (S. Mader, A. Bhartiya, T. Kramer).
- Body prose avoids colons and dashes.
- Section II gives the physical model (FMO H, Drude spectral density J(w), bath
  correlation C(t)) and the defining formulas of all three maps (Lindblad,
  HEOM hierarchy, TEMPO influence functional).
- Four appendices: A (vectorization + Liouvillian + Choi-Kraus), B (construction
  of the three maps: secular Redfield, HEOM, path integral), C (transfer tensors,
  companion operator, and stability), D (comparison with Seneviratne / Hu).

## Build
```bash
python figures/make_enhanced_figures.py    # regenerate the three data figures
tectonic main.tex                          # compile (pulls LaTeX packages)
```
