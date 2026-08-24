# Continuous-grid open-system dynamics — paper draft

`continuous_grid.tex` + `refs.bib` → `continuous_grid.pdf` (revtex4-2, PRX style).

Build:

    tectonic -X compile continuous_grid.tex

## Where the content comes from

| Section | Source |
| --- | --- |
| Sec. II (Markovian, Stinespring) | `Markov/Theory_markov.ipynb`, `Markov/single_grid_markov.ipynb` |
| Sec. III (non-Markovian, contraction gauge) | `non-Markov/Theorie_durchgaengiges_grid_compakt.ipynb`, `heom_gauge.py`, `gauge_circuit.py` |
| Sec. IV (results) | `non-Markov/main_durchgaengiges_grid.ipynb` |
| Sec. V–VI (cost, limitations) | `non-Markov/Kritische_Pruefung.ipynb` |

Figures in `figures/` are copies of `non-Markov/pictures/`.

## Before submitting — please check

* **Bibliography.** The entries were written from memory and the bibliographic
  details (volume/page/year) have **not** been verified against the originals.
  `Seneviratne2024` and `Lambert2023` in particular should be checked.
* **Author/affiliation** are placeholders.
* Two numbers in the paper come from earlier runs rather than from the
  currently checked-in notebooks: the Markovian accuracy (1.7e-15 vs the
  step-by-step reference) is quoted in the text of Sec. II only qualitatively,
  and the CNOT counts (94/423/1783 at 4/5/6 qubits) were measured in
  `Kritische_Pruefung.ipynb` before the sparse/Cholesky code was reverted.
* The central claim is deliberately narrow — *classical cost independent of the
  number of time steps*, not "most of the work is on the quantum computer".
  Sec. VI ("Where the work happens") states the qualification that the Arnoldi
  basis itself spans the first `m` steps; please keep that qualification if the
  text is shortened.
