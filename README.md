# Multifractal Large Deviations in Quantum Entanglement

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23193512.svg)](https://doi.org/10.5281/zenodo.23193512)

Code, data, figures and manuscript for

> **R. Chen**, *Multifractal Large Deviations in Quantum Entanglement: Ideal Fermi-Gas Mapping, Markov Tensor
> Cascades, and Quartic Casimir Symmetry* (2026). DOI: [10.5281/zenodo.23193512](https://doi.org/10.5281/zenodo.23193512)

## Summary

The thermodynamic formalism of multifractals turns Rényi entropies into generalised dimensions and the refined
Rényi entropy into the singularity spectrum f(α). The paper uses it to separate two notions that are often
conflated: multifractality of single-particle wave functions and multifractality of many-body entanglement spectra.

- **Ideal Fermi-gas theorem.** For Gaussian fermionic states the refined Rényi entropy equals the entropy of an
  ideal Fermi gas at temperature 1/q whose levels are the entanglement energies. Hence f_ent ≥ 0 is concave and
  is fixed by the density of entanglement energies; wave-function multifractality enters only through
  inter-eigenstate overlaps. Checked to 5×10⁻¹⁰ on critical power-law random banded matrices.
- **Markov tensor cascades.** With layer-to-layer correlations modelled by a Markov chain, τ(q) is set by a Perron
  root, its correction is linear in λ₂, it is analytic for λ₂ < 1, and a Maxwell (first-order) transition appears
  only at λ₂ = 1.
- **Casimir structure at the quantum Hall transition.** Weyl invariance makes Δ_q a function of u = q(1−q); the
  measured non-parabolicity is the u² coefficient, c₂ = −2b₁. It appears at four loops in the 2+ε expansion with the
  observed sign and excludes theories in which the multifractal operators are current-algebra (Sugawara) primaries.
- **Gauss–Bonnet cosmic branes (d = 4).** The second Taylor coefficient of τ(q) tracks C_T/a to 10⁻³; the refined
  Rényi entropy turns negative at finite q for positive Gauss–Bonnet coupling; no correspondence with the quantum
  Hall coefficient b₁ is warranted.

## Repository structure

| Path | Contents |
|---|---|
| `paper/` | LaTeX source and compiled PDF of the manuscript |
| `code/multifractal_entanglement.py` | Thermodynamic dictionary for a two-scale Cantor measure read as an entanglement spectrum (Fig. 1) |
| `code/frontier2_deepdive.py` | PRBM Fermi-gas check, Markov cascades, quantum Hall Casimir form, Gauss–Bonnet branes (Fig. 2, Table 1) |
| `figures/` | Figures as vector PDF (used by the paper) and PNG |
| `data/` | Numerical data behind the figures and Table 1, as CSV (metadata in `#` header lines) |
| `results/` | Console output of both scripts (the numbers quoted in the paper) |

## Reproducing the results

Requirements: Python ≥ 3.10 and the packages in `requirements.txt`
(tested with Python 3.12.4, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.8.4).

```bash
pip install -r requirements.txt
python code/multifractal_entanglement.py          # Fig. 1; options: --p 0.6 0.4 --r 0.25 0.4 --eps 1e-80
python code/frontier2_deepdive.py                 # Fig. 2 and Table 1 (a few minutes: PRBM diagonalisations)
```

Run from the repository root; figures go to `figures/` and data to `data/` (override with `MQE_FIG_DIR`,
`MQE_DATA_DIR`). Add `--show` to display the figures. The PRBM ensemble uses a fixed random seed, so the
output is reproducible.

| Data file | Content | Paper |
|---|---|---|
| `cantor_tau_D_alpha_f.csv`, `cantor_exact_spectrum_check.csv`, `cantor_eigenvalue_histogram.csv` | τ, D_q, α, f; exact-spectrum checks | §2.1, Fig. 1 |
| `quantum_hall_spectrum.csv` | f(α) from Evers–Mildenberger–Mirlin exponents | Fig. 1, §4 |
| `prbm_wavefunction_spectrum.csv`, `prbm_entanglement_spectrum.csv` | PRBM spectra | §2.3, Fig. 2 |
| `markov_cascade.csv` | τ, α, f for λ₂ = 0, 0.9, 0.99, 1 | §3, Fig. 2 |
| `gauss_bonnet_t2.csv`, `gauss_bonnet_branes.csv` | t₂ ratios; brane spectra | §5, Table 1, Fig. 2 |

To rebuild the paper (pdfLaTeX, two passes):

```bash
cd paper
pdflatex Chen_2026_Multifractal_Quantum_Entanglement.tex
pdflatex Chen_2026_Multifractal_Quantum_Entanglement.tex
```

## Citation

```bibtex
@misc{Chen2026MultifractalEntanglement,
  author = {Chen, Ruqing},
  title  = {Multifractal Large Deviations in Quantum Entanglement: Ideal Fermi-Gas Mapping,
            Markov Tensor Cascades, and Quartic Casimir Symmetry},
  year   = {2026},
  doi    = {10.5281/zenodo.23193512},
  url    = {https://doi.org/10.5281/zenodo.23193512}
}
```

## License

- **Code** (`code/`): [MIT License](LICENSE)
- **Manuscript, figures, data and results** (`paper/`, `figures/`, `data/`, `results/`):
  [CC BY 4.0](LICENSE-CC-BY-4.0.md)

## Contact

Ruqing Chen — GUT Geoservice Inc., Montreal — ruqing@hotmail.com
