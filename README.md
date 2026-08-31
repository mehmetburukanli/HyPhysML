# HyPhysML

Reference implementation for:

> Burukanli, M. *HyPhysML: physics-informed hybrid machine learning for predicting contamination-induced flashover voltage of outdoor insulators.* Submitted to *Scientific Reports*.

The script reproduces every quantitative result, figure and table reported in the manuscript and its Supplementary Information: the 14-model benchmark over 10 random seeds, the Optuna-TPE hyperparameter search, the out-of-fold stacking ensemble, the Obenaus physics-validation layer, the statistical tests, SHAP and permutation importance, the noise-sensitivity study and the ablation analysis.

## Dataset

The data are not redistributed here. Download the open-access dataset and place it in `data/`:

- **Flashover voltage dataset for polymeric insulators under environmental conditions**, Mendeley Data — <https://doi.org/10.17632/8r7k4cgkg8.1>
- Required file: `FOV dataset.xlsx`, sheet `Whole samples` (6,656 records, 4 insulator types)

```
data/FOV dataset.xlsx
```

Alternatively point the script elsewhere with `FOV_DATA=/path/to/FOV dataset.xlsx`.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python hyphysml.py
```

Outputs are written to `results/` (override with `FOV_OUT`). A full run performs 200 Optuna trials for each of 11 tunable models and then trains 14 models across 10 seeds: roughly **50 minutes on a single GPU** or **2–3 hours on CPU**.

For a fast structural check that skips the hyperparameter search, set `SKIP_HPO = True` near the top of the script. This finishes in a few minutes but reproduces the reported metrics only approximately, because the models then use default rather than optimized hyperparameters.

## Configuration

The evaluation protocol is defined in one block near the top of `hyphysml.py`:

| Variable | Value | Meaning |
| --- | --- | --- |
| `SEEDS` | 10 seeds | Train/test splits used for the multi-seed evaluation |
| `TEST_SIZE` | 0.20 | Test fraction, stratified by insulator type |
| `N_FOLDS` | 5 | Folds for out-of-fold stacking and cross-validation |
| `N_OPTUNA` | 200 | TPE trials per tunable model |
| `N_BOOT` | 1000 | Bootstrap replicates for the 95% confidence intervals |
| `SKIP_HPO` | `False` | `True` disables the hyperparameter search |

Hyperparameter optimization runs on an independent holdout (seed 999) that is disjoint from all 10 evaluation seeds, so no information reaches the test data.

## Output map

`results/` contains the figures, tables and CSV files behind the manuscript. The correspondence is:

| Output | Appears in the paper as |
| --- | --- |
| `fig_*_predicted_vs_actual.png` | Figure 1 |
| `fig_*_shap_bar.png` | Figure 2 |
| `fig_*_sensitivity_noise.png` | Figure 3 |
| `fig_*_eda_*.png` | Supplementary Figures S1–S7 |
| `fig_*_model_comparison_boxplot.png` | Supplementary Figure S8 |
| `fig_*_permutation_importance.png` | Supplementary Figure S10 |
| `fig_*_shap_beeswarm.png` | Supplementary Figure S11 |
| `fig_*_hyphysml_meta_weights.png` | Supplementary Figure S12 |
| `fig_*_physics_coefficients.png` | Supplementary Figure S13 |
| `fig_*_vif_multicollinearity.png` | Supplementary Figure S14 |
| `fig_*_r2_bootstrap_ci.png` | Supplementary Figure S21 |
| `results_summary.csv` | Table 1 |
| `statistical_tests.csv` | Table 2 |
| `sensitivity_analysis.csv` | Supplementary Table S1 |
| `train_test_r2.csv`, `computation_time.csv` | Supplementary Tables S3, S4 |
| `vif_analysis.csv` | Supplementary Table S2 |

Figure files are numbered in the order the script writes them; the architecture diagram (Figure 4 of the paper) is drawn separately and is not produced by this script.

## Notes on reproducibility

All seeds are fixed and the protocol is deterministic given the package versions in `requirements.txt`. Small deviations in the last reported digit can still occur across BLAS builds, CPU/GPU execution and library patch releases; the ranking of the 14 models and the outcome of every statistical test are stable under such variation.

The Wilcoxon signed-rank comparisons are one-sided (upper-tailed), matching the directional hypothesis H1 stated in the manuscript. For n = 10 paired seeds the exact test returns its floor, P = 2⁻¹⁰ = 0.000977, whenever the reference model wins on every seed. The corresponding two-sided values are 0.001953 and remain below the Bonferroni-corrected threshold of 0.00385.

The physics layer validates six sign constraints on the Obenaus log-linear coefficients. The admissible magnitude band for the contamination exponent, `THEORETICAL["log_SDD"] = (-0.35, -0.10)`, is fixed in the script before the coefficients are estimated.

## Citation

If you use this code, please cite the manuscript above and the dataset:

> Larzeh Paydar Azerbaijan Science Incorporation. Flashover voltage dataset for polymeric insulators under environmental conditions. *Mendeley Data* <https://doi.org/10.17632/8r7k4cgkg8.1> (2026).

## License

Released under the MIT License; see `LICENSE`.

## Contact

Mehmet Burukanli — Department of Computer Engineering, Faculty of Engineering and Architecture, Bitlis Eren University, Bitlis 13000, Turkey — <mburukanli@beu.edu.tr> — ORCID [0000-0003-4459-0455](https://orcid.org/0000-0003-4459-0455)
