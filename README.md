# HyPhysML

Reference implementation for:

> Burukanli, M. *Physics-guided hybrid machine learning predicts contamination-induced flashover voltage of outdoor insulators.* Submitted to *Scientific Reports* (revised version).

`hyphysml.py` reproduces every quantitative result, figure and table of the revised manuscript and its Supplementary Information:

- the 14-model benchmark over 10 random seeds, with Optuna-TPE hyperparameter optimization **nested inside the training set of each seed**;
- the out-of-fold stacking ensemble HyPhysML and its monotone-constrained variant HyPhysML-MC;
- the auxiliary Obenaus regression and the prediction-level monotonicity audit;
- the statistical tests and bootstrap confidence intervals;
- SHAP and permutation importance of the complete stack;
- the noise-sensitivity, learning-curve, leave-one-level-out (extrapolation) and ablation studies;
- a data-level analysis of the measured flashover voltage.

## Version history

| Version | File | Used for |
| --- | --- | --- |
| **2 (current)** | `hyphysml.py` | Revised manuscript (second review round) |
| 1 | `legacy/hyphysml_v1_original_submission.py` | Original submission. Kept for transparency only; do not use. |

Version 2 corrects the following problems of version 1, identified during peer review:

1. **Hyperparameter-optimization leakage.** In version 1 a single tuning subset (seed 999, 80% of the records) was shared by all evaluation seeds, and 79–82% of every test set belonged to it. Version 2 repeats the search inside each training set, so no test record is used for tuning. The overlap of the old protocol is reported in `results_published/tables/R2-1_old_HPO_test_overlap.csv`.
2. **Noise analysis.** Version 1 perturbed only the raw column of the feature matrix; the derived features (log_SDD, env_stress, …) kept their clean values. It also ran a single repetition. Version 2 perturbs the raw input, clips it to its physical range and rebuilds all 25 features, with 5 repetitions × 10 seeds.
3. **Learning curve.** Version 1 used an unshuffled 5-fold split on rows ordered by insulator type. Version 2 uses the main evaluation protocol.
4. **SHAP and permutation importance.** In version 1 SHAP was computed for the XGBoost base learner. Version 2 computes both for the complete stack, using φ_stack = Σ w_b φ_b.
5. **Bootstrap intervals.** Version 2 reports the CI of the 10-seed mean and the CI of a single split separately.
6. **Model settings.**
   - The KNN base learner now uses the tuned settings; version 1 used a fixed k = 3.
   - Scalers are fitted inside a pipeline during tuning.
   - The meta-learner intercept is stored and reported.
   - Meta-weights are collected for every seed.

The headline accuracy is practically unchanged by these corrections (HyPhysML R² = 0.9890 ± 0.0007 over 10 seeds).

## Dataset

The data are not redistributed here. Download the open-access dataset and place it in `data/`:

- **Flashover voltage dataset for polymeric insulators under environmental conditions**, Mendeley Data — <https://doi.org/10.17632/8r7k4cgkg8.1>
- Required file: `FOV dataset.xlsx`, sheet `Whole samples` (6,656 records, 4 insulator types)

Alternatively point the script elsewhere with `FOV_DATA=/path/to/FOV dataset.xlsx`.

## Installation

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The published results were produced with the versions in `results_published/environment_versions.json`:
Python 3.13.15, scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, Optuna 5.0.0, SHAP 0.52.0, SciPy 1.16.3, NumPy 2.1.3 and pandas 2.2.3.

## Running

```bash
python hyphysml.py                         # full run, outputs in results/
HYPHYSML_FAST=1 python hyphysml.py         # smoke test: 2 seeds, 2 trials (minutes, not the paper results)
```

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FOV_DATA` | `data/FOV dataset.xlsx` | Dataset path |
| `FOV_OUT` | `results/` | Output folder (checkpoints, tables, figures) |
| `HYPHYSML_N_OPTUNA` | `200` | Optuna trials per model and seed |
| `HYPHYSML_SEEDS` | all 10 | Run a subset of seeds, e.g. `42,7,13` |
| `HYPHYSML_CV_JOBS` | `min(5, cores)` | Parallel CV folds during tuning |
| `HYPHYSML_GPU` | `0` | `1` runs XGBoost on CUDA |
| `HYPHYSML_EXTRAPOLATION` | `1` | Leave-one-level-out tests (about 1 h) |
| `HYPHYSML_LC_DIAGNOSTIC` | `1` | Reproduces the version-1 learning curve for comparison (about 30 min) |

**Run time.** The nested search (11 models × 200 trials × 10 seeds) is the expensive part. On a Google Colab A100 runtime (12 vCPUs, XGBoost on the GPU, two processes in parallel) the full run took about one day.

**Checkpoints.** Every finished hyperparameter search and every finished seed is checkpointed in `results/checkpoints/`, so an interrupted run resumes where it stopped. Several processes with different `HYPHYSML_SEEDS` may share one output folder; the final tables and figures are written once all ten seeds are present.

`notebooks/HyPhysML_Colab.ipynb` runs the pipeline on Google Colab with Drive-backed checkpoints and splits the seeds over parallel processes automatically.

## Output map

`results/figures/` and `results/tables/` contain the items of the revised paper.

| Output | Paper item |
| --- | --- |
| `Tab2_results_summary_all_models.csv` | Table 2 (and HyPhysML-MC) |
| `Tab3_statistical_tests.csv`, `Tab3_friedman.csv` | Table 3 |
| `Tab4_physics_coefficients.csv` | Table 4 |
| `Fig1_predicted_vs_actual_seed42.png` | Figure 1 |
| `Fig2_shap_bar_full_stack.png` | Figure 2 |
| `Fig3_noise_raw_inputs_propagated.png` | Figure 3 |
| `FigS01`–`FigS07` | Supplementary Figs. S1–S7 (exploratory analysis) |
| `FigR1_data_response_by_type.png`, `R1-1_*.csv` | Supplementary Fig. S8, Table S1 |
| `FigS21_mean_r2_seedlevel_ci.png` | Supplementary Fig. S9 |
| `FigS08_model_comparison_boxplots_R2_RMSE_MAPE.png` | Supplementary Fig. S10 |
| `FigS18_residuals.png`, `TabS6_residuals.csv` | Supplementary Fig. S11, Table S2 |
| `FigS09_all_models_pred_vs_actual_seed42.png` | Supplementary Fig. S12 |
| `FigS10_permutation_importance_full_stack.png` | Supplementary Fig. S13 |
| `FigS11_shap_beeswarm_full_stack.png` | Supplementary Fig. S14 |
| `FigS12_meta_weights_10seeds.png`, `TabS12*_*.csv` | Supplementary Fig. S15, Table S3 |
| `FigS13_physics_coefficients_auxiliary.png` | Supplementary Fig. S16 |
| `FigS22_monotonicity_audit.png`, `R2-2_*.csv` | Supplementary Fig. S17, Table S4 |
| `FigS20_raw_input_response_curves.png` | Supplementary Fig. S18 |
| `TabS1_noise_*.csv`, `FigS_noise_single_column_old_protocol.png` | Supplementary Table S5, Fig. S19 |
| `FigS14_vif.png`, `TabS2_vif.csv` | Supplementary Fig. S20, Table S6 |
| `FigS15_train_vs_test_r2.png` | Supplementary Fig. S21, Table S7 |
| `results_all_seeds.csv` (timing columns) | Supplementary Table S8 |
| `FigS17_per_type_performance.png`, `TabS5_*.csv` | Supplementary Fig. S22, Table S9 |
| `FigS19_learning_curve_main_protocol.png`, `TabS6b_*.csv`, `R2-4_*.csv` | Supplementary Fig. S23, Table S10 |
| `TabS7_ablation.csv` | Supplementary Table S11 |
| `FigS23_extrapolation_leave_one_level_out.png`, `R1-3_*.csv` | Supplementary Fig. S24, Table S12 |
| `R2-1_old_HPO_test_overlap.csv` | Supplementary Table S13 |
| `TabS8_hyperparameters_*.csv`, `TabS8b_preprocessing.csv` | Supplementary Tables S14–S15 |
| `split_indices_all_seeds.csv.gz` | Train/test indices of all 10 seeds |
| `revision_key_numbers.md` | All numbers quoted in the text |

## Published results

`results_published/` contains the outputs of the run reported in the paper:

- `tables/` — all CSV tables, including the split indices of all 10 seeds and the indices of the version-1 tuning subset;
- `hpo_selected_settings/` — the selected hyperparameters and inner-CV scores of every model and seed;
- `revision_key_numbers.md` — all numbers quoted in the text;
- `environment_versions.json` — the package versions used.

`paper/` contains three helper scripts used to build the manuscript files from these outputs:

- the architecture figure;
- the LaTeX tables of the Supplementary Information;
- the unmarked manuscript version.

## Notes on reproducibility

- **Determinism.** All seeds are fixed. Small deviations in the last digit can still occur between package versions, BLAS builds and CPU/GPU execution; XGBoost on CUDA is not bit-identical to CPU execution.
- **Timing.** Training times depend strongly on hardware and on concurrent processes.
- **Wilcoxon tests.** The comparisons are one-sided (upper-tailed), matching the directional hypothesis H1. For n = 10 paired seeds the exact test returns its floor, P = 2⁻¹⁰ = 0.000977, whenever the reference model wins on every seed; the two-sided value is 0.001953.
- **Obenaus regression.** The Obenaus regression is an auxiliary model that does not constrain HyPhysML. Physical consistency of the predictions is tested by the monotonicity audit and enforced only in HyPhysML-MC.

## Citation

If you use this code, please cite the manuscript above (see `CITATION.cff`) and the dataset:

> Larzeh Paydar Azerbaijan Science Incorporation. Flashover voltage dataset for polymeric insulators under environmental conditions. *Mendeley Data* <https://doi.org/10.17632/8r7k4cgkg8.1> (2026).

## License

Released under the MIT License; see `LICENSE`.

## Contact

Mehmet Burukanli — Department of Computer Engineering, Faculty of Engineering and Architecture, Bitlis Eren University, Bitlis 13000, Turkey — <mburukanli@beu.edu.tr> — ORCID [0000-0003-4459-0455](https://orcid.org/0000-0003-4459-0455)
