"""Builds the LaTeX tables of the revised Supplementary Information directly from
the CSV outputs of hyphysml_revision.py, so that no number is copied by hand."""
import os
import pandas as pd
import numpy as np

T = os.path.join("results_revision_final", "tables")
OUT = os.path.join("revision3_latex", "supp_tables")
os.makedirs(OUT, exist_ok=True)
rd = lambda f: pd.read_csv(os.path.join(T, f))

def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

def write(name, body):
    BS = chr(92)
    body = body.replace(BS + "begin{table}[htbp]" + BS + "centering",
                        BS + "begin{table}[htbp]" + BS + "centering" + BS + "color{blue}")
    body = body.replace(BS + "begin{longtable}", "{" + BS + "color{blue}\n" + BS + "begin{longtable}")
    body = body.replace(BS + "end{longtable}", BS + "end{longtable}\n}")
    body = body.replace(" & -", " & $-$")
    body = body.replace(BS + "begin{tabular}", BS + "begin{adjustbox}{max width=" + BS + "linewidth}" + BS + "begin{tabular}")
    body = body.replace(BS + "end{tabular}", BS + "end{tabular}" + BS + "end{adjustbox}")
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body)
    print("wrote", name)

NAMES = {"GBR": "Gradient boosting", "HistGBR": "Histogram gradient boosting", "SVR": "Support vector regression",
         "MLP": "Multilayer perceptron", "RF": "Random forest", "Extra Trees": "Extra trees",
         "Decision Tree": "Decision tree", "KNN": "$k$-nearest neighbours", "Ridge": "Ridge regression"}
nm = lambda m: NAMES.get(m, m)

# ---- S1: data-level physics ---------------------------------------------------
mono = rd("R1-1_data_monotonicity.csv")
exp = rd("R1-1_SDD_exponent_summary.csv")
rows = "\n".join(f"{r.Factor} & {r.n_adjacent_steps:,} & {r.pct_steps_FOV_decreases:.2f} & {r.pct_cells_strictly_monotone:.2f} & {r.median_step_kV:.2f} \\\\"
                 for r in mono.itertuples())
lab = {"Sample": "Insulator type", "RH": "RH (\\%)", "Aging": "Aging (day)", "all": "All conditions"}
erows = []
for r in exp.itertuples():
    key = r[1]; by = r.by
    level = f"Smp\\_{key}" if by == "Sample" else ("---" if by == "all" else key)
    erows.append(f"{lab[by]} & {level} & {float(r.mean):.3f} & {float(r.std):.3f} & {float(r.min):.3f} & {float(r.max):.3f} \\\\")
write("tabS01_data_physics.tex", r"""\begin{table}[htbp]\centering
\caption{Physical behaviour of the measured flashover voltage (data only, no model). (a) Monotonicity of the measured FOV: share of adjacent level steps of one factor, with all other factors held at fixed design values, in which FOV decreases. For $J$ and $K$ only the 25 combinations with $K>0$ are used, because $J=1$ occurs only with $K=0$. (b) Empirical contamination exponent $b$ of $\mathrm{FOV}\propto\mathrm{SDD}^{\,b}$, obtained by a log--log fit over the four SDD levels for each of the 1,664 test conditions (type $\times$ RH $\times$ aging $\times$ $(J,K)$).}\label{stab:dataphys}
\small
\textbf{(a)}\par
\begin{tabular}{lcccc}
\toprule
Factor & Adjacent steps & FOV decreases (\%) & Strictly monotone cells (\%) & Median step (kV) \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}

\medskip
\textbf{(b)}\par
\begin{tabular}{llcccc}
\toprule
Grouping & Level & Mean $b$ & s.d. & Min & Max \\
\midrule
""" + "\n".join(erows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S2: residual statistics ---------------------------------------------------
res = rd("TabS6_residuals.csv")
r42, rall = res.iloc[0], res.iloc[1]
write("tabS02_residuals.tex", r"""\begin{table}[htbp]\centering
\caption{Residual statistics of HyPhysML ($y-\hat y$). The Shapiro--Wilk test of the pooled residuals uses a random subsample of 5,000 values (the limit of the test).}\label{stab:resid}
\small
\begin{tabular}{lcccc}
\toprule
Scope & $n$ & Mean (kV) & s.d. (kV) & Shapiro--Wilk $W$ ($P$) \\
\midrule
""" + f"Seed-42 test set & {int(r42.n):,} & {r42.mean_kV:.3f} & {r42.sd_kV:.3f} & {r42.Shapiro_W:.4f} ($<0.001$) \\\\\n"
      f"All 10 test sets pooled & {int(rall.n):,} & {rall.mean_kV:.3f} & {rall.sd_kV:.3f} & {rall.Shapiro_W:.4f} ($<0.001$) \\\\\n" + r"""\bottomrule
\end{tabular}
\end{table}
""")

# ---- S3: meta-learner ---------------------------------------------------------
mw = rd("TabS12_meta_weights_summary.csv").set_index("Unnamed: 0")
oof = rd("TabS12b_base_learner_oof_r2.csv")
cor = rd("TabS12c_oof_residual_correlation.csv").set_index("Unnamed: 0")
BL = ["XGBoost", "LightGBM", "GBR", "HistGBR", "RF", "Extra Trees", "KNN"]
rows = "\n".join(f"{nm(b)} & {mw.loc[b,'mean']:.3f} $\\pm$ {mw.loc[b,'std']:.3f} & [{mw.loc[b,'min']:.3f}, {mw.loc[b,'max']:.3f}] & {int(mw.loc[b,'n_negative_of_10'])} & {oof[b].mean():.4f} \\\\" for b in BL)
rows += f"\nIntercept $w_0$ (kV) & {mw.loc['intercept','mean']:.3f} $\\pm$ {mw.loc['intercept','std']:.3f} & [{mw.loc['intercept','min']:.3f}, {mw.loc['intercept','max']:.3f}] & {int(mw.loc['intercept','n_negative_of_10'])} & --- \\\\"
crow = "\n".join(nm(b) + " & " + " & ".join(f"{cor.loc[b, c]:.2f}" for c in BL) + r" \\" for b in BL)
write("tabS03_meta.tex", r"""\begin{table}[htbp]\centering
\caption{Ridge meta-learner of HyPhysML over the 10 evaluation seeds. (a) Weights $w_b$ and intercept $w_0$ (Eq.~8), with the out-of-fold $R^2$ of each base learner on the training data. (b) Correlation of the out-of-fold residuals of the base learners (mean over seeds).}\label{stab:meta}
\small
\textbf{(a)}\par
\begin{tabular}{lcccc}
\toprule
Base learner & Weight (mean $\pm$ s.d.) & [min, max] & Negative (of 10) & OOF $R^2$ \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}

\medskip
\textbf{(b)}\par
\footnotesize
\begin{tabular}{l""" + "c" * 7 + r"""}
\toprule
 & XGB & LGBM & GBR & HGBR & RF & ET & KNN \\
\midrule
""" + crow + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S4: monotonicity audit ---------------------------------------------------
au = rd("R2-2_monotonicity_audit_summary.csv")
rows = []
for v in ["SDD", "RH", "Aging", "J"]:
    for m in ["HyPhysML", "XGBoost", "HyPhysML-MC"]:
        r = au[(au.Model == m) & (au.Variable == v)].iloc[0]
        rows.append(f"{v if m == 'HyPhysML' else ''} & {m} & {r.pct_steps_increasing:.3f} & {r.pct_steps_increasing_gt_0_1kV:.3f} & {r.pct_points_any_violation:.2f} & {r.max_increase_kV:.2f} & {r.data_pct_steps_increasing if m == 'HyPhysML' else ''} \\\\")
    rows.append(r"\addlinespace")
write("tabS04_audit.tex", r"""\begin{table}[htbp]\centering
\caption{Prediction-level monotonicity audit (mean over the 10 seed-specific test sets). For every test record one raw variable is set in turn to each of its design levels, all 25 features are rebuilt and the model predicts; a violation is an adjacent level step at which the predicted FOV rises although physics expects it to fall. For $J$ only records with $K>0$ and the levels 1.5--15 are used. Max.\ rise: largest predicted increase over all steps and seeds (negative = no rise). The last column gives the same share for the measured data (Supplementary Table~S1).}\label{stab:audit}
\small
\begin{tabular}{llccccc}
\toprule
Variable & Model & Steps rising (\%) & Rising $>0.1$ kV (\%) & Records with $\ge$1 rise (\%) & Max.\ rise (kV) & Data (\%) \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S5: noise ----------------------------------------------------------------
nz = rd("TabS1_noise_summary.csv")
rows = []
for model in ["HyPhysML", "HyPhysML-MC"]:
    for mode, lab_ in [("raw_propagated", "raw input, features rebuilt"), ("single_column", "single column (old protocol)")]:
        if model == "HyPhysML-MC" and mode == "single_column":
            continue
        for f_ in ["SDD", "RH", "Aging"]:
            d = nz[(nz.Model == model) & (nz["mode"] == mode) & (nz.Feature == f_)].sort_values("Noise_pct")
            cells = " & ".join(f"{a:+.4f} ({b:.4f})" for a, b in zip(d.dR2_mean, d.dR2_sd))
            clip = d.pct_clipped.max()
            rows.append(f"{model} & {lab_} & {f_} & {cells} & {clip:.1f} \\\\")
        rows.append(r"\addlinespace")
write("tabS05_noise.tex", r"""\begin{table}[htbp]\centering
\caption{Noise sensitivity: change in test $R^2$ relative to the clean test set, mean (s.d.) over 10 seeds $\times$ 5 repetitions ($n=50$ per cell). Gaussian noise with s.d.\ equal to the stated percentage of the training-set s.d.\ of the raw variable. In the main protocol the raw value is perturbed, clipped to its physical range (SDD $\ge 0.001$ mg cm$^{-2}$; $1\le$ RH $\le 100$\%; aging $\ge 0$ day) and all derived features are rebuilt. The single-column protocol of the original submission perturbs only the raw column of the feature matrix and leaves every derived feature at its clean value; it is reported only to explain the earlier result. HyPhysML-MC uses raw inputs only, so both protocols coincide for it.}\label{stab:noise}
\footnotesize
\begin{tabular}{lllccccc}
\toprule
Model & Protocol & Feature & +5\% & +10\% & +15\% & +20\% & Clipped (\%) \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S6: VIF ------------------------------------------------------------------
vif = rd("TabS2_vif.csv")
def fv(v):
    return r"$\infty$" if v > 1e12 else (f"{v:,.0f}" if v >= 100 else f"{v:.2f}")
sev = {"High (>10)": r"High ($>$10)", "Moderate (5-10)": r"Moderate (5--10)", "Low (<5)": r"Low ($<$5)"}
rows = "\n".join(f"{esc(r.Feature)} & {fv(r.VIF)} & {sev[r.Severity]} \\\\" for r in vif.itertuples())
write("tabS06_vif.tex", r"""\begin{table}[htbp]\centering
\caption{Variance inflation factors of the 21 continuous features. CD, AD, CF and log\_CD take one value per insulator type and are therefore exactly collinear with each other and with the type indicators ($\mathrm{VIF}=\infty$).}\label{stab:vif}
\footnotesize
\begin{tabular}{lcl}
\toprule
Feature & VIF & Severity \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S7: train vs test, S8: time ----------------------------------------------
agg = rd("Tab2_results_summary_all_models.csv").set_index("Model")
allr = rd("results_all_seeds.csv")
ORDER = ["HyPhysML", "GBR", "XGBoost", "HistGBR", "LightGBM", "SVR", "MLP", "RF", "Extra Trees",
         "Decision Tree", "KNN", "Ridge", "Obenaus", "Rizk", "HyPhysML-MC"]
rows = "\n".join(f"{nm(m)} & {agg.loc[m,'R2_train_mean']:.4f} & {agg.loc[m,'R2_mean']:.4f} & {agg.loc[m,'Overfit_gap']:+.4f} \\\\" for m in ORDER)
write("tabS07_train_test.tex", r"""\begin{table}[htbp]\centering
\caption{Training and test $R^2$ (mean over 10 seeds) and their difference.}\label{stab:overfit}
\small
\begin{tabular}{lccc}
\toprule
Model & $R^2$ (train) & $R^2$ (test) & Difference \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
tm = allr.groupby("Model").agg(tmin=("Time_fit_s", "min"), tmed=("Time_fit_s", "median"),
                               pred=("Time_predict_ms_per_sample", "median"))
rows = "\n".join(f"{nm(m)} & {tm.loc[m,'tmin']:.2f} & {tm.loc[m,'tmed']:.2f} & {tm.loc[m,'pred']:.3f} \\\\"
                 for m in tm.sort_values("tmin").index if m in ORDER)
write("tabS08_time.tex", r"""\begin{table}[htbp]\centering
\caption{Training time per split (s) and prediction time (ms per record), excluding hyperparameter optimization. Measured on a Google Colab A100 runtime (12 virtual CPU cores; XGBoost on the GPU) while two evaluation processes ran concurrently, so wall-clock times vary strongly between seeds; the minimum over the 10 seeds is the best indication of the uncontended cost, and the median is given for completeness.}\label{stab:time}
\small
\begin{tabular}{lccc}
\toprule
Model & Minimum (s) & Median (s) & Prediction (ms/record, median) \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S9: per type ------------------------------------------------------------
pt = rd("TabS5_per_type_summary.csv")
rows = "\n".join(f"Smp\\_{r.Type[-1]} & {r.n_test} & {r.R2_mean:.4f} $\\pm$ {r.R2_sd:.4f} & {r.RMSE_mean:.3f} & {r.MAE_mean:.3f} & {r.MAPE_mean:.3f} \\\\" for r in pt.itertuples())
write("tabS09_per_type.tex", r"""\begin{table}[htbp]\centering
\caption{HyPhysML performance by insulator type, mean over the 10 seed-specific test sets (333 test records per type in every seed).}\label{stab:bytype}
\small
\begin{tabular}{lccccc}
\toprule
Insulator type & $n_\mathrm{test}$ & $R^2$ (mean $\pm$ s.d.) & RMSE (kV) & MAE (kV) & MAPE (\%) \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S10: learning curve + diagnostic -----------------------------------------
lc = rd("TabS6b_learning_curve_summary.csv")
rows = "\n".join(f"{r.frac:.0%} & {int(r.n_train):,} & {r.R2_train_mean:.4f} $\\pm$ {r.R2_train_sd:.4f} & {r.R2_test_mean:.4f} $\\pm$ {r.R2_test_sd:.4f} & {r.gap:.4f} \\\\".replace("%", r"\%") for r in lc.itertuples())
dg = rd("R2-4_learning_curve_diagnostic.csv")
dgs = dg.groupby("protocol").R2.agg(["mean", "std", "min", "max"])
lab_d = {k: ("Unshuffled 5-fold CV on type-ordered rows (original learning curve)" if k.startswith("unshuffled") else "Shuffled 5-fold CV") for k in dgs.index}
drows = "\n".join(f"{lab_d[k]} & {r['mean']:.4f} $\\pm$ {r['std']:.4f} & [{r['min']:.4f}, {r['max']:.4f}] \\\\" for k, r in dgs.iterrows())
write("tabS10_learning_curve.tex", r"""\begin{table}[htbp]\centering
\caption{(a) Learning curve under the main evaluation protocol: for seeds 42, 7 and 13 a stratified sub-sample of the training set is used for fitting and the fixed seed-specific test set for evaluation; at 100\% the protocol coincides with the main evaluation. (b) Diagnostic of the learning curve of the original submission, which used scikit-learn's \texttt{learning\_curve} with an unshuffled 5-fold split on rows ordered by insulator type, so that each validation fold was dominated by one or two types; HyPhysML with the seed-42 settings.}\label{stab:lc}
\small
\textbf{(a)}\par
\begin{tabular}{lcccc}
\toprule
Training fraction & $n_\mathrm{train}$ & $R^2$ (train) & $R^2$ (test) & Gap \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}

\medskip
\textbf{(b)}\par
\begin{tabular}{lcc}
\toprule
Protocol (full data, 5 folds) & $R^2$ (mean $\pm$ s.d.) & [min, max] \\
\midrule
""" + drows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S11: ablation ------------------------------------------------------------
ab = rd("TabS7_ablation.csv").set_index("Model")
labels = {"HyPhysML": "HyPhysML (full)", "HyPhysML-MC": "HyPhysML-MC (monotone-constrained)",
          "XGBoost": "XGBoost (single model, tuned)", "HyPhysML-noHPO": "HyPhysML without HPO (library defaults)",
          "HyPhysML-MeanStack": "HyPhysML with mean instead of ridge meta-learner"}
rows = "\n".join(f"{labels[m]} & {ab.loc[m,'R2_mean']:.5f} $\\pm$ {ab.loc[m,'R2_std']:.5f} & {ab.loc[m,'RMSE_mean']:.3f} & {ab.loc[m,'MAPE_mean']:.3f} & {ab.loc[m,'dR2_vs_full']:+.5f} \\\\"
                 for m in ["HyPhysML", "XGBoost", "HyPhysML-noHPO", "HyPhysML-MeanStack", "HyPhysML-MC"])
write("tabS11_ablation.tex", r"""\begin{table}[htbp]\centering
\caption{Ablation study (mean over 10 seeds; every variant is fitted on the same splits).}\label{stab:ablation}
\small
\begin{tabular}{lcccc}
\toprule
Variant & $R^2$ (mean $\pm$ s.d.) & RMSE (kV) & MAPE (\%) & $\Delta R^2$ vs full \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S12: extrapolation -------------------------------------------------------
ex = rd("R1-3_extrapolation_leave_one_level_out.csv")
models = ["HyPhysML", "HyPhysML-MC", "XGBoost", "Obenaus"]
rows = []
for (fac, lev), g in ex.groupby(["Factor", "Held_out_level"], sort=False):
    g = g.set_index("Model")
    facl = {"random 80/20 (seed 42)": "Random 80/20 (seed 42)", "Sample": "Insulator type"}.get(fac, fac)
    levl = "---" if lev == "-" else (f"Smp\\_{lev}" if fac == "Sample" else lev)
    pos = g["Position"].iloc[0].replace("boundary (extrapolation)", "boundary").replace("unseen insulator type", "unseen type")
    cells = " & ".join(f"{g.loc[m,'RMSE']:.2f}" for m in models)
    rows.append(f"{facl} & {levl} & {pos} & {cells} \\\\")
write("tabS12_extrapolation.tex", r"""\begin{table}[htbp]\centering
\caption{Scope of applicability: RMSE (kV) when all 1,664 records at one level of a factor (or of one insulator type) are withheld from training and used as the test set. Fixed default settings are used for every model (no tuning), so that no tuning data overlap the held-out level; the first row gives the same models on the random seed-42 split for reference. Interior levels require interpolation between neighbouring levels of that factor; boundary levels require extrapolation beyond the training range.}\label{stab:extrap}
\small
\begin{tabular}{lllcccc}
\toprule
Held-out factor & Level & Position & HyPhysML & HyPhysML-MC & XGBoost & Obenaus \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S13: HPO overlap of original protocol -------------------------------------
ov = rd("R2-1_old_HPO_test_overlap.csv")
rows = "\n".join(f"{r.Seed} & {r.n_test:,} & {r.test_in_old_HPO_subset:,} & {r.pct:.1f} & {r.test_in_new_HPO_data} \\\\" for r in ov.itertuples())
write("tabS13_hpo_overlap.tex", r"""\begin{table}[htbp]\centering
\caption{Overlap between the test set of each evaluation seed and the data used for hyperparameter optimization. Original submission: one HPO subset (seed 999, 80\% of the records) shared by all seeds. Revised protocol: HPO is repeated inside the training set of each seed, so the overlap is zero by construction. All split indices are archived with the code.}\label{stab:overlap}
\small
\begin{tabular}{ccccc}
\toprule
Seed & $n_\mathrm{test}$ & Test records in original HPO subset & (\%) & Test records used for HPO (revised) \\
\midrule
""" + rows + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---- S14: hyperparameters -----------------------------------------------------
hp = rd("TabS8_hyperparameters_summary.csv")
rows = []
prev = None
for r in hp.itertuples():
    sel = esc(r[4]).replace("[", "[").replace("{", r"\{").replace("}", r"\}")
    ss = esc(r[3]).replace("{", r"\{").replace("}", r"\}")
    s42 = r[5]
    try:
        s42 = f"{float(s42):.4g}"
    except Exception:
        s42 = esc(s42)
    rows.append(f"{nm(r.Model) if r.Model != prev else ''} & {esc(r.Hyperparameter)} & {ss} & {sel} & {s42} \\\\")
    prev = r.Model
pre = rd("TabS8b_preprocessing.csv")
prows = "\n".join(f"{esc(r.Model)} & {esc(r.Preprocessing)} \\\\" for r in pre.itertuples())
write("tabS14_hyperparameters.tex", r"""\begin{longtable}{p{2.4cm}p{2.7cm}p{3.0cm}p{4.3cm}p{1.5cm}}
\caption{Hyperparameter search spaces and selected values of all 11 tuned models. Optuna-TPE, 200 trials per model, maximizing the mean 5-fold CV $R^2$ inside the training set of each seed. Selected values are summarized as median [min, max] over the 10 seeds (categorical: frequency); the seed-42 value is listed separately. Integers are sampled uniformly, ``log'' denotes log-uniform sampling.}\label{stab:hpo}\\
\toprule
Model & Hyperparameter & Search space & Selected: median [min, max] & Seed 42 \\
\midrule
\endfirsthead
\multicolumn{5}{c}{\tablename\ \thetable\ -- continued}\\
\toprule
Model & Hyperparameter & Search space & Selected: median [min, max] & Seed 42 \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(rows) + r"""
\end{longtable}

\begin{table}[htbp]\centering
\caption{Preprocessing of each model. Scalers are fitted inside a scikit-learn \texttt{Pipeline}, i.e.\ on the training folds only, both during HPO and in the final fits.}\label{stab:prep}
\small
\begin{tabular}{p{4.2cm}p{10.5cm}}
\toprule
Model & Preprocessing \\
\midrule
""" + prows + r"""
\bottomrule
\end{tabular}
\end{table}
""")
