# ========================================================================
# HyPhysML — analysis pipeline, version 2 (revised manuscript, Scientific Reports)
# ========================================================================
#
# This script replaces version 1 (legacy/hyphysml_v1_original_submission.py). It changes the
# evaluation protocol where the reviewers identified a problem and adds the
# analyses requested in the review. Every change is tagged [R1-x] / [R2-x]
# with the reviewer and comment number it answers.
#
#   [R2-1]  Hyperparameter optimisation is NESTED: for every evaluation seed,
#           Optuna-TPE runs on that seed's training set only (5-fold CV inside
#           it). The test set of a seed is never seen during tuning. The old
#           seed-999 HPO subset overlapped ~80% of every test set; that overlap
#           is computed and saved for the response letter. All split indices
#           are saved.
#   [R2-2]  The Obenaus regression is treated explicitly as an AUXILIARY model.
#           Its coefficients are reported as mean ± s.d. over the 10 seeds, with
#           and without the insulator-type indicators (CD and AD are collinear
#           with the type indicators, so their coefficients are only identified
#           without them). The unsourced grey "theoretical" bands are removed.
#           New: a prediction-level monotonicity audit of every test prediction
#           (does predicted FOV fall when SDD, RH, aging or J increase?), and a
#           genuinely physics-constrained variant, HyPhysML-MC, whose prediction
#           is monotone in SDD, RH, aging and J by construction.
#   [R2-3]  Noise is injected into the RAW inputs and all 25 features are then
#           rebuilt (logs, products, ratios, composites). Clipping rules are
#           explicit and the clipped fraction is reported. Five repetitions per
#           level, all 10 seeds, mean ± s.d. The old single-column perturbation
#           is also run, labelled as such, to explain the earlier result.
#   [R2-4]  Learning curve uses the main evaluation protocol (stratified 80/20
#           split, stratified sub-samples of the training set, fixed test set);
#           at 100% it reproduces the main result. A diagnostic reproduces the
#           old curve (unshuffled 5-fold CV on type-ordered data) to explain the
#           0.953 vs 0.989 discrepancy.
#   [R2-5]  Two bootstrap intervals, clearly separated: (a) seed-level CI of the
#           10-seed MEAN R² (resampling the 10 seed values), (b) observation-
#           level CI of the R² of ONE split (seed 42 test set).
#   [R2-6]  Search spaces and selected values for ALL 11 tuned models (incl.
#           decision tree, SVR, MLP), per seed. Scaling (RobustScaler) is done
#           inside a Pipeline, so in HPO it is fitted on the CV-training folds
#           only. The ridge meta-learner intercept is stored and reported.
#           The KNN base learner now uses the tuned KNN settings (the old code
#           used a fixed k = 3, contradicting the text).
#   [R2-7]  Meta-learner weights and intercept stored for every seed; mean ± s.d.
#   [R2-8]  Figures are regenerated so that captions can match them: a real
#           14-model predicted-vs-actual grid, the box-plot of R²/RMSE/MAPE, and
#           raw-input response curves (in place of the PDPs of engineered
#           features).
#   [R2-9]  Test-set size is printed and saved (n = 1,332 for every seed).
#   [R1-1]  Data-level physical analysis: empirical SDD exponent per test
#           condition, and monotonicity of the measured FOV in each factor.
#   [R1-3]  Scope of applicability: leave-one-level-out extrapolation tests
#           (SDD, RH, aging, insulator type).
#
# Also corrected:
#   * SHAP is computed for the full stacking model (Shapley values are linear,
#     so phi_stack = sum_b w_b * phi_b with a common interventional background).
#     The old Fig. 2 showed SHAP of the XGBoost base learner only.
#   * Permutation importance is computed on the full stacking model, plus a
#     grouped version that permutes a RAW variable and rebuilds its derived
#     features.
#   * The out-of-fold KFold seed equals the evaluation seed (the manuscript said
#     "fold seed 42, held fixed"; the code never did that). Kept, and reported.
#
# Run:            python hyphysml.py
# Smoke test:     HYPHYSML_FAST=1 python hyphysml.py
# Seed subset:    HYPHYSML_SEEDS=42,7,13 python hyphysml.py
#                 (several processes may share the same output folder; each
#                  seed is checkpointed, and the final aggregation runs once all
#                  10 seeds are present.)
# Trials per model and seed: N_OPTUNA (default 200; override with HYPHYSML_N_OPTUNA)
# ========================================================================

import os, sys, json, time, pickle, warnings, platform
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro, probplot, gaussian_kde

import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import xgboost as xgb
import lightgbm as lgb
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ════════════════════════════════════════════════════════════════════════
# §0  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()

def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return paths[-1]

DATA_PATH = os.environ.get("FOV_DATA") or _first_existing([
    os.path.join(_HERE, "data", "FOV dataset.xlsx"),
    os.path.join(_HERE, "FOV dataset.xlsx"),
    os.path.join(os.path.dirname(_HERE), "FOV dataset.xlsx"),
])
OUT_DIR = os.environ.get("FOV_OUT", os.path.join(_HERE, "results"))

FAST = os.environ.get("HYPHYSML_FAST", "0") == "1"

SEEDS        = [42, 7, 13, 99, 2024, 17, 88, 55, 101, 314]
TEST_SIZE    = 0.20
N_FOLDS      = 5                     # OOF folds inside HyPhysML and HPO CV folds
N_OPTUNA     = int(os.environ.get("HYPHYSML_N_OPTUNA", 200))
N_BOOT       = 1000
HPO_CV_JOBS  = int(os.environ.get("HYPHYSML_CV_JOBS", min(5, os.cpu_count() or 1)))  # parallel CV folds in HPO
USE_GPU      = os.environ.get("HYPHYSML_GPU", "0") == "1"   # XGBoost on CUDA (Colab GPU runtime)
SKIP_HPO     = False                 # True: fixed default settings, no tuning

NOISE_FEATS  = ["SDD", "RH", "Aging"]
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]          # fraction of the raw training s.d.
N_NOISE_REP  = 5
CLIP_RULES   = {"SDD": (1e-3, None), "RH": (1.0, 100.0), "Aging": (0.0, None)}

LC_SEEDS     = [42, 7, 13]                        # learning curve seeds
LC_FRACS     = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
POST_SEED    = 42                                 # seed used for SHAP / figures

N_SHAP_BG    = 100
N_SHAP       = 200
N_KERNEL     = 300                                # KernelExplainer budget (KNN only)
N_PERM_REP   = 10

RUN_EDA             = True
RUN_EXTRAPOLATION   = True   # [R1-3] leave-one-level-out tests (~1 h CPU)
RUN_LC_DIAGNOSTIC   = True   # [R2-4] reproduce old learning-curve protocol (~30 min CPU)

if FAST:
    SEEDS = [42, 7]; N_OPTUNA = int(os.environ.get("HYPHYSML_N_OPTUNA", 2))
    N_NOISE_REP = 2; LC_SEEDS = [42]; LC_FRACS = [0.30, 1.00]
    N_SHAP_BG = 30; N_SHAP = 30; N_KERNEL = 40; N_PERM_REP = 2; N_BOOT = 200
    RUN_EXTRAPOLATION = True; RUN_LC_DIAGNOSTIC = False

if os.environ.get("HYPHYSML_EXTRAPOLATION") is not None:
    RUN_EXTRAPOLATION = os.environ["HYPHYSML_EXTRAPOLATION"] == "1"
if os.environ.get("HYPHYSML_LC_DIAGNOSTIC") is not None:
    RUN_LC_DIAGNOSTIC = os.environ["HYPHYSML_LC_DIAGNOSTIC"] == "1"

_seed_env = os.environ.get("HYPHYSML_SEEDS")
RUN_SEEDS = [int(s) for s in _seed_env.split(",")] if _seed_env else list(SEEDS)

FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
CK_DIR  = os.path.join(OUT_DIR, "checkpoints")
for _d in (OUT_DIR, FIG_DIR, TAB_DIR, CK_DIR):
    os.makedirs(_d, exist_ok=True)

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}\nDownload 'FOV dataset.xlsx' from "
        "https://doi.org/10.17632/8r7k4cgkg8.1 or set FOV_DATA.")

def section(t):
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}", flush=True)

section("§0  Configuration")
print(f"  Data      : {DATA_PATH}")
print(f"  Output    : {OUT_DIR}")
print(f"  FAST={FAST}  SEEDS={SEEDS}  running now={RUN_SEEDS}  N_OPTUNA={N_OPTUNA}")
print(f"  CPU cores={os.cpu_count()}  HPO_CV_JOBS={HPO_CV_JOBS}  XGBoost GPU={USE_GPU}")
VERSIONS = {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "xgboost": xgb.__version__, "lightgbm": lgb.__version__,
            "optuna": optuna.__version__, "shap": shap.__version__ if HAS_SHAP else "n/a",
            "scipy": __import__("scipy").__version__}
print("  Versions  :", VERSIONS)
with open(os.path.join(OUT_DIR, "environment_versions.json"), "w") as f:
    json.dump(VERSIONS, f, indent=2)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass
PALETTE = ["#2166AC", "#D7191C", "#4DAC26", "#E08214", "#762A83", "#1B7837",
           "#F4A582", "#8073AC", "#B35806", "#01665E", "#C2A5CF", "#A6DBA0",
           "#3288BD", "#FDDBC7"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11,
                     "axes.labelsize": 12, "axes.titlesize": 12,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

def savefig(name):
    fp = os.path.join(FIG_DIR, name)
    plt.savefig(fp); plt.close("all"); print(f"  [fig] {name}")

def savetab(df_, name, index=False):
    df_.to_csv(os.path.join(TAB_DIR, name), index=index); print(f"  [tab] {name}")

# ════════════════════════════════════════════════════════════════════════
# §1  DATA + FEATURE ENGINEERING (one function, reused for noise / probes)
# ════════════════════════════════════════════════════════════════════════
section("§1  Data + feature engineering")
RAW_COLS = ["Sample", "CD", "AD", "CF", "RH", "Aging", "J", "K", "SDD"]
df_raw = pd.read_excel(DATA_PATH, sheet_name="Whole samples")
df_raw.columns = RAW_COLS + ["FOV"]
SMP_LEVELS = sorted(df_raw["Sample"].unique())
SMP_COLS = [f"Smp_{s}" for s in SMP_LEVELS]
NUM_RAW = ["CD", "AD", "CF", "RH", "Aging", "J", "K", "SDD"]
DERIVED = ["log_SDD", "log_RH", "log_J", "log_CD", "RH_x_SDD", "RH_x_logSDD",
           "JK_product", "Age_x_SDD", "env_stress", "SDD_norm", "logRH_x_logSDD",
           "logJ_x_logSDD", "Age_x_logSDD"]
FN = NUM_RAW + DERIVED + SMP_COLS          # 8 + 13 + 4 = 25 features (same order as before)

def build_features(raw):
    """Raw design variables -> 25-column feature frame. Every derived feature is
    recomputed from the raw values passed in, so perturbing a raw input
    propagates to all features that depend on it."""
    f = pd.DataFrame(index=raw.index)
    for c in NUM_RAW:
        f[c] = raw[c].astype(float)
    f["log_SDD"] = np.log(f["SDD"]); f["log_RH"] = np.log(f["RH"])
    f["log_J"] = np.log(f["J"].clip(lower=0.5)); f["log_CD"] = np.log(f["CD"])
    f["RH_x_SDD"] = f["RH"] * f["SDD"]; f["RH_x_logSDD"] = f["RH"] * f["log_SDD"]
    f["JK_product"] = f["J"] * f["K"]; f["Age_x_SDD"] = f["Aging"] * f["SDD"]
    f["env_stress"] = (f["RH"] / 100) * f["SDD"] * (1 + f["Aging"] / 45)
    f["SDD_norm"] = f["SDD"] / (f["CD"] / 100)
    f["logRH_x_logSDD"] = f["log_RH"] * f["log_SDD"]
    f["logJ_x_logSDD"] = f["log_J"] * f["log_SDD"]
    f["Age_x_logSDD"] = f["Aging"] * f["log_SDD"]
    for s, c in zip(SMP_LEVELS, SMP_COLS):
        f[c] = (raw["Sample"] == s).astype(float)
    return f[FN]

X_df = build_features(df_raw)
X_all = X_df.values.astype(float)
y_all = df_raw["FOV"].values.astype(float)
idx_all = np.arange(len(y_all))
STRAT = df_raw["Sample"].values
print(f"  {len(y_all)} records | {len(FN)} features | FOV {y_all.min():.2f}–{y_all.max():.2f} kV")
print("  Levels:", {c: sorted(df_raw[c].unique().tolist()) for c in ["RH", "Aging", "SDD", "J", "K"]})

def split(seed):
    return train_test_split(idx_all, test_size=TEST_SIZE, random_state=seed, stratify=STRAT)

# ════════════════════════════════════════════════════════════════════════
# §2  OLD-PROTOCOL DIAGNOSTICS + SAVED SPLITS  [R2-1] [R2-9]
# ════════════════════════════════════════════════════════════════════════
section("§2  Split indices and overlap with the old seed-999 HPO subset")
idx_hpo_old, _ = train_test_split(idx_all, test_size=0.20, random_state=999, stratify=STRAT)
_hpo_old = set(idx_hpo_old.tolist())
_ov_rows, _split_rows = [], []
for s in SEEDS:
    tr, te = split(s)
    ov = sum(i in _hpo_old for i in te)
    _ov_rows.append({"Seed": s, "n_train": len(tr), "n_test": len(te),
                     "test_in_old_HPO_subset": ov, "pct": round(100 * ov / len(te), 1),
                     "test_in_new_HPO_data": 0})      # new HPO uses idx_tr only
    _split_rows += [{"Seed": s, "index": int(i), "role": "train"} for i in tr]
    _split_rows += [{"Seed": s, "index": int(i), "role": "test"} for i in te]
_ov_df = pd.DataFrame(_ov_rows)
print(_ov_df.to_string(index=False))
savetab(_ov_df, "R2-1_old_HPO_test_overlap.csv")
pd.DataFrame(_split_rows).to_csv(os.path.join(TAB_DIR, "split_indices_all_seeds.csv.gz"),
                                 index=False, compression="gzip")
pd.DataFrame({"index": idx_hpo_old}).to_csv(os.path.join(TAB_DIR, "old_seed999_HPO_indices.csv"), index=False)

# ════════════════════════════════════════════════════════════════════════
# §3  DATA-LEVEL PHYSICAL ANALYSIS  [R1-1]
# ════════════════════════════════════════════════════════════════════════
section("§3  Data-level physical analysis (no model involved)")
_keys = ["Sample", "RH", "Aging", "SDD", "J", "K"]
_mono_rows = []
for f_ in ["SDD", "RH", "Aging", "J", "K"]:
    if f_ in ("J", "K"):
        oth = ["Sample", "RH", "Aging", "SDD", "K" if f_ == "J" else "J"]
        sub = df_raw[df_raw["K"] > 0]          # J = 1 occurs only with K = 0
    else:
        oth = [k for k in _keys if k != f_]; sub = df_raw
    diffs = [np.diff(g.sort_values(f_)["FOV"].values) for _, g in sub.groupby(oth)]
    diffs = [d for d in diffs if len(d)]
    allv = np.concatenate(diffs)
    _mono_rows.append({"Factor": f_, "n_adjacent_steps": len(allv),
                       "pct_steps_FOV_decreases": round(100 * np.mean(allv < 0), 2),
                       "pct_cells_strictly_monotone": round(100 * np.mean([np.all(d < 0) for d in diffs]), 2),
                       "median_step_kV": round(float(np.median(allv)), 3)})
_mono_data = pd.DataFrame(_mono_rows); print(_mono_data.to_string(index=False))
savetab(_mono_data, "R1-1_data_monotonicity.csv")

_cells = ["Sample", "RH", "Aging", "J", "K"]
_exp = (df_raw.groupby(_cells)
        .apply(lambda g: pd.Series({"b_SDD": np.polyfit(np.log(g["SDD"]), np.log(g["FOV"]), 1)[0],
                                    "r2_loglog": np.corrcoef(np.log(g["SDD"]), np.log(g["FOV"]))[0, 1] ** 2}))
        .reset_index())
savetab(_exp, "R1-1_SDD_exponent_per_condition.csv")
_exp_sum = pd.concat([
    _exp.groupby("Sample")["b_SDD"].describe()[["mean", "std", "min", "max"]].assign(by="Sample"),
    _exp.groupby("RH")["b_SDD"].describe()[["mean", "std", "min", "max"]].assign(by="RH"),
    _exp.groupby("Aging")["b_SDD"].describe()[["mean", "std", "min", "max"]].assign(by="Aging"),
]).round(4)
_exp_sum.loc["ALL"] = [_exp.b_SDD.mean(), _exp.b_SDD.std(), _exp.b_SDD.min(), _exp.b_SDD.max(), "all"]
print(_exp_sum.to_string())
savetab(_exp_sum, "R1-1_SDD_exponent_summary.csv", index=True)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, f_ in zip(axes, ["SDD", "RH", "Aging", "J"]):
    sub = df_raw if f_ != "J" else df_raw[df_raw["K"] > 0]
    m = sub.groupby(["Sample", f_])["FOV"].mean().unstack(0)
    for i, c in enumerate(m.columns):
        ax.plot(m.index, m[c], "o-", color=PALETTE[i], label=f"Smp_{c}")
    if f_ == "SDD":
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel=f_, ylabel="Mean measured FOV (kV)", title=f"Measured FOV vs {f_}")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
plt.suptitle("Measured flashover voltage: marginal means by insulator type (data only)", fontweight="bold")
plt.tight_layout(); savefig("FigR1_data_response_by_type.png")

# ════════════════════════════════════════════════════════════════════════
# §4  EDA + VIF (unchanged analyses; regenerated for completeness)
# ════════════════════════════════════════════════════════════════════════
if RUN_EDA:
    section("§4  EDA + VIF")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    _VIF_COLS = NUM_RAW + DERIVED
    _Xv = add_constant(X_df[_VIF_COLS].astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        _vif = pd.DataFrame({"Feature": _VIF_COLS,
                             "VIF": [variance_inflation_factor(_Xv.values, i + 1) for i in range(len(_VIF_COLS))]})
    _vif = _vif.sort_values("VIF", ascending=False).reset_index(drop=True)
    _vif["Severity"] = _vif["VIF"].apply(lambda v: "High (>10)" if v > 10 else ("Moderate (5-10)" if v > 5 else "Low (<5)"))
    savetab(_vif, "TabS2_vif.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    _vc = np.minimum(_vif["VIF"].replace(np.inf, 1e7).values[::-1], 1e7)
    ax.barh(range(len(_vc)), _vc, color=["#D7191C" if v > 10 else ("#F4A582" if v > 5 else "#2166AC") for v in _vc])
    ax.set_xscale("log"); ax.axvline(5, color="orange", ls="--"); ax.axvline(10, color="red", ls="--")
    ax.set_yticks(range(len(_vc))); ax.set_yticklabels(_vif["Feature"].values[::-1], fontsize=9)
    ax.set_xlabel("VIF (log scale; infinite values drawn at 1e7)"); ax.set_title("Variance inflation factors")
    plt.tight_layout(); savefig("FigS14_vif.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(y_all, bins=55, color=PALETTE[0], alpha=0.82, edgecolor="white")
    axes[0].axvline(y_all.mean(), color="crimson", ls="--", label=f"Mean={y_all.mean():.1f} kV")
    axes[0].axvline(np.median(y_all), color="darkorange", ls=":", label=f"Median={np.median(y_all):.1f} kV")
    axes[0].set(xlabel="FOV (kV)", ylabel="Frequency"); axes[0].legend()
    probplot(y_all, plot=axes[1]); axes[1].set_title("Normal Q-Q plot")
    _kx = np.linspace(y_all.min(), y_all.max(), 300); _kd = gaussian_kde(y_all)(_kx)
    axes[2].plot(_kx, _kd, color=PALETTE[0]); axes[2].fill_between(_kx, _kd, alpha=0.2)
    axes[2].set(xlabel="FOV (kV)", ylabel="Density")
    plt.tight_layout(); savefig("FigS01_fov_distribution.png")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for i, c in enumerate(["CD", "AD", "SDD", "RH", "Aging", "J", "K"]):
        ax = axes[i // 4, i % 4]; ax.hist(df_raw[c], bins=45, color=PALETTE[i], alpha=0.82)
        ax.set(xlabel=c, ylabel="Frequency")
    axes[1, 3].set_visible(False); plt.tight_layout(); savefig("FigS02_input_histograms.png")

    _cc = X_df[["CD", "AD", "SDD", "RH", "Aging", "J", "K", "log_SDD", "log_RH", "log_J", "log_CD"]].copy()
    _cc["FOV"] = y_all; _cm = _cc.corr()
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(_cm, mask=np.triu(np.ones_like(_cm, dtype=bool)), annot=True, fmt=".2f",
                cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0, vmin=-1, vmax=1, ax=ax)
    plt.tight_layout(); savefig("FigS03_correlation_heatmap.png")

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for ax, (c, lab) in zip(axes.flat, [("log_SDD", "ln(SDD)"), ("log_CD", "ln(CD)"), ("log_RH", "ln(RH)"),
                                         ("Aging", "Aging"), ("log_J", "ln(J)"), ("K", "K")]):
        v = X_df[c].values; ax.scatter(v, y_all, s=6, alpha=0.35, c=y_all, cmap="plasma")
        ax.set(xlabel=lab, ylabel="FOV (kV)", title=f"FOV vs {lab} (r={np.corrcoef(v, y_all)[0, 1]:.3f})")
    plt.tight_layout(); savefig("FigS04_fov_scatter_grid.png")

    _pp = _cc[["log_SDD", "log_CD", "log_RH", "Aging", "FOV"]]
    g = sns.pairplot(_pp.sample(min(2000, len(_pp)), random_state=1), plot_kws={"s": 5, "alpha": 0.3})
    g.savefig(os.path.join(FIG_DIR, "FigS05_pairplot.png")); plt.close("all"); print("  [fig] FigS05_pairplot.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([y_all[STRAT == s] for s in SMP_LEVELS], patch_artist=True)
    ax.set_xticks(range(1, len(SMP_COLS) + 1)); ax.set_xticklabels(SMP_COLS); ax.set_ylabel("FOV (kV)"); plt.tight_layout(); savefig("FigS06_fov_by_type.png")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for col, (c, lc) in enumerate([("SDD", "log_SDD"), ("RH", "log_RH"), ("CD", "log_CD"), ("J", "log_J")]):
        for row, cc in enumerate([c, lc]):
            v = X_df[cc].values; axes[row, col].scatter(v, y_all, s=5, alpha=0.3, color=PALETTE[col + 4 * row])
            axes[row, col].set(xlabel=cc, ylabel="FOV (kV)", title=f"r={np.corrcoef(v, y_all)[0, 1]:.3f}")
    plt.tight_layout(); savefig("FigS07_log_transform.png")
    savetab(df_raw[NUM_RAW + ["FOV"]].describe().round(3), "Tab1_descriptive_stats.csv", index=True)

# ════════════════════════════════════════════════════════════════════════
# §5  METRICS / STATISTICS HELPERS
# ════════════════════════════════════════════════════════════════════════
def compute_metrics(yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp)
    return dict(R2=float(r2_score(yt, yp)),
                RMSE=float(np.sqrt(mean_squared_error(yt, yp))),
                MAE=float(mean_absolute_error(yt, yp)),
                MAPE=float(np.mean(np.abs((yt - yp) / yt)) * 100),
                NSE=float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)))

def boot_ci_observations(yt, yp, n_boot=None, seed=42):
    """[R2-5] CI of the R² of ONE split: resample test observations."""
    n_boot = n_boot or N_BOOT
    rng = np.random.RandomState(seed); n = len(yt)
    v = [r2_score(yt[i], yp[i]) for i in (rng.randint(0, n, n) for _ in range(n_boot))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def boot_ci_seed_mean(vals, n_boot=None, seed=42):
    """[R2-5] CI of the 10-seed MEAN: resample the seed-level values."""
    n_boot = n_boot or N_BOOT
    vals = np.asarray(vals); rng = np.random.RandomState(seed)
    v = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def cliffs_delta(a, b):
    stat, _ = mannwhitneyu(a, b, alternative="two-sided")
    d = 2. * float(stat) / (len(a) * len(b)) - 1.
    sz = "large" if abs(d) >= .474 else "medium" if abs(d) >= .33 else "small" if abs(d) >= .147 else "negligible"
    return round(d, 4), sz

# ════════════════════════════════════════════════════════════════════════
# §6  MODEL ZOO: search spaces, builders, default settings  [R2-6]
# ════════════════════════════════════════════════════════════════════════
TUNABLE = ["XGBoost", "LightGBM", "GBR", "HistGBR", "RF", "Extra Trees",
           "Ridge", "Decision Tree", "KNN", "SVR", "MLP"]
SCALED = {"Ridge", "KNN", "SVR", "MLP"}               # RobustScaler inside a Pipeline
MLP_ARCHS = [(64, 32), (128, 64), (128, 64, 32), (256, 128, 64), (256, 128, 64, 32), (128, 64, 32, 16)]

SEARCH_SPACES = {   # human-readable, written to Table S8
    "XGBoost": {"n_estimators": "int [300, 2000]", "learning_rate": "float [0.005, 0.15] log",
                "max_depth": "int [4, 10]", "subsample": "float [0.6, 1.0]",
                "colsample_bytree": "float [0.5, 1.0]", "min_child_weight": "int [1, 20]",
                "gamma": "float [0, 0.5]", "reg_lambda": "float [0.01, 10] log", "reg_alpha": "float [0, 2]"},
    "LightGBM": {"n_estimators": "int [300, 2000]", "learning_rate": "float [0.005, 0.15] log",
                 "num_leaves": "int [31, 255]", "subsample": "float [0.6, 1.0]",
                 "colsample_bytree": "float [0.5, 1.0]", "min_child_samples": "int [5, 100]",
                 "reg_lambda": "float [0.01, 10] log", "reg_alpha": "float [0, 2]"},
    "GBR": {"n_estimators": "int [300, 1500]", "learning_rate": "float [0.005, 0.1] log",
            "max_depth": "int [3, 7]", "subsample": "float [0.6, 1.0]",
            "min_samples_leaf": "int [1, 10]", "max_features": "float [0.4, 1.0]"},
    "HistGBR": {"max_iter": "int [300, 1500]", "learning_rate": "float [0.005, 0.1] log",
                "max_depth": "int [3, 10]", "l2_regularization": "float [0, 1]",
                "min_samples_leaf": "int [5, 50]", "max_leaf_nodes": "int [20, 255]"},
    "RF": {"n_estimators": "int [200, 800]", "min_samples_leaf": "int [1, 10]",
           "max_features": "float [0.3, 0.9]", "min_samples_split": "int [2, 10]"},
    "Extra Trees": {"n_estimators": "int [200, 800]", "min_samples_leaf": "int [1, 10]",
                    "max_features": "float [0.3, 0.9]", "min_samples_split": "int [2, 10]"},
    "Ridge": {"alpha": "float [0.001, 100] log"},
    "Decision Tree": {"max_depth": "int [3, 20]", "min_samples_leaf": "int [1, 20]",
                      "min_samples_split": "int [2, 20]",
                      "max_features": "{sqrt, log2, None, 0.5, 0.7, 0.9}"},
    "KNN": {"n_neighbors": "int [2, 20]", "weights": "{uniform, distance}", "p": "{1, 2}"},
    "SVR": {"C": "float [0.1, 1000] log", "gamma": "{scale, auto}", "epsilon": "float [0.001, 1] log",
            "kernel": "fixed: rbf"},
    "MLP": {"hidden_layer_sizes": "{" + ", ".join(str(a) for a in MLP_ARCHS) + "}",
            "alpha": "float [1e-5, 1e-2] log", "learning_rate_init": "float [1e-4, 1e-2] log",
            "batch_size": "{32, 64, 128, auto}",
            "fixed": "max_iter=500, early_stopping=True"},
}

def suggest(name, t):
    if name == "XGBoost":
        return dict(n_estimators=t.suggest_int("n_estimators", 300, 2000),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.15, log=True),
                    max_depth=t.suggest_int("max_depth", 4, 10),
                    subsample=t.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_weight=t.suggest_int("min_child_weight", 1, 20),
                    gamma=t.suggest_float("gamma", 0., 0.5),
                    reg_lambda=t.suggest_float("reg_lambda", 0.01, 10., log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 0., 2.))
    if name == "LightGBM":
        return dict(n_estimators=t.suggest_int("n_estimators", 300, 2000),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.15, log=True),
                    num_leaves=t.suggest_int("num_leaves", 31, 255),
                    subsample=t.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 100),
                    reg_lambda=t.suggest_float("reg_lambda", 0.01, 10., log=True),
                    reg_alpha=t.suggest_float("reg_alpha", 0., 2.))
    if name == "GBR":
        return dict(n_estimators=t.suggest_int("n_estimators", 300, 1500),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                    max_depth=t.suggest_int("max_depth", 3, 7),
                    subsample=t.suggest_float("subsample", 0.6, 1.0),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
                    max_features=t.suggest_float("max_features", 0.4, 1.0))
    if name == "HistGBR":
        return dict(max_iter=t.suggest_int("max_iter", 300, 1500),
                    learning_rate=t.suggest_float("learning_rate", 0.005, 0.1, log=True),
                    max_depth=t.suggest_int("max_depth", 3, 10),
                    l2_regularization=t.suggest_float("l2_regularization", 0., 1.),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 50),
                    max_leaf_nodes=t.suggest_int("max_leaf_nodes", 20, 255))
    if name in ("RF", "Extra Trees"):
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 800),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
                    max_features=t.suggest_float("max_features", 0.3, 0.9),
                    min_samples_split=t.suggest_int("min_samples_split", 2, 10))
    if name == "Ridge":
        return dict(alpha=t.suggest_float("alpha", 1e-3, 100., log=True))
    if name == "Decision Tree":
        return dict(max_depth=t.suggest_int("max_depth", 3, 20),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20),
                    min_samples_split=t.suggest_int("min_samples_split", 2, 20),
                    max_features=t.suggest_categorical("max_features", ["sqrt", "log2", None, 0.5, 0.7, 0.9]))
    if name == "KNN":
        return dict(n_neighbors=t.suggest_int("n_neighbors", 2, 20),
                    weights=t.suggest_categorical("weights", ["uniform", "distance"]),
                    p=t.suggest_int("p", 1, 2))
    if name == "SVR":
        return dict(C=t.suggest_float("C", 0.1, 1000., log=True),
                    gamma=t.suggest_categorical("gamma", ["scale", "auto"]),
                    epsilon=t.suggest_float("epsilon", 0.001, 1., log=True))
    if name == "MLP":
        a = t.suggest_categorical("arch", list(range(len(MLP_ARCHS))))
        return dict(hidden_layer_sizes=list(MLP_ARCHS[a]),
                    alpha=t.suggest_float("alpha", 1e-5, 1e-2, log=True),
                    learning_rate_init=t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                    batch_size=t.suggest_categorical("batch_size", [32, 64, 128, "auto"]))
    raise KeyError(name)

# Fixed settings used when SKIP_HPO=True and for the extrapolation study
DEFAULT_PARAMS = {
    "XGBoost": dict(n_estimators=1000, learning_rate=0.02, max_depth=6, subsample=0.8,
                    colsample_bytree=0.7, min_child_weight=3, gamma=0., reg_lambda=1.),
    "LightGBM": dict(n_estimators=1000, learning_rate=0.02, num_leaves=127, subsample=0.8,
                     colsample_bytree=0.7, min_child_samples=10, reg_lambda=1.),
    "GBR": dict(n_estimators=800, learning_rate=0.03, max_depth=4, subsample=0.8, min_samples_leaf=2),
    "HistGBR": dict(max_iter=500, learning_rate=0.03, max_depth=6, l2_regularization=0., min_samples_leaf=5),
    "RF": dict(n_estimators=400, min_samples_leaf=1, max_features=0.6),
    "Extra Trees": dict(n_estimators=400, min_samples_leaf=1, max_features=0.5),
    "Ridge": dict(alpha=1.),
    "Decision Tree": dict(max_depth=12, min_samples_leaf=5),
    "KNN": dict(n_neighbors=5, weights="distance"),
    "SVR": dict(C=100., gamma="scale", epsilon=0.05),
    "MLP": dict(hidden_layer_sizes=[128, 64, 32], alpha=1e-4, learning_rate_init=0.001, batch_size="auto"),
}
# Library defaults: the "without HPO" ablation variant
LIBDEFAULT_PARAMS = {
    "XGBoost": dict(n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0),
    "LightGBM": dict(n_estimators=100, num_leaves=31, learning_rate=0.1),
    "GBR": dict(n_estimators=100, max_depth=3, learning_rate=0.1),
    "HistGBR": dict(max_iter=100),
    "RF": dict(n_estimators=100), "Extra Trees": dict(n_estimators=100),
    "KNN": dict(n_neighbors=5, weights="distance"),
}

def build(name, p, seed, n_jobs=-1):
    p = dict(p)
    if name == "XGBoost":
        gpu = {"device": "cuda"} if USE_GPU else {}
        m = xgb.XGBRegressor(**p, **gpu, random_state=seed, verbosity=0, n_jobs=n_jobs, tree_method="hist")
    elif name == "LightGBM":
        m = lgb.LGBMRegressor(**p, random_state=seed, n_jobs=n_jobs, verbose=-1)
    elif name == "GBR":
        m = GradientBoostingRegressor(**p, random_state=seed)
    elif name == "HistGBR":
        m = HistGradientBoostingRegressor(**p, random_state=seed)
    elif name == "RF":
        m = RandomForestRegressor(**p, n_jobs=n_jobs, random_state=seed)
    elif name == "Extra Trees":
        m = ExtraTreesRegressor(**p, n_jobs=n_jobs, random_state=seed)
    elif name == "Ridge":
        m = Ridge(**p)
    elif name == "Decision Tree":
        m = DecisionTreeRegressor(**p, random_state=seed)
    elif name == "KNN":
        m = KNeighborsRegressor(**p)
    elif name == "SVR":
        m = SVR(kernel="rbf", **p)
    elif name == "MLP":
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        m = MLPRegressor(**p, max_iter=500, early_stopping=True, random_state=seed)
    else:
        raise KeyError(name)
    return Pipeline([("sc", RobustScaler()), ("m", m)]) if name in SCALED else m

# ── Physics-based empirical baselines (unchanged) ───────────────────────
_ix = {c: FN.index(c) for c in FN}
_SMP_IX = [_ix[c] for c in SMP_COLS]
PHY_NAMES = ["log_CD", "log_AD", "log_SDD", "log_RH", "Aging", "log_J", "K"]

def physics_design(X, with_type=True):
    cols = [np.log(X[:, _ix["CD"]]), np.log(X[:, _ix["AD"]]), np.log(X[:, _ix["SDD"]]),
            np.log(X[:, _ix["RH"]]), X[:, _ix["Aging"]], np.log(np.clip(X[:, _ix["J"]], 0.5, None)),
            X[:, _ix["K"]]]
    if with_type:
        cols += [X[:, i] for i in _SMP_IX]
    return np.column_stack(cols)

class ObenausModel(BaseEstimator, RegressorMixin):
    """Log-linearised Obenaus model (Eq. 6). Used (i) as an empirical baseline and
    (ii) as the AUXILIARY physics regression whose coefficient signs are checked.
    It is fitted separately and does NOT constrain HyPhysML."""
    def __init__(self, alpha=0.01, with_type=True):
        self.alpha = alpha; self.with_type = with_type
    def fit(self, X, y):
        self.ridge_ = Ridge(alpha=self.alpha).fit(physics_design(X, self.with_type), np.log(y))
        names = PHY_NAMES + (SMP_COLS if self.with_type else [])
        self.coef_dict_ = dict(zip(names, self.ridge_.coef_))
        self.intercept_ = float(self.ridge_.intercept_)
        return self
    def predict(self, X):
        return np.exp(self.ridge_.predict(physics_design(X, self.with_type)))

SIGN_CHECKS = {"log_SDD": -1, "log_RH": -1, "log_CD": +1, "log_AD": +1, "Aging": -1, "log_J": -1}

class RizkModel(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def _Xr(self, X):
        return np.column_stack([np.log(X[:, _ix["CD"]]), np.log(X[:, _ix["SDD"]]),
                                np.log(X[:, _ix["RH"]]), X[:, _ix["Aging"]]] + [X[:, i] for i in _SMP_IX])
    def fit(self, X, y):
        self.ridge_ = Ridge(alpha=self.alpha).fit(self._Xr(X), np.log(y)); return self
    def predict(self, X):
        return np.exp(self.ridge_.predict(self._Xr(X)))

# ── HyPhysML stacking ensemble ──────────────────────────────────────────
BASE_LEARNERS = ["XGBoost", "LightGBM", "GBR", "HistGBR", "RF", "Extra Trees", "KNN"]

class HyPhysML(BaseEstimator, RegressorMixin):
    """Seven base learners -> ridge meta-learner on out-of-fold predictions.
    meta="ridge": y_hat = w0 + sum_b w_b f_b(x)  (w0 = ridge intercept)
    meta="mean" : y_hat = mean_b f_b(x)          (ablation)"""
    def __init__(self, params=None, n_folds=5, random_state=42, meta_alpha=0.1, meta="ridge"):
        self.params = params; self.n_folds = n_folds; self.random_state = random_state
        self.meta_alpha = meta_alpha; self.meta = meta

    def _make_base(self):
        P = self.params or DEFAULT_PARAMS
        return {nm: build(nm, P[nm], self.random_state) for nm in BASE_LEARNERS}

    def fit(self, X, y):
        bls = self._make_base(); self.base_names_ = list(bls)
        if self.meta == "ridge":
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            oof = np.zeros((len(y), len(bls)))
            for b, (nm, est) in enumerate(bls.items()):
                for ti, vi in kf.split(X):
                    oof[vi, b] = clone(est).fit(X[ti], y[ti]).predict(X[vi])
            self.oof_ = oof
            self.oof_r2_ = {nm: float(r2_score(y, oof[:, b])) for b, nm in enumerate(bls)}
            self.oof_resid_corr_ = np.corrcoef((y[:, None] - oof).T)
            m = Ridge(alpha=self.meta_alpha, fit_intercept=True).fit(oof, y)
            self.meta_coef_ = m.coef_.copy(); self.meta_intercept_ = float(m.intercept_)
        else:
            self.meta_coef_ = np.ones(len(bls)) / len(bls); self.meta_intercept_ = 0.
        self.bl_fit_ = {nm: clone(est).fit(X, y) for nm, est in bls.items()}
        return self

    def base_predictions(self, X):
        return np.column_stack([est.predict(X) for est in self.bl_fit_.values()])

    def predict(self, X):
        return self.base_predictions(X) @ self.meta_coef_ + self.meta_intercept_

# ── HyPhysML-MC: monotone-constrained variant  [R1-2] [R2-2] ────────────
MC_INPUTS = NUM_RAW + SMP_COLS            # raw inputs only: derived features would make
MC_SIGNS = {"SDD": -1, "RH": -1, "Aging": -1, "J": -1}   # feature-level constraints inconsistent
MC_LEARNERS = ["XGBoost", "LightGBM", "HistGBR"]           # learners that support monotone constraints

class HyPhysMLMC(BaseEstimator, RegressorMixin):
    """Physics-constrained stacking: XGBoost, LightGBM and HistGBR trained on the
    raw inputs with monotone constraints (FOV non-increasing in SDD, RH, aging and
    J), combined by a ridge meta-learner restricted to non-negative weights.
    A non-negative combination of non-increasing functions is non-increasing, so
    every prediction satisfies the constraints by construction."""
    def __init__(self, params=None, n_folds=5, random_state=42, meta_alpha=0.1):
        self.params = params; self.n_folds = n_folds; self.random_state = random_state
        self.meta_alpha = meta_alpha

    def _cols(self):
        return [_ix[c] for c in MC_INPUTS]

    def _make_base(self):
        P = self.params or DEFAULT_PARAMS; s = self.random_state
        cst = [MC_SIGNS.get(c, 0) for c in MC_INPUTS]
        return {
            "XGBoost": xgb.XGBRegressor(**P["XGBoost"], monotone_constraints="(" + ",".join(map(str, cst)) + ")",
                                        **({"device": "cuda"} if USE_GPU else {}),
                                        random_state=s, verbosity=0, n_jobs=-1, tree_method="hist"),
            "LightGBM": lgb.LGBMRegressor(**P["LightGBM"], monotone_constraints=cst,
                                          random_state=s, n_jobs=-1, verbose=-1),
            "HistGBR": HistGradientBoostingRegressor(**P["HistGBR"], monotonic_cst=cst, random_state=s),
        }

    def fit(self, X, y):
        Xr = X[:, self._cols()]; bls = self._make_base(); self.base_names_ = list(bls)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof = np.zeros((len(y), len(bls)))
        for b, est in enumerate(bls.values()):
            for ti, vi in kf.split(Xr):
                oof[vi, b] = clone(est).fit(Xr[ti], y[ti]).predict(Xr[vi])
        m = Ridge(alpha=self.meta_alpha, positive=True).fit(oof, y)
        self.meta_coef_ = m.coef_.copy(); self.meta_intercept_ = float(m.intercept_)
        self.bl_fit_ = {nm: clone(est).fit(Xr, y) for nm, est in bls.items()}
        return self

    def predict(self, X):
        Xr = X[:, self._cols()]
        return np.column_stack([e.predict(Xr) for e in self.bl_fit_.values()]) @ self.meta_coef_ + self.meta_intercept_

MODEL_NAMES = ["Obenaus", "Rizk", "Ridge", "Decision Tree", "KNN", "Extra Trees", "GBR",
               "HistGBR", "RF", "SVR", "MLP", "HyPhysML", "XGBoost", "LightGBM"]
EXTRA_VARIANTS = ["HyPhysML-MC", "HyPhysML-noHPO", "HyPhysML-MeanStack"]

def get_model(name, P, seed):
    if name == "Obenaus": return ObenausModel()
    if name == "Rizk": return RizkModel()
    if name == "HyPhysML": return HyPhysML(params=P, random_state=seed)
    if name == "HyPhysML-MC": return HyPhysMLMC(params=P, random_state=seed)
    if name == "HyPhysML-noHPO": return HyPhysML(params=LIBDEFAULT_PARAMS, random_state=seed)
    if name == "HyPhysML-MeanStack": return HyPhysML(params=P, random_state=seed, meta="mean")
    return build(name, P[name], seed)

# ════════════════════════════════════════════════════════════════════════
# §7  NESTED HPO (per seed, training data only)  [R2-1]
# ════════════════════════════════════════════════════════════════════════
def run_hpo(Xtr, ytr, seed):
    ck = os.path.join(CK_DIR, f"hpo_seed{seed}.json")
    store = json.load(open(ck)) if os.path.isfile(ck) else {}
    if store.get("_n_trials") not in (None, N_OPTUNA):
        print(f"    [HPO] checkpoint has {store.get('_n_trials')} trials, N_OPTUNA={N_OPTUNA} -> re-tuning")
        store = {}
    store["_n_trials"] = N_OPTUNA
    cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for name in TUNABLE:
        if name in store:
            continue
        t0 = time.time()
        def objective(trial, name=name):
            p = suggest(name, trial)
            try:
                return cross_val_score(build(name, p, seed, n_jobs=1), Xtr, ytr, cv=cv,
                                       scoring="r2", n_jobs=HPO_CV_JOBS).mean()
            except Exception:
                return -1e9
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=N_OPTUNA, n_jobs=1)
        best = suggest(name, optuna.trial.FixedTrial(study.best_params))
        store[name] = {"params": best, "cv_r2": float(study.best_value),
                       "minutes": round((time.time() - t0) / 60, 2)}
        json.dump(store, open(ck, "w"), indent=1)
        print(f"    [HPO seed {seed}] {name:<14} CV R²={study.best_value:.4f}  ({store[name]['minutes']} min)", flush=True)
    return {k: v["params"] for k, v in store.items() if not k.startswith("_")}, store

# ════════════════════════════════════════════════════════════════════════
# §8  PROBES: monotonicity audit, response curves, noise, permutation
# ════════════════════════════════════════════════════════════════════════
AUDIT_LEVELS = {"SDD": sorted(df_raw["SDD"].unique()), "RH": sorted(df_raw["RH"].unique()),
                "Aging": sorted(df_raw["Aging"].unique()),
                "J": sorted(df_raw.loc[df_raw["K"] > 0, "J"].unique())}

def response_matrix(model, raw_te, var, levels):
    """Set `var` to each level for every test row, rebuild all features, predict.
    Returns an (n_rows x n_levels) matrix."""
    out = []
    for L in levels:
        r = raw_te.copy(); r[var] = L
        out.append(model.predict(build_features(r).values))
    return np.column_stack(out)

def monotonicity_audit(model, raw_te, ice_n=0, seed=0):
    res, curves, ice = {}, {}, {}
    for var, levels in AUDIT_LEVELS.items():
        rt = raw_te[raw_te["K"] > 0] if var == "J" else raw_te
        P = response_matrix(model, rt, var, levels)
        d = np.diff(P, axis=1)           # expected <= 0 (FOV falls as var rises)
        res[var] = {"n_points": len(rt), "n_steps": int(d.size),
                    "pct_steps_increasing": 100 * float(np.mean(d > 0)),
                    "pct_steps_increasing_gt_0.1kV": 100 * float(np.mean(d > 0.1)),
                    "pct_points_any_violation": 100 * float(np.mean((d > 0).any(axis=1))),
                    "max_increase_kV": float(d.max())}
        curves[var] = P.mean(axis=0)
        if ice_n:
            sel = np.random.RandomState(seed).choice(len(P), min(ice_n, len(P)), replace=False)
            ice[var] = P[sel]
    return res, curves, ice

def noise_eval(model, raw_te, yte, std_ref, seed):
    """[R2-3] Returns rows for the raw-propagated test and the single-column test."""
    rows = []
    X0 = build_features(raw_te).values
    r2_0 = r2_score(yte, model.predict(X0))
    for var in NOISE_FEATS:
        for nl in NOISE_LEVELS:
            for rep in range(N_NOISE_REP):
                rng = np.random.RandomState(seed * 1000 + NOISE_FEATS.index(var) * 100 + int(nl * 100) + rep * 7)
                eps = rng.normal(0., nl * std_ref[var], len(raw_te))
                # (a) raw input perturbed, all derived features rebuilt, then clipped
                r = raw_te.copy(); v = r[var].values.astype(float) + eps
                lo, hi = CLIP_RULES[var]
                clipped = np.zeros(len(v), bool)
                if lo is not None: clipped |= v < lo; v = np.maximum(v, lo)
                if hi is not None: clipped |= v > hi; v = np.minimum(v, hi)
                r[var] = v
                yp = model.predict(build_features(r).values)
                rows.append({"mode": "raw_propagated", "Feature": var, "Noise_pct": int(nl * 100), "rep": rep,
                             "R2": r2_score(yte, yp), "RMSE": float(np.sqrt(mean_squared_error(yte, yp))),
                             "dR2": r2_score(yte, yp) - r2_0, "pct_clipped": 100 * clipped.mean()})
                # (b) single engineered column perturbed (old protocol, for comparison)
                Xs = X0.copy(); Xs[:, _ix[var]] += eps
                yp = model.predict(Xs)
                rows.append({"mode": "single_column", "Feature": var, "Noise_pct": int(nl * 100), "rep": rep,
                             "R2": r2_score(yte, yp), "RMSE": float(np.sqrt(mean_squared_error(yte, yp))),
                             "dR2": r2_score(yte, yp) - r2_0, "pct_clipped": 0.})
    return rows, r2_0

RAW_GROUPS = {"SDD": ["SDD"], "RH": ["RH"], "Aging": ["Aging"], "J": ["J"], "K": ["K"],
              "Insulator type (geometry)": ["Sample", "CD", "AD", "CF"]}

def grouped_raw_permutation(model, raw_te, yte, n_rep, seed):
    base = r2_score(yte, model.predict(build_features(raw_te).values)); rows = []
    for g, cols in RAW_GROUPS.items():
        drops = []
        for rep in range(n_rep):
            perm = np.random.RandomState(seed + rep).permutation(len(raw_te))
            r = raw_te.copy()
            r[cols] = raw_te[cols].values[perm]
            drops.append(base - r2_score(yte, model.predict(build_features(r).values)))
        rows.append({"Group": g, "mean_R2_drop": float(np.mean(drops)), "sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("mean_R2_drop", ascending=False)

def stack_shap(hyp, X_bg, X_ex):
    """Exact SHAP of the linear stack from per-learner interventional SHAP values:
    phi_stack = sum_b w_b * phi_b (Shapley values are linear in the model)."""
    phi = np.zeros(X_ex.shape); base = hyp.meta_intercept_; method = {}
    for w, (nm, est) in zip(hyp.meta_coef_, hyp.bl_fit_.items()):
        try:
            if nm == "KNN":
                raise TypeError
            e = shap.TreeExplainer(est, data=X_bg, feature_perturbation="interventional")
            pb = np.asarray(e.shap_values(X_ex, check_additivity=False)); eb = float(np.ravel(e.expected_value)[0])
            method[nm] = "TreeExplainer"
        except Exception:
            e = shap.KernelExplainer(est.predict, X_bg)
            pb = np.asarray(e.shap_values(X_ex, nsamples=N_KERNEL, silent=True)); eb = float(e.expected_value)
            method[nm] = "KernelExplainer"
        phi += w * pb; base += w * eb
    add_err = float(np.max(np.abs(base + phi.sum(1) - hyp.predict(X_ex))))
    return phi, base, method, add_err

# ════════════════════════════════════════════════════════════════════════
# §9  PER-SEED RUN (checkpointed)
# ════════════════════════════════════════════════════════════════════════
def run_seed(seed):
    ck = os.path.join(CK_DIR, f"seed{seed}.pkl")
    if os.path.isfile(ck):
        print(f"  seed {seed}: checkpoint found, skipping"); return
    T0 = time.time()
    tr, te = split(seed)
    Xtr, Xte, ytr, yte = X_all[tr], X_all[te], y_all[tr], y_all[te]
    raw_tr = df_raw.iloc[tr].reset_index(drop=True); raw_te = df_raw.iloc[te].reset_index(drop=True)
    R = {"seed": seed, "idx_tr": tr, "idx_te": te, "n_test": len(te)}

    section(f"Seed {seed}: nested HPO on {len(tr)} training records (test set untouched)")
    if SKIP_HPO:
        P, hpo_store = dict(DEFAULT_PARAMS), {}
    else:
        P, hpo_store = run_hpo(Xtr, ytr, seed)
    R["params"] = P; R["hpo"] = hpo_store

    section(f"Seed {seed}: fitting {len(MODEL_NAMES) + len(EXTRA_VARIANTS)} models")
    metrics, preds, fitted = [], {}, {}
    for name in MODEL_NAMES + EXTRA_VARIANTS:
        m = get_model(name, P, seed); t0 = time.time()
        m.fit(Xtr, ytr); t_fit = time.time() - t0
        t1 = time.time(); yp = m.predict(Xte); t_pred = time.time() - t1
        ytp = m.predict(Xtr)
        row = compute_metrics(yte, yp)
        row.update({"Model": name, "Seed": seed, "Time_fit_s": round(t_fit, 3),
                    "Time_predict_ms_per_sample": round(1000 * t_pred / len(te), 4),
                    "R2_train": float(r2_score(ytr, ytp))})
        metrics.append(row); preds[name] = yp
        if name in ("HyPhysML", "HyPhysML-MC", "XGBoost", "HyPhysML-MeanStack"):
            fitted[name] = m
        print(f"    {name:<20} R²={row['R2']:.4f}  RMSE={row['RMSE']:.3f}  fit {t_fit:.1f}s", flush=True)
    R["metrics"] = metrics; R["yte"] = yte; R["preds"] = preds
    hyp, mc = fitted["HyPhysML"], fitted["HyPhysML-MC"]

    # meta-learner [R2-6, R2-7]
    R["meta"] = {"names": hyp.base_names_, "coef": hyp.meta_coef_, "intercept": hyp.meta_intercept_,
                 "oof_r2": hyp.oof_r2_, "oof_resid_corr": hyp.oof_resid_corr_}
    R["meta_mc"] = {"names": mc.base_names_, "coef": mc.meta_coef_, "intercept": mc.meta_intercept_}

    # auxiliary physics regression [R2-2]
    R["physics"] = {"with_type": ObenausModel(with_type=True).fit(Xtr, ytr).coef_dict_,
                    "without_type": ObenausModel(with_type=False).fit(Xtr, ytr).coef_dict_}

    # per-type performance [R2-9]
    st = raw_te["Sample"].values
    R["per_type"] = [dict(Type=f"Smp_{s}", n_test=int((st == s).sum()),
                          **compute_metrics(yte[st == s], preds["HyPhysML"][st == s])) for s in SMP_LEVELS]

    # monotonicity audit + response curves [R2-2] [R2-8]
    R["audit"], R["curves"], R["ice"] = {}, {}, {}
    for nm in ("HyPhysML", "HyPhysML-MC", "XGBoost"):
        a, c, i = monotonicity_audit(fitted[nm], raw_te, ice_n=60 if seed == POST_SEED else 0, seed=seed)
        R["audit"][nm], R["curves"][nm], R["ice"][nm] = a, c, i
        print(f"    audit {nm:<12}", {k: round(v["pct_steps_increasing"], 2) for k, v in a.items()})

    # noise [R2-3]
    std_ref = {v: float(raw_tr[v].std()) for v in NOISE_FEATS}
    R["noise"] = {}
    for nm in ("HyPhysML", "HyPhysML-MC"):
        rows, r0 = noise_eval(fitted[nm], raw_te, yte, std_ref, seed)
        R["noise"][nm] = {"rows": rows, "R2_0": r0}
    R["noise_std_ref"] = std_ref

    # learning curve [R2-4]
    if seed in LC_SEEDS:
        lc = []
        for frac in LC_FRACS:
            if frac < 1:
                sub, _ = train_test_split(np.arange(len(tr)), train_size=frac, random_state=seed,
                                          stratify=raw_tr["Sample"].values)
            else:
                sub = np.arange(len(tr))
            mdl = HyPhysML(params=P, random_state=seed).fit(Xtr[sub], ytr[sub])
            lc.append({"Seed": seed, "frac": frac, "n_train": len(sub),
                       "R2_train": float(r2_score(ytr[sub], mdl.predict(Xtr[sub]))),
                       "R2_test": float(r2_score(yte, mdl.predict(Xte)))})
            print(f"    LC frac={frac:.2f} n={len(sub)} test R²={lc[-1]['R2_test']:.4f}", flush=True)
        R["learning_curve"] = lc

    # interpretation on the representative split
    if seed == POST_SEED:
        section(f"Seed {seed}: interpretation (SHAP of the full stack, permutation importance)")
        R["ci_obs"] = {nm: boot_ci_observations(yte, preds[nm]) for nm in MODEL_NAMES + EXTRA_VARIANTS}
        if HAS_SHAP:
            rng = np.random.RandomState(42)
            X_bg = Xtr[rng.choice(len(Xtr), N_SHAP_BG, replace=False)]
            sel = rng.choice(len(Xte), N_SHAP, replace=False); X_ex = Xte[sel]
            phi, base, meth, err = stack_shap(hyp, X_bg, X_ex)
            R["shap"] = {"phi": phi, "X_ex": X_ex, "base": base, "method": meth, "additivity_err": err}
            print(f"    stack SHAP methods={meth}  max additivity error={err:.4f} kV")
            try:
                e = shap.TreeExplainer(hyp.bl_fit_["XGBoost"], data=X_bg, feature_perturbation="interventional")
                R["shap_xgb_only"] = np.asarray(e.shap_values(X_ex, check_additivity=False))
            except Exception as ex:
                print("    XGBoost-only SHAP failed:", ex)
        pi = permutation_importance(hyp, Xte, yte, n_repeats=N_PERM_REP, random_state=42, n_jobs=1, scoring="r2")
        R["perm"] = {"mean": pi.importances_mean, "sd": pi.importances_std}
        R["perm_raw"] = grouped_raw_permutation(hyp, raw_te, yte, N_PERM_REP, seed)
        print(R["perm_raw"].to_string(index=False))

    R["minutes"] = round((time.time() - T0) / 60, 1)
    pickle.dump(R, open(ck, "wb"))
    print(f"  seed {seed} done in {R['minutes']} min -> {ck}", flush=True)

section("§9  Per-seed runs")
for s in RUN_SEEDS:
    run_seed(s)

missing = [s for s in SEEDS if not os.path.isfile(os.path.join(CK_DIR, f"seed{s}.pkl"))]
if missing:
    print(f"\n  Seeds still missing: {missing}. Aggregation runs once all seeds are checkpointed.")
    sys.exit(0)

# ════════════════════════════════════════════════════════════════════════
# §10  OPTIONAL: old learning-curve protocol (diagnostic)  [R2-4]
# ════════════════════════════════════════════════════════════════════════
RES = {s: pickle.load(open(os.path.join(CK_DIR, f"seed{s}.pkl"), "rb")) for s in SEEDS}
P42 = RES[POST_SEED]["params"]

lcdiag_ck = os.path.join(CK_DIR, "lc_diagnostic.pkl")
if RUN_LC_DIAGNOSTIC and not os.path.isfile(lcdiag_ck):
    section("§10  Learning-curve diagnostic: old vs corrected 5-fold CV at full size")
    rows = []
    for lab, cv in [("unshuffled KFold on type-ordered rows (old learning curve)", KFold(5, shuffle=False)),
                    ("shuffled KFold (seed 42)", KFold(5, shuffle=True, random_state=42))]:
        sc = []
        for k, (ti, vi) in enumerate(cv.split(X_all)):
            m = HyPhysML(params=P42, random_state=42).fit(X_all[ti], y_all[ti])
            sc.append(r2_score(y_all[vi], m.predict(X_all[vi])))
            types_in_val = sorted(set(STRAT[vi]))
            rows.append({"protocol": lab, "fold": k + 1, "R2": sc[-1], "types_in_validation": str(types_in_val),
                         "RH_levels_in_validation": str(sorted(set(df_raw['RH'].values[vi])))})
            print(f"    {lab[:30]:<30} fold {k + 1}: R²={sc[-1]:.4f}  types={types_in_val}", flush=True)
        print(f"    -> mean {np.mean(sc):.4f} ± {np.std(sc):.4f}")
    pickle.dump(pd.DataFrame(rows), open(lcdiag_ck, "wb"))

# ════════════════════════════════════════════════════════════════════════
# §11  OPTIONAL: leave-one-level-out extrapolation  [R1-3]
# ════════════════════════════════════════════════════════════════════════
extra_ck = os.path.join(CK_DIR, "extrapolation.pkl")
if RUN_EXTRAPOLATION and not os.path.isfile(extra_ck):
    section("§11  Leave-one-level-out extrapolation (fixed default settings, no tuning)")
    ex_rows = []
    factors = {"SDD": AUDIT_LEVELS["SDD"], "RH": AUDIT_LEVELS["RH"],
               "Aging": AUDIT_LEVELS["Aging"], "Sample": SMP_LEVELS}
    if FAST:
        factors = {"SDD": AUDIT_LEVELS["SDD"]}
    ex_models = {"HyPhysML": lambda: HyPhysML(params=DEFAULT_PARAMS, random_state=42),
                 "HyPhysML-MC": lambda: HyPhysMLMC(params=DEFAULT_PARAMS, random_state=42),
                 "XGBoost": lambda: build("XGBoost", DEFAULT_PARAMS["XGBoost"], 42),
                 "Obenaus": lambda: ObenausModel()}
    tr, te = split(42)            # in-grid reference with the same fixed settings
    for nm, mk in ex_models.items():
        m = mk().fit(X_all[tr], y_all[tr])
        ex_rows.append({"Factor": "random 80/20 (seed 42)", "Held_out_level": "-", "Position": "interpolation",
                        "Model": nm, "n_test": len(te), **compute_metrics(y_all[te], m.predict(X_all[te]))})
    for fac, levels in factors.items():
        for i, L in enumerate(levels):
            msk = df_raw[fac].values == L
            pos = "interior" if 0 < i < len(levels) - 1 else "boundary (extrapolation)"
            if fac == "Sample":
                pos = "unseen insulator type"
            for nm, mk in ex_models.items():
                m = mk().fit(X_all[~msk], y_all[~msk]); yp = m.predict(X_all[msk])
                ex_rows.append({"Factor": fac, "Held_out_level": L, "Position": pos, "Model": nm,
                                "n_test": int(msk.sum()), **compute_metrics(y_all[msk], yp)})
                print(f"    hold out {fac}={L:<6} {nm:<12} RMSE={ex_rows[-1]['RMSE']:.3f}  "
                      f"MAPE={ex_rows[-1]['MAPE']:.2f}%", flush=True)
    pickle.dump(pd.DataFrame(ex_rows), open(extra_ck, "wb"))

# ════════════════════════════════════════════════════════════════════════
# §12  AGGREGATION: tables
# ════════════════════════════════════════════════════════════════════════
section("§12  Aggregation")
res_df = pd.DataFrame([r for s in SEEDS for r in RES[s]["metrics"]])
savetab(res_df, "results_all_seeds.csv")
ALL_M = MODEL_NAMES + EXTRA_VARIANTS
agg = (res_df.groupby("Model").agg(R2_mean=("R2", "mean"), R2_std=("R2", "std"),
                                   RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                                   MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                                   MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
                                   NSE_mean=("NSE", "mean"), R2_train_mean=("R2_train", "mean"),
                                   Time_fit_mean=("Time_fit_s", "mean"), Time_fit_std=("Time_fit_s", "std"),
                                   Pred_ms_per_sample=("Time_predict_ms_per_sample", "mean"))
       .sort_values("R2_mean", ascending=False))
agg["Overfit_gap"] = agg["R2_train_mean"] - agg["R2_mean"]
_ci = {m: boot_ci_seed_mean(res_df.loc[res_df.Model == m, "R2"].values) for m in ALL_M}
agg["R2_CI95_seedmean_lo"] = [_ci[m][0] for m in agg.index]
agg["R2_CI95_seedmean_hi"] = [_ci[m][1] for m in agg.index]
_ci42 = RES[POST_SEED]["ci_obs"]
agg["R2_seed42"] = [res_df[(res_df.Model == m) & (res_df.Seed == POST_SEED)]["R2"].values[0] for m in agg.index]
agg["R2_CI95_seed42_obs_lo"] = [_ci42[m][0] for m in agg.index]
agg["R2_CI95_seed42_obs_hi"] = [_ci42[m][1] for m in agg.index]
savetab(agg.round(5), "Tab2_results_summary_all_models.csv", index=True)
agg14 = agg.loc[[m for m in agg.index if m in MODEL_NAMES]]
BEST = agg14.index[0]
print(agg14[["R2_mean", "R2_std", "RMSE_mean", "MAPE_mean"]].round(4).to_string())
print(f"  Best of the 14 models: {BEST}")

# statistical tests (14 models; reference = HyPhysML)  [R2-5]
R2M = {m: res_df[res_df.Model == m].sort_values("Seed")["R2"].values for m in ALL_M}
fr, fp = stats.friedmanchisquare(*[R2M[m] for m in MODEL_NAMES])
alpha_b = 0.05 / (len(MODEL_NAMES) - 1)
st_rows = []
REF = "HyPhysML"
for m in [x for x in agg14.index if x != REF] + EXTRA_VARIANTS:
    try:
        p1 = stats.wilcoxon(R2M[REF], R2M[m], alternative="greater").pvalue
        p2 = stats.wilcoxon(R2M[REF], R2M[m], alternative="two-sided").pvalue
    except ValueError:
        p1 = p2 = 1.
    d, sz = cliffs_delta(R2M[REF], R2M[m])
    diff = R2M[REF] - R2M[m]
    lo, hi = boot_ci_seed_mean(diff)
    st_rows.append({"vs": m, "wins_of_10": int((diff > 0).sum()), "mean_dR2": diff.mean(),
                    "dR2_CI95_lo": lo, "dR2_CI95_hi": hi,
                    "mean_dRMSE_kV": (res_df[res_df.Model == m].sort_values("Seed")["RMSE"].values
                                      - res_df[res_df.Model == REF].sort_values("Seed")["RMSE"].values).mean(),
                    "p_one_sided": p1, "p_two_sided": p2,
                    "significant_bonferroni": (p1 < alpha_b) if m in MODEL_NAMES else "n/a (variant)",
                    "cliffs_delta": d, "effect": sz})
st_df = pd.DataFrame(st_rows)
print(f"  Friedman chi2({len(MODEL_NAMES) - 1})={fr:.2f}  P={fp:.2e}   Bonferroni alpha={alpha_b:.5f}")
print(st_df.round(6).to_string(index=False))
savetab(st_df.round(6), "Tab3_statistical_tests.csv")
pd.DataFrame([{"Friedman_chi2": fr, "df": len(MODEL_NAMES) - 1, "P": fp, "alpha_bonferroni": alpha_b}]) \
    .to_csv(os.path.join(TAB_DIR, "Tab3_friedman.csv"), index=False)

# hyperparameters per seed  [R2-6]
hp_rows = []
for s in SEEDS:
    for nm, p in RES[s]["params"].items():
        for k, v in p.items():
            hp_rows.append({"Model": nm, "Hyperparameter": k, "Seed": s, "Value": v,
                            "Search space": SEARCH_SPACES.get(nm, {}).get(k, "-"),
                            "Inner CV R2": RES[s]["hpo"].get(nm, {}).get("cv_r2", np.nan) if RES[s]["hpo"] else np.nan})
hp_df = pd.DataFrame(hp_rows); savetab(hp_df, "TabS8_hyperparameters_per_seed.csv")
def _summ(v):
    v = list(v)
    if all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) for x in v):
        return f"{np.median(v):.4g} [{np.min(v):.4g}, {np.max(v):.4g}]"
    vc = pd.Series([str(x) for x in v]).value_counts()
    return "; ".join(f"{k} ({c}/{len(v)})" for k, c in vc.items())
hp_sum = (hp_df.groupby(["Model", "Hyperparameter", "Search space"])["Value"]
          .apply(_summ).reset_index().rename(columns={"Value": "Selected: median [min, max] over seeds"}))
hp_s42 = hp_df[hp_df.Seed == POST_SEED][["Model", "Hyperparameter", "Value"]].rename(columns={"Value": "Seed 42"})
hp_sum = hp_sum.merge(hp_s42, on=["Model", "Hyperparameter"], how="left")
hp_sum["Model"] = pd.Categorical(hp_sum["Model"], TUNABLE, ordered=True)
hp_sum = hp_sum.sort_values(["Model", "Hyperparameter"])
savetab(hp_sum, "TabS8_hyperparameters_summary.csv")
pre = pd.DataFrame([
    {"Model": m, "Preprocessing": "RobustScaler (median/IQR) fitted on the training data of each fit, inside a Pipeline"
     if m in SCALED else "none (raw 25 features)"} for m in TUNABLE] + [
    {"Model": "Obenaus / Rizk", "Preprocessing": "log-transform of inputs and target; ridge alpha=0.01 with intercept"},
    {"Model": "HyPhysML meta-learner", "Preprocessing": "ridge alpha=0.1 WITH intercept on OOF predictions (no scaling)"},
    {"Model": "HyPhysML-MC meta-learner", "Preprocessing": "ridge alpha=0.1 with intercept, weights constrained >= 0"}])
savetab(pre, "TabS8b_preprocessing.csv")

# meta weights  [R2-7]
mw = pd.DataFrame([dict(zip(RES[s]["meta"]["names"], RES[s]["meta"]["coef"]),
                        intercept=RES[s]["meta"]["intercept"], Seed=s) for s in SEEDS])
mw_sum = mw.drop(columns="Seed").agg(["mean", "std", "min", "max"]).T
mw_sum["n_negative_of_10"] = (mw.drop(columns="Seed") < 0).sum()
savetab(mw, "TabS12_meta_weights_per_seed.csv"); savetab(mw_sum.round(4), "TabS12_meta_weights_summary.csv", index=True)
print(mw_sum.round(3).to_string())
oof = pd.DataFrame([dict(RES[s]["meta"]["oof_r2"], Seed=s) for s in SEEDS])
savetab(oof.round(5), "TabS12b_base_learner_oof_r2.csv")
corr_mean = np.mean([RES[s]["meta"]["oof_resid_corr"] for s in SEEDS], axis=0)
savetab(pd.DataFrame(corr_mean, index=BASE_LEARNERS, columns=BASE_LEARNERS).round(3),
        "TabS12c_oof_residual_correlation.csv", index=True)
mwmc = pd.DataFrame([dict(zip(RES[s]["meta_mc"]["names"], RES[s]["meta_mc"]["coef"]),
                          intercept=RES[s]["meta_mc"]["intercept"], Seed=s) for s in SEEDS])
savetab(mwmc, "TabS_MC_meta_weights_per_seed.csv")

# physics regression  [R2-2]
ph_rows = []
for variant in ("with_type", "without_type"):
    C = pd.DataFrame([RES[s]["physics"][variant] for s in SEEDS])
    for k in PHY_NAMES:
        exp_sign = SIGN_CHECKS.get(k)
        ph_rows.append({"Regression": "with type indicators" if variant == "with_type" else "without type indicators",
                        "Coefficient": k, "mean": C[k].mean(), "sd": C[k].std(), "min": C[k].min(), "max": C[k].max(),
                        "expected_sign": {-1: "negative", 1: "positive", None: "-"}[exp_sign],
                        "sign_as_expected_seeds": int((np.sign(C[k]) == exp_sign).sum()) if exp_sign else "-",
                        "identifiable": "no (collinear with type indicators)"
                        if (variant == "with_type" and k in ("log_CD", "log_AD")) else "yes"})
ph_df = pd.DataFrame(ph_rows); savetab(ph_df.round(5), "Tab4_physics_coefficients.csv"); print(ph_df.round(4).to_string(index=False))

# per type
pt = pd.DataFrame([dict(r, Seed=s) for s in SEEDS for r in RES[s]["per_type"]])
pt_sum = pt.groupby("Type").agg(n_test=("n_test", "first"), R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                                RMSE_mean=("RMSE", "mean"), MAE_mean=("MAE", "mean"), MAPE_mean=("MAPE", "mean"))
savetab(pt, "TabS5_per_type_all_seeds.csv"); savetab(pt_sum.round(4), "TabS5_per_type_summary.csv", index=True)
savetab(pt[pt.Seed == POST_SEED].round(4), "TabS5_per_type_seed42.csv")

# residuals
resid_all = np.concatenate([RES[s]["yte"] - RES[s]["preds"]["HyPhysML"] for s in SEEDS])
r42 = RES[POST_SEED]["yte"] - RES[POST_SEED]["preds"]["HyPhysML"]
W42, p42 = shapiro(r42)
Wall, pall = shapiro(np.random.RandomState(0).choice(resid_all, 5000, replace=False)) if len(resid_all) > 5000 else shapiro(resid_all)
res_tab = pd.DataFrame([
    {"Scope": "seed 42 test set", "n": len(r42), "mean_kV": r42.mean(), "sd_kV": r42.std(ddof=1), "Shapiro_W": W42, "Shapiro_p": p42},
    {"Scope": "pooled, 10 seeds (Shapiro on random 5000)", "n": len(resid_all), "mean_kV": resid_all.mean(),
     "sd_kV": resid_all.std(ddof=1), "Shapiro_W": Wall, "Shapiro_p": pall}])
savetab(res_tab.round(5), "TabS6_residuals.csv")

# learning curve
lc = pd.DataFrame([r for s in SEEDS for r in RES[s].get("learning_curve", [])])
lc_sum = lc.groupby("frac").agg(n_train=("n_train", "mean"), R2_train_mean=("R2_train", "mean"),
                                R2_train_sd=("R2_train", "std"), R2_test_mean=("R2_test", "mean"),
                                R2_test_sd=("R2_test", "std")).reset_index()
lc_sum["gap"] = lc_sum["R2_train_mean"] - lc_sum["R2_test_mean"]
savetab(lc, "TabS6b_learning_curve_raw.csv"); savetab(lc_sum.round(5), "TabS6b_learning_curve_summary.csv")
if os.path.isfile(lcdiag_ck):
    lcd = pickle.load(open(lcdiag_ck, "rb")); savetab(lcd.round(5), "R2-4_learning_curve_diagnostic.csv")

# noise
nz = pd.DataFrame([dict(r, Model=nm, Seed=s) for s in SEEDS for nm in RES[s]["noise"] for r in RES[s]["noise"][nm]["rows"]])
nz_sum = (nz.groupby(["Model", "mode", "Feature", "Noise_pct"])
          .agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"), dR2_mean=("dR2", "mean"), dR2_sd=("dR2", "std"),
               RMSE_mean=("RMSE", "mean"), pct_clipped=("pct_clipped", "mean"), n=("R2", "size")).reset_index())
r0 = pd.DataFrame([{"Model": nm, "Seed": s, "R2_0": RES[s]["noise"][nm]["R2_0"]} for s in SEEDS for nm in RES[s]["noise"]])
savetab(nz, "TabS1_noise_raw.csv"); savetab(nz_sum.round(6), "TabS1_noise_summary.csv")
print(nz_sum[(nz_sum.Model == "HyPhysML")].round(5).to_string(index=False))

# monotonicity audit
au = pd.DataFrame([dict(v, Model=nm, Variable=var, Seed=s) for s in SEEDS for nm in RES[s]["audit"]
                   for var, v in RES[s]["audit"][nm].items()])
au_sum = au.groupby(["Model", "Variable"]).agg(
    pct_steps_increasing=("pct_steps_increasing", "mean"),
    pct_steps_increasing_gt_0_1kV=("pct_steps_increasing_gt_0.1kV", "mean"),
    pct_points_any_violation=("pct_points_any_violation", "mean"),
    max_increase_kV=("max_increase_kV", "max")).reset_index()
_dm = _mono_data.set_index("Factor")["pct_steps_FOV_decreases"]
au_sum["data_pct_steps_increasing"] = [round(100 - _dm.get(v, np.nan), 2) for v in au_sum["Variable"]]
savetab(au, "R2-2_monotonicity_audit_all_seeds.csv"); savetab(au_sum.round(4), "R2-2_monotonicity_audit_summary.csv")
print(au_sum.round(3).to_string(index=False))

# ablation
abl = agg.loc[[m for m in ["HyPhysML", "HyPhysML-MC", "XGBoost", "HyPhysML-noHPO", "HyPhysML-MeanStack"] if m in agg.index],
              ["R2_mean", "R2_std", "RMSE_mean", "MAPE_mean", "Time_fit_mean"]]
abl["dR2_vs_full"] = abl["R2_mean"] - agg.loc["HyPhysML", "R2_mean"]
savetab(abl.round(5), "TabS7_ablation.csv", index=True)

# extrapolation
if os.path.isfile(extra_ck):
    ex = pickle.load(open(extra_ck, "rb")); savetab(ex.round(4), "R1-3_extrapolation_leave_one_level_out.csv")
    print(ex.pivot_table(index=["Factor", "Held_out_level"], columns="Model", values="RMSE").round(3).to_string())

# HPO time
hpo_t = pd.DataFrame([{"Seed": s, "Model": nm, "minutes": v["minutes"], "inner_cv_r2": v["cv_r2"]}
                      for s in SEEDS for nm, v in RES[s]["hpo"].items() if not nm.startswith("_")]) if not SKIP_HPO else pd.DataFrame()
if len(hpo_t):
    savetab(hpo_t, "hpo_time_and_inner_cv.csv")

# ════════════════════════════════════════════════════════════════════════
# §13  FIGURES
# ════════════════════════════════════════════════════════════════════════
section("§13  Figures")
R42 = RES[POST_SEED]; yte42 = R42["yte"]; LABEL = {m: m for m in ALL_M}
ORD = list(agg14.index)

# Fig 1 — predicted vs actual, seed 42  [R2-5] [R2-9]
yp = R42["preds"]["HyPhysML"]; rr = yte42 - yp; m42 = compute_metrics(yte42, yp); lo42, hi42 = _ci42["HyPhysML"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(yte42, yp, c=rr, cmap="RdBu", s=14, alpha=0.6, vmin=-5, vmax=5)
mv, xv = min(yte42.min(), yp.min()), max(yte42.max(), yp.max()); axes[0].plot([mv, xv], [mv, xv], "k--")
axes[0].set(xlabel="Actual FOV (kV)", ylabel="Predicted FOV (kV)",
            title=f"(a) HyPhysML, seed 42, n={len(yte42)}\nR²={m42['R2']:.4f} [95% CI of this split {lo42:.4f}, {hi42:.4f}]  RMSE={m42['RMSE']:.3f} kV")
plt.colorbar(sc, ax=axes[0], label="Residual (kV)", fraction=0.03)
axes[1].scatter(yp, rr, s=10, alpha=0.5, color=PALETTE[0]); axes[1].axhline(0, color="k", ls="--")
for sg in (2, -2):
    axes[1].axhline(sg * rr.std(), color="red", ls=":", label=f"{sg}σ = {sg * rr.std():.2f} kV")
axes[1].set(xlabel="Predicted FOV (kV)", ylabel="Residual (kV)", title="(b) Residuals"); axes[1].legend()
plt.tight_layout(); savefig("Fig1_predicted_vs_actual_seed42.png")

# Fig S9 — 14-model grid  [R2-8]
fig, axes = plt.subplots(4, 4, figsize=(18, 18)); axes = axes.ravel()
for ax, nm in zip(axes, ORD):
    p_ = R42["preds"][nm]; mm = compute_metrics(yte42, p_)
    ax.scatter(yte42, p_, s=4, alpha=0.4, color=PALETTE[0] if nm == "HyPhysML" else "gray")
    ax.plot([yte42.min(), yte42.max()], [yte42.min(), yte42.max()], "r--", lw=1)
    ax.set_title(f"{nm}\nR²={mm['R2']:.4f}  RMSE={mm['RMSE']:.2f} kV", fontsize=10)
    ax.set_xlabel("Actual (kV)", fontsize=9); ax.set_ylabel("Predicted (kV)", fontsize=9)
for ax in axes[len(ORD):]:
    ax.set_visible(False)
plt.suptitle(f"Predicted vs actual FOV, 14 models, seed-42 test set (n={len(yte42)})", fontweight="bold")
plt.tight_layout(); savefig("FigS09_all_models_pred_vs_actual_seed42.png")

# Fig S8 — box plots R², RMSE, MAPE  [R2-8]
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for ax, (met, lab) in zip(axes, [("R2", "R² (higher is better)"), ("RMSE", "RMSE, kV (lower is better)"),
                                 ("MAPE", "MAPE, % (lower is better)")]):
    data = [res_df[res_df.Model == m][met].values for m in ORD]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black"))
    for patch, col in zip(bp["boxes"], PALETTE): patch.set_facecolor(col); patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(ORD) + 1)); ax.set_xticklabels(ORD, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(lab)
plt.suptitle("Distribution over the 10 evaluation seeds (14 models)", fontweight="bold")
plt.tight_layout(); savefig("FigS08_model_comparison_boxplots_R2_RMSE_MAPE.png")

# Fig S21 — mean R² with seed-level CI  [R2-5]
fig, ax = plt.subplots(figsize=(14, 6)); x = np.arange(len(ORD))
mu = agg14["R2_mean"].values
ax.bar(x, mu, color=PALETTE[:len(ORD)], alpha=0.8)
ax.errorbar(x, mu, yerr=[mu - agg14["R2_CI95_seedmean_lo"].values, agg14["R2_CI95_seedmean_hi"].values - mu],
            fmt="none", color="black", capsize=4, label="95% bootstrap CI of the 10-seed mean")
ax.set_xticks(x); ax.set_xticklabels(ORD, rotation=45, ha="right"); ax.set_ylabel("Mean test R² over 10 seeds")
ax.set_ylim(max(0, mu.min() - 0.05), 1.0); ax.legend()
plt.tight_layout(); savefig("FigS21_mean_r2_seedlevel_ci.png")

# Fig 2 / S11 — SHAP of the full stack
if "shap" in R42:
    phi = R42["shap"]["phi"]; Xex = R42["shap"]["X_ex"]; mabs = np.abs(phi).mean(0); so = np.argsort(mabs)[::-1]
    savetab(pd.DataFrame({"Feature": [FN[i] for i in so], "mean_abs_SHAP_kV": mabs[so]}).round(5), "Fig2_shap_stack_values.csv")
    if "shap_xgb_only" in R42:
        mx = np.abs(R42["shap_xgb_only"]).mean(0)
        savetab(pd.DataFrame({"Feature": FN, "stack": mabs, "xgboost_only": mx}).sort_values("stack", ascending=False).round(5),
                "Fig2_shap_stack_vs_xgboost_only.csv")
    top = so[:15]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(15), mabs[top][::-1], color=PALETTE[0], alpha=0.85)
    ax.set_yticks(range(15)); ax.set_yticklabels([FN[i] for i in top][::-1])
    ax.set(xlabel="Mean |SHAP| (kV)", title=f"SHAP of the full HyPhysML stack (seed 42, {len(Xex)} test points)")
    plt.tight_layout(); savefig("Fig2_shap_bar_full_stack.png")
    fig, ax = plt.subplots(figsize=(11, 8))
    for rank, fi in enumerate(so[:10][::-1]):
        v = Xex[:, fi]; vn = (v - v.min()) / (v.max() - v.min() + 1e-12)
        ax.scatter(phi[:, fi], rank + np.random.RandomState(fi).uniform(-.3, .3, len(v)), c=vn, cmap="coolwarm", s=10, alpha=.6)
    ax.set_yticks(range(10)); ax.set_yticklabels([FN[i] for i in so[:10][::-1]]); ax.axvline(0, color="k", ls="--")
    ax.set(xlabel="SHAP value (kV)  (colour: feature value, blue low - red high)", title="SHAP beeswarm, full stack")
    plt.tight_layout(); savefig("FigS11_shap_beeswarm_full_stack.png")

# Fig S10 — permutation importance (engineered + grouped raw)
pm = R42["perm"]; po = np.argsort(pm["mean"])[::-1][:15]
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
axes[0].barh(range(15), pm["mean"][po][::-1], xerr=pm["sd"][po][::-1], color=PALETTE[2], alpha=.8)
axes[0].set_yticks(range(15)); axes[0].set_yticklabels([FN[i] for i in po][::-1])
axes[0].set(xlabel="Mean decrease in R²", title="(a) Single engineered column permuted")
pr = R42["perm_raw"]
axes[1].barh(range(len(pr)), pr["mean_R2_drop"].values[::-1], xerr=pr["sd"].values[::-1], color=PALETTE[3], alpha=.8)
axes[1].set_yticks(range(len(pr))); axes[1].set_yticklabels(pr["Group"].values[::-1])
axes[1].set(xlabel="Mean decrease in R²", title="(b) Raw input permuted, all derived features rebuilt")
plt.tight_layout(); savefig("FigS10_permutation_importance_full_stack.png")
savetab(pd.DataFrame({"Feature": FN, "mean": pm["mean"], "sd": pm["sd"]}).sort_values("mean", ascending=False).round(6),
        "FigS10_perm_engineered.csv")
savetab(pr.round(6), "FigS10_perm_raw_grouped.csv")

# Fig S12 — meta weights mean ± sd  [R2-7]
fig, ax = plt.subplots(figsize=(10, 4.5)); cols = BASE_LEARNERS
ax.bar(range(len(cols)), mw[cols].mean(), yerr=mw[cols].std(), capsize=5, color=PALETTE[:len(cols)], alpha=.85)
for i, c in enumerate(cols):
    ax.scatter(np.full(len(mw), i) + np.random.RandomState(i).uniform(-.15, .15, len(mw)), mw[c], s=10, color="k", zorder=3)
    ax.text(i, mw[c].mean() + mw[c].std() + 0.01, f"{mw[c].mean():.3f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=20)
ax.set(ylabel="Ridge meta-weight", title=f"Meta-learner weights, mean ± s.d. over 10 seeds (intercept {mw['intercept'].mean():.3f} ± {mw['intercept'].std():.3f} kV)")
plt.tight_layout(); savefig("FigS12_meta_weights_10seeds.png")

# Fig S13 — auxiliary physics regression, no unsourced bands  [R2-2]
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, variant, title in zip(axes, ["with type indicators", "without type indicators"],
                              ["(a) with type indicators (CD, AD not identifiable)", "(b) without type indicators"]):
    d = ph_df[ph_df.Regression == variant].set_index("Coefficient").loc[PHY_NAMES]
    ax.barh(range(len(d)), d["mean"], xerr=d["sd"], color=["#2166AC" if v > 0 else "#D7191C" for v in d["mean"]], alpha=.85)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d.index); ax.axvline(0, color="k")
    ax.set(xlabel="β (log space), mean ± s.d. over 10 seeds", title=title)
plt.suptitle("Auxiliary log-linear Obenaus regression (separate from HyPhysML; does not constrain it)", fontweight="bold")
plt.tight_layout(); savefig("FigS13_physics_coefficients_auxiliary.png")

# Fig 3 — noise, raw propagated (main) and single-column (comparison)  [R2-3]
for mode, fname, ttl in [("raw_propagated", "Fig3_noise_raw_inputs_propagated.png",
                          "Noise added to the raw input; all derived features rebuilt"),
                         ("single_column", "FigS_noise_single_column_old_protocol.png",
                          "Single-column perturbation (old protocol): derived features keep the clean value")]:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, var in zip(axes, NOISE_FEATS):
        for nm, col in [("HyPhysML", PALETTE[0]), ("HyPhysML-MC", PALETTE[1])]:
            d = nz_sum[(nz_sum.Model == nm) & (nz_sum["mode"] == mode) & (nz_sum.Feature == var)]
            xs = [0] + d["Noise_pct"].tolist(); ys = [0] + d["dR2_mean"].tolist(); es = [0] + d["dR2_sd"].tolist()
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=col, label=nm)
        clip = nz_sum[(nz_sum.Model == "HyPhysML") & (nz_sum["mode"] == mode) & (nz_sum.Feature == var)]["pct_clipped"].max()
        ax.axhline(0, color="k", lw=.8)
        ax.set(xlabel=f"Noise s.d. (% of raw {var} s.d.)", ylabel="ΔR² vs clean test set",
               title=f"{var}" + (f"  (max clipped {clip:.1f}%)" if mode == "raw_propagated" else ""))
        ax.legend(fontsize=8)
    plt.suptitle(ttl + f" — mean ± s.d. over {len(SEEDS)} seeds × {N_NOISE_REP} repetitions", fontweight="bold")
    plt.tight_layout(); savefig(fname)

# Fig S19 — learning curve  [R2-4]
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(lc_sum["n_train"], lc_sum["R2_train_mean"], yerr=lc_sum["R2_train_sd"], marker="o", label="Training R²", capsize=3)
ax.errorbar(lc_sum["n_train"], lc_sum["R2_test_mean"], yerr=lc_sum["R2_test_sd"], marker="s", label="Test R² (fixed test set)", capsize=3)
ax.set(xlabel="Number of training records (stratified sub-sample of the 80% training set)", ylabel="R²",
       title=f"HyPhysML learning curve, seeds {LC_SEEDS}"); ax.legend()
plt.tight_layout(); savefig("FigS19_learning_curve_main_protocol.png")

# Fig S20 — raw-input response curves (replace PDP)  [R2-8] [R1-1]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
for ax, var in zip(axes, AUDIT_LEVELS):
    lv = AUDIT_LEVELS[var]
    for nm, col in [("HyPhysML", PALETTE[0]), ("HyPhysML-MC", PALETTE[1]), ("XGBoost", PALETTE[2])]:
        if nm == "HyPhysML" and R42["ice"].get(nm):
            for line in R42["ice"][nm][var]:
                ax.plot(lv, line, color=col, alpha=.07, lw=.8)
        cs = np.array([RES[s]["curves"][nm][var] for s in SEEDS])
        ax.errorbar(lv, cs.mean(0), yerr=cs.std(0), marker="o", color=col, label=nm, capsize=3)
    sub = df_raw if var != "J" else df_raw[df_raw.K > 0]
    ax.plot(lv, sub.groupby(var)["FOV"].mean().loc[lv].values, "k--", marker="x", label="measured (marginal mean)")
    ax.set(xlabel=var, ylabel="FOV (kV)", title=f"Response to {var}")
    if var == "SDD": ax.set_xscale("log")
axes[0].legend(fontsize=8)
plt.suptitle("Raw-input response curves: one raw variable set to each level for every test record, all features rebuilt",
             fontweight="bold")
plt.tight_layout(); savefig("FigS20_raw_input_response_curves.png")

# Monotonicity audit figure  [R2-2]
fig, ax = plt.subplots(figsize=(10, 4.5))
vars_ = list(AUDIT_LEVELS); w = 0.2
for k, nm in enumerate(["HyPhysML", "XGBoost", "HyPhysML-MC"]):
    vals = [au_sum[(au_sum.Model == nm) & (au_sum.Variable == v)]["pct_steps_increasing"].values[0] for v in vars_]
    ax.bar(np.arange(len(vars_)) + (k - 1.5) * w, vals, w, label=nm, color=PALETTE[[0, 2, 1][k]])
ax.bar(np.arange(len(vars_)) + 1.5 * w, [100 - _dm[v] for v in vars_], w, label="measured data", color="gray")
ax.set_xticks(np.arange(len(vars_))); ax.set_xticklabels(vars_)
ax.set(ylabel="% of adjacent level steps where FOV rises", title="Prediction-level physical-consistency audit (10 seeds)")
ax.legend(); plt.tight_layout(); savefig("FigS22_monotonicity_audit.png")

# Fig S15, S16, S17, S18
fig, ax = plt.subplots(figsize=(14, 5)); x = np.arange(len(ORD))
ax.bar(x - .2, agg14["R2_train_mean"], .4, label="Train R²"); ax.bar(x + .2, agg14["R2_mean"], .4, label="Test R²")
ax.set_xticks(x); ax.set_xticklabels(ORD, rotation=45, ha="right"); ax.set_ylim(0.6, 1.01); ax.legend()
plt.tight_layout(); savefig("FigS15_train_vs_test_r2.png")
tagg = agg14.sort_values("Time_fit_mean")
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(tagg)), tagg["Time_fit_mean"], yerr=tagg["Time_fit_std"], capsize=4, color=PALETTE[:len(tagg)])
ax.set_yscale("log"); ax.set_xticks(range(len(tagg))); ax.set_xticklabels(tagg.index, rotation=45, ha="right")
ax.set_ylabel("Training time per split (s, log scale)"); plt.tight_layout(); savefig("FigS16_computation_time.png")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, met in zip(axes, ["R2", "RMSE", "MAPE"]):
    g = pt.groupby("Type")[met]; ax.bar(g.mean().index, g.mean(), yerr=g.std(), capsize=4, color=PALETTE[:4])
    ax.set_title(f"{met} by insulator type (mean ± s.d., 10 seeds)")
    if met == "R2": ax.set_ylim(0.97, 1.0)
plt.tight_layout(); savefig("FigS17_per_type_performance.png")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(resid_all, bins=60, color=PALETTE[0]); axes[0].set(xlabel="Residual (kV)", title=f"Pooled residuals, 10 seeds (n={len(resid_all)})")
axes[1].scatter(np.concatenate([RES[s]["preds"]["HyPhysML"] for s in SEEDS]), resid_all, s=2, alpha=.2)
axes[1].axhline(0, color="r"); axes[1].set(xlabel="Predicted (kV)", ylabel="Residual (kV)")
probplot(resid_all, plot=axes[2]); plt.tight_layout(); savefig("FigS18_residuals.png")

# extrapolation figure  [R1-3]
if os.path.isfile(extra_ck):
    ex = pickle.load(open(extra_ck, "rb"))
    facs = [f for f in ["SDD", "RH", "Aging", "Sample"] if f in ex.Factor.unique()]
    fig, axes = plt.subplots(1, len(facs), figsize=(5 * len(facs), 4.5), squeeze=False); axes = axes[0]
    ref = ex[ex.Factor.str.startswith("random")].set_index("Model")["RMSE"]
    for ax, fac in zip(axes, facs):
        d = ex[ex.Factor == fac]
        for k, nm in enumerate(["HyPhysML", "HyPhysML-MC", "XGBoost", "Obenaus"]):
            dd = d[d.Model == nm]
            ax.plot([str(v) for v in dd.Held_out_level], dd.RMSE, marker="o", color=PALETTE[[0, 1, 2, 4][k]], label=nm)
            ax.axhline(ref[nm], color=PALETTE[[0, 1, 2, 4][k]], ls=":", lw=1)
        ax.set(xlabel=f"held-out {fac} level", ylabel="RMSE (kV)", title=f"Leave one {fac} level out")
    axes[0].legend(fontsize=8)
    plt.suptitle("Extrapolation to an unseen factor level (dotted: random 80/20 split, same settings)", fontweight="bold")
    plt.tight_layout(); savefig("FigS23_extrapolation_leave_one_level_out.png")

# ════════════════════════════════════════════════════════════════════════
# §14  KEY NUMBERS FOR THE MANUSCRIPT + EXCEL WORKBOOK
# ════════════════════════════════════════════════════════════════════════
section("§14  Key numbers")
h = agg.loc["HyPhysML"]; xg = agg.loc["XGBoost"]; ob = agg.loc["Obenaus"]
sx = st_df.set_index("vs")
L = []
L.append("# HyPhysML revision — key numbers for the manuscript\n")
L.append(f"Seeds: {SEEDS}; test n per seed: {sorted(set(RES[s]['n_test'] for s in SEEDS))}; N_OPTUNA={N_OPTUNA} (nested, per seed)\n")
L.append("## Table 2 (10-seed means)\n")
L.append(agg14[["R2_mean", "R2_std", "RMSE_mean", "MAE_mean", "MAPE_mean", "NSE_mean"]].round(4).to_string())
L.append(f"\nBest of 14: {BEST}. HyPhysML R²={h.R2_mean:.4f} ± {h.R2_std:.4f}, RMSE={h.RMSE_mean:.3f} kV, MAPE={h.MAPE_mean:.2f}%")
L.append(f"RMSE reduction vs Obenaus: {100 * (1 - h.RMSE_mean / ob.RMSE_mean):.1f}%;  ΔRMSE vs XGBoost: {xg.RMSE_mean - h.RMSE_mean:.3f} kV")
L.append(f"Seed-level 95% CI of the mean R² (HyPhysML): [{h.R2_CI95_seedmean_lo:.4f}, {h.R2_CI95_seedmean_hi:.4f}]")
L.append(f"Seed-42 split: R²={h.R2_seed42:.4f}, observation-bootstrap 95% CI [{h.R2_CI95_seed42_obs_lo:.4f}, {h.R2_CI95_seed42_obs_hi:.4f}], n={len(yte42)}")
L.append(f"Friedman chi2({len(MODEL_NAMES) - 1}) = {fr:.2f}, P = {fp:.2e}")
L.append("\n## Table 3\n" + st_df.round(6).to_string(index=False))
L.append("\n## Meta-weights (mean, sd)\n" + mw_sum.round(4).to_string())
L.append("\n## Auxiliary physics regression\n" + ph_df.round(4).to_string(index=False))
L.append("\n## Monotonicity audit (% of adjacent steps where predicted FOV rises)\n" + au_sum.round(3).to_string(index=False))
L.append("\n## Noise (HyPhysML)\n" + nz_sum[nz_sum.Model == "HyPhysML"].round(5).to_string(index=False))
L.append("\n## Learning curve\n" + lc_sum.round(5).to_string(index=False))
if os.path.isfile(lcdiag_ck):
    L.append("\n## Learning-curve diagnostic\n" + pickle.load(open(lcdiag_ck, "rb")).round(4).to_string(index=False))
L.append("\n## Ablation\n" + abl.round(5).to_string())
L.append("\n## Per type (mean over seeds)\n" + pt_sum.round(4).to_string())
L.append("\n## Residuals\n" + res_tab.round(4).to_string(index=False))
L.append("\n## Data-level physics\n" + _mono_data.to_string(index=False) + "\n\n" + _exp_sum.to_string())
if os.path.isfile(extra_ck):
    L.append("\n## Extrapolation\n" + pickle.load(open(extra_ck, "rb")).round(4).to_string(index=False))
L.append("\n## HPO/test overlap of the OLD protocol\n" + _ov_df.to_string(index=False))
if "shap" in R42:
    L.append(f"\nSHAP (full stack) methods: {R42['shap']['method']}; additivity error {R42['shap']['additivity_err']:.4f} kV")
with open(os.path.join(OUT_DIR, "revision_key_numbers.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("\n".join(L[:12]))

with pd.ExcelWriter(os.path.join(OUT_DIR, "revision_all_tables.xlsx"), engine="openpyxl") as wr:
    for fn in sorted(os.listdir(TAB_DIR)):
        if fn.endswith(".csv"):
            pd.read_csv(os.path.join(TAB_DIR, fn)).to_excel(wr, sheet_name=fn[:-4][:31], index=False)
print(f"\n  Done. Figures: {FIG_DIR}\n  Tables: {TAB_DIR}\n  Summary: revision_key_numbers.md")
